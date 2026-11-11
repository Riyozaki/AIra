"""B-NN / ternary byte-level transformer LM on enwik8 (JAX, CPU).

Architecture (baseline, modern defaults): pre-norm RMSNorm, RoPE, causal MHA, ReLU^2 MLP (4x),
no biases, untied fp embedding / head. --quant ternary: every block Linear becomes BitNet-b1.58
(absmean ternary weights + per-token int8 absmax activations, STE); embedding/head stay fp.

Eval: sliding windows (len T, stride T/2); only the last T/2 positions of each window are scored,
so every scored byte has >= T/2 context (first window scores all). Reports bits per byte.

Example:
  .venv/bin/python tools/lm.py --d 256 --layers 4 --quant none --lr 3e-3 --minutes 10 --seed 0
"""
import argparse, json, math, os, time
import numpy as np
import jax, jax.numpy as jnp
import optax
import sys
sys.path.insert(0, os.path.dirname(__file__))
import ngram_kn as ng

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


# ---------------- quantizers ----------------
def ste(x, q):
    return x + jax.lax.stop_gradient(q - x)


def ternary_w(w):
    a = jnp.mean(jnp.abs(w)) + 1e-8
    return ste(w, jnp.clip(jnp.round(w / a), -1, 1) * a)


def int8_act(x):
    s = jnp.max(jnp.abs(x), axis=-1, keepdims=True) / 127.0 + 1e-8
    return ste(x, jnp.clip(jnp.round(x / s), -128, 127) * s)


def linear(x, w, quant):
    if quant == "ternary":
        return int8_act(x) @ ternary_w(w)
    return x @ w


# ---------------- model ----------------
def rmsnorm(x, g):
    return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6) * g


def rope(x):  # x: B,T,H,Dh
    T, Dh = x.shape[1], x.shape[-1]
    inv = 1.0 / (10000 ** (jnp.arange(0, Dh, 2) / Dh))
    ang = jnp.arange(T)[:, None] * inv[None]
    c, s = jnp.cos(ang)[None, :, None], jnp.sin(ang)[None, :, None]
    x1, x2 = x[..., ::2], x[..., 1::2]
    return jnp.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).reshape(x.shape)


def init(key, d, L, heads, V=256):
    ks = jax.random.split(key, 2 + 4 * L)
    std = 0.02
    p = {"emb": jax.random.normal(ks[0], (V, d)) * std,
         "head": jax.random.normal(ks[1], (d, V)) * std,
         "gf": jnp.ones(d), "blocks": []}
    for l in range(L):
        k = ks[2 + 4 * l: 6 + 4 * l]
        p["blocks"].append({
            "g1": jnp.ones(d), "g2": jnp.ones(d),
            "qkv": jax.random.normal(k[0], (d, 3 * d)) * std,
            "o": jax.random.normal(k[1], (d, d)) * std / math.sqrt(2 * L),
            "up": jax.random.normal(k[2], (d, 4 * d)) * std,
            "down": jax.random.normal(k[3], (4 * d, d)) * std / math.sqrt(2 * L)})
    return p


def forward(p, x, heads, quant, lp=None):
    B, T = x.shape
    h = p["emb"][x]
    d = h.shape[-1]
    Dh = d // heads
    mask = jnp.tril(jnp.ones((T, T), bool))
    for b in p["blocks"]:
        a = rmsnorm(h, b["g1"])
        qkv = linear(a, b["qkv"], quant).reshape(B, T, 3, heads, Dh)
        q, k, v = rope(qkv[:, :, 0]), rope(qkv[:, :, 1]), qkv[:, :, 2]
        att = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(Dh)
        att = jax.nn.softmax(jnp.where(mask, att, -1e30), -1)
        o = jnp.einsum("bhts,bshd->bthd", att, v).reshape(B, T, d)
        h = h + linear(o, b["o"], quant)
        m = rmsnorm(h, b["g2"])
        h = h + linear(jnp.square(jax.nn.relu(linear(m, b["up"], quant))), b["down"], quant)
    out = rmsnorm(h, p["gf"]) @ p["head"]
    return out if lp is None else out + lp  # residual over n-gram log-probs (docs/04 model R)


def nll(p, x, y, heads, quant, lp=None):  # mean nats over all positions
    lg = forward(p, x, heads, quant, lp)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(lg, y))


def count(p):
    emb = p["emb"].size + p["head"].size
    lin = sum(b[k].size for b in p["blocks"] for k in ("qkv", "o", "up", "down"))
    other = sum(x.size for x in jax.tree_util.tree_leaves(p)) - emb - lin
    return emb, lin, other


# ---------------- data ----------------
def ng_logp(full, orders, starts, T):
    """log p_ng for windows full[s:s+T] -> (len(starts), T, 256) float32 (context = K bytes ending at each pos)"""
    K = len(orders) - 1
    ctx = np.concatenate([np.lib.stride_tricks.sliding_window_view(np.asarray(full[s - K + 1:s + T]), K)
                          for s in starts])
    return np.log(ng.dense(orders, ctx)).astype(np.float32).reshape(len(starts), T, 256)


def batches(data, B, T, rng, lo=0, hi=None, orders=None):
    hi = len(data) if hi is None else hi
    while True:
        ix = rng.integers(lo, hi - T - 1, B)
        xy = np.stack([data[i:i + T + 1] for i in ix]).astype(np.int32)
        if orders is None:
            yield xy[:, :-1], xy[:, 1:], None
        else:
            yield xy[:, :-1], xy[:, 1:], ng_logp(data, orders, ix, T)


def evaluate(p, data, T, heads, quant, B=32, lpfn=None):
    """sliding-window bpb on `data` (1-D uint8). lpfn(starts)->(B,T,256) n-gram log-probs (relative starts)."""
    S = T // 2

    @jax.jit
    def tok_nll(p, x, y, lp):
        lg = forward(p, x, heads, quant, lp)
        return optax.softmax_cross_entropy_with_integer_labels(lg, y)

    starts = list(range(0, len(data) - T - 1 + 1, S))
    tot, n = 0.0, 0
    for i in range(0, len(starts), B):
        st = starts[i:i + B]
        pad = B - len(st)
        st_p = st + [st[-1]] * pad
        xy = np.stack([data[s:s + T + 1] for s in st_p]).astype(np.int32)
        lp = None if lpfn is None else lpfn(st_p)
        l = np.asarray(tok_nll(p, xy[:, :-1], xy[:, 1:], lp))[: len(st)]
        for j, s in enumerate(st):
            seg = l[j] if s == 0 else l[j, S:]
            tot += float(seg.sum()); n += seg.size
    return tot / n / math.log(2), n


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--quant", default="none", choices=["none", "ternary"])
    ap.add_argument("--opt", default="adamw", choices=["adamw"])
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=0, help="fixed step budget (preferred for fairness)")
    ap.add_argument("--minutes", type=float, default=0, help="wall-clock budget, used only if --steps=0")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-bytes", type=int, default=1_000_000, help="prefix of valid used (pilot)")
    ap.add_argument("--split", default="valid", choices=["valid", "test"])
    ap.add_argument("--log", default="results/runs/lm.jsonl")
    ap.add_argument("--tag", default="")
    ap.add_argument("--probe", action="store_true", help="only measure step time")
    ap.add_argument("--save", default=None, help="save final params (.npz) for PTQ eval")
    ap.add_argument("--ngram", type=int, default=0, help="K>0: residual hybrid over mKN (docs/04 model R)")
    ap.add_argument("--ng-bytes", type=int, default=80_000_000, help="table built on train[:ng_bytes]; NN trained on train[ng_bytes:90M]")
    a = ap.parse_args()

    full = np.memmap(os.path.join(RAW, "enwik8"), dtype=np.uint8, mode="r")
    train = full[:90_000_000]
    off = {"valid": 90_000_000, "test": 95_000_000}[a.split]
    evdata = np.asarray(full[off: off + a.eval_bytes])

    p = init(jax.random.PRNGKey(a.seed), a.d, a.layers, a.heads)
    emb, lin, other = count(p)
    orders, tab_s, lo, lpfn, n_entries = None, 0.0, 0, None, 0
    if a.ngram > 0:
        t0 = time.time()
        orders = ng.build(np.asarray(full[:a.ng_bytes]), a.ngram)
        tab_s = time.time() - t0
        n_entries = int(sum(len(o.keys) for o in orders))
        lo = a.ng_bytes  # cross-fitting (docs/04 §4)
        p["head"] = jnp.zeros_like(p["head"])  # f == 0 at init -> p = p_ng exactly
        lpfn = lambda st: ng_logp(full, orders, [off + s for s in st], a.T)
    if a.steps <= 0 and not a.probe:
        assert a.minutes > 0
    total_steps = a.steps if a.steps > 0 else 10**9
    sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, a.warmup, max(total_steps, a.warmup + 1), a.lr * 0.1) \
        if a.steps > 0 else optax.linear_schedule(0.0, a.lr, a.warmup)
    # no weight decay on gains (1-D)
    wd_mask = jax.tree_util.tree_map(lambda x: x.ndim >= 2, p)
    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.adamw(sched, b1=0.9, b2=0.95, weight_decay=a.wd, mask=wd_mask))
    st = opt.init(p)

    @jax.jit
    def step(p, st, x, y, lp):
        l, g = jax.value_and_grad(nll)(p, x, y, a.heads, a.quant, lp)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    rng = np.random.default_rng(a.seed)
    it = batches(train, a.B, a.T, rng, lo=lo, orders=orders)
    x, y, lp = next(it)
    t0 = time.time(); p, st, l = step(p, st, x, y, lp); l.block_until_ready(); tcomp = time.time() - t0
    if a.probe:
        t0 = time.time()
        for _ in range(5):
            x, y, lp = next(it); p, st, l = step(p, st, x, y, lp)
        l.block_until_ready()
        dt = (time.time() - t0) / 5
        print(json.dumps(dict(d=a.d, L=a.layers, quant=a.quant, N_emb=emb, N_lin=lin,
                              step_s=round(dt, 3), tok_per_s=round(a.B * a.T / dt), compile_s=round(tcomp, 1))))
        return
    t0 = time.time(); s = 1; hist = []
    while s < total_steps:
        if a.steps <= 0 and time.time() - t0 > a.minutes * 60:
            break
        x, y, lp = next(it); p, st, l = step(p, st, x, y, lp); s += 1
        if s % 100 == 0:
            hist.append((s, round(float(l) / math.log(2), 4)))
    train_s = time.time() - t0
    te = time.time()
    bpb, n = evaluate(p, evdata, a.T, a.heads, a.quant, lpfn=lpfn)
    res = dict(tag=a.tag, model="transformer", quant=a.quant, d=a.d, L=a.layers, heads=a.heads, T=a.T, B=a.B,
               N_emb=emb, N_lin=lin, N_other=other, lr=a.lr, wd=a.wd, seed=a.seed, steps=s,
               tokens=s * a.B * a.T, train_s=round(train_s, 1), eval_s=round(time.time() - te, 1),
               split=a.split, eval_bytes=n, bpb=round(bpb, 5), ngram_K=a.ngram,
               ng_bytes=a.ng_bytes if a.ngram else 0, table_s=round(tab_s, 1), table_entries=n_entries,
               train_lo=lo, train_curve=hist[-5:],
               jax=jax.__version__)
    print(json.dumps(res))
    if a.save:
        flat, _ = jax.tree_util.tree_flatten_with_path(p)
        np.savez(a.save, **{jax.tree_util.keystr(k): np.asarray(v) for k, v in flat})
    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    with open(a.log, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
