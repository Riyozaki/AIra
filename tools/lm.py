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


def forward(p, x, heads, quant):
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
    return rmsnorm(h, p["gf"]) @ p["head"]


def nll(p, x, y, heads, quant):  # mean nats over all positions
    lg = forward(p, x, heads, quant)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(lg, y))


def count(p):
    emb = p["emb"].size + p["head"].size
    lin = sum(b[k].size for b in p["blocks"] for k in ("qkv", "o", "up", "down"))
    other = sum(x.size for x in jax.tree_util.tree_leaves(p)) - emb - lin
    return emb, lin, other


# ---------------- data ----------------
def batches(data, B, T, rng):
    while True:
        ix = rng.integers(0, len(data) - T - 1, B)
        xy = np.stack([data[i:i + T + 1] for i in ix]).astype(np.int32)
        yield xy[:, :-1], xy[:, 1:]


def evaluate(p, data, T, heads, quant, B=32):
    """sliding-window bpb on `data` (1-D uint8)."""
    S = T // 2

    @jax.jit
    def tok_nll(p, x, y):
        lg = forward(p, x, heads, quant)
        return optax.softmax_cross_entropy_with_integer_labels(lg, y)

    starts = list(range(0, len(data) - T - 1 + 1, S))
    tot, n = 0.0, 0
    for i in range(0, len(starts), B):
        st = starts[i:i + B]
        pad = B - len(st)
        st_p = st + [st[-1]] * pad
        xy = np.stack([data[s:s + T + 1] for s in st_p]).astype(np.int32)
        l = np.asarray(tok_nll(p, xy[:, :-1], xy[:, 1:]))[: len(st)]
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
    a = ap.parse_args()

    full = np.memmap(os.path.join(RAW, "enwik8"), dtype=np.uint8, mode="r")
    train = full[:90_000_000]
    off = {"valid": 90_000_000, "test": 95_000_000}[a.split]
    evdata = np.asarray(full[off: off + a.eval_bytes])

    p = init(jax.random.PRNGKey(a.seed), a.d, a.layers, a.heads)
    emb, lin, other = count(p)
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
    def step(p, st, x, y):
        l, g = jax.value_and_grad(nll)(p, x, y, a.heads, a.quant)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    rng = np.random.default_rng(a.seed)
    it = batches(train, a.B, a.T, rng)
    x, y = next(it)
    t0 = time.time(); p, st, l = step(p, st, x, y); l.block_until_ready(); tcomp = time.time() - t0
    if a.probe:
        t0 = time.time()
        for _ in range(5):
            x, y = next(it); p, st, l = step(p, st, x, y)
        l.block_until_ready()
        dt = (time.time() - t0) / 5
        print(json.dumps(dict(d=a.d, L=a.layers, quant=a.quant, N_emb=emb, N_lin=lin,
                              step_s=round(dt, 3), tok_per_s=round(a.B * a.T / dt), compile_s=round(tcomp, 1))))
        return
    t0 = time.time(); s = 1; hist = []
    while s < total_steps:
        if a.steps <= 0 and time.time() - t0 > a.minutes * 60:
            break
        x, y = next(it); p, st, l = step(p, st, x, y); s += 1
        if s % 100 == 0:
            hist.append((s, round(float(l) / math.log(2), 4)))
    train_s = time.time() - t0
    te = time.time()
    bpb, n = evaluate(p, evdata, a.T, a.heads, a.quant)
    res = dict(tag=a.tag, model="transformer", quant=a.quant, d=a.d, L=a.layers, heads=a.heads, T=a.T, B=a.B,
               N_emb=emb, N_lin=lin, N_other=other, lr=a.lr, wd=a.wd, seed=a.seed, steps=s,
               tokens=s * a.B * a.T, train_s=round(train_s, 1), eval_s=round(time.time() - te, 1),
               split=a.split, eval_bytes=n, bpb=round(bpb, 5), train_curve=hist[-5:],
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
