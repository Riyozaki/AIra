"""docs/04 model M: p = lam * p_ng + (1 - lam) * p_nn, NN = separately trained B-NN checkpoint.

lam is chosen on valid bytes [500k, 1M) and reported on valid bytes [1, 499968] -- the same scored positions
as every other pilot number (lm.evaluate with --eval-bytes 500000). Test is not touched.
Usage: .venv/bin/python tools/mix_eval.py results/ckpt/fp_d128_s0.npz --d 128 --layers 4 --K 7
"""
import argparse, json, math, os, sys
import numpy as np, jax, jax.numpy as jnp, optax
sys.path.insert(0, os.path.dirname(__file__))
import lm, ngram_kn as ng


def per_byte_nll(p, data, T, heads):
    """same windows as lm.evaluate; returns nats for target positions 1..len-? in order."""
    S = T // 2

    @jax.jit
    def f(p, x, y):
        return optax.softmax_cross_entropy_with_integer_labels(lm.forward(p, x, heads, "none"), y)

    starts = list(range(0, len(data) - T - 1 + 1, S))
    out = []
    for i in range(0, len(starts), 32):
        st = starts[i:i + 32]
        st_p = st + [st[-1]] * (32 - len(st))
        xy = np.stack([data[s:s + T + 1] for s in st_p]).astype(np.int32)
        l = np.asarray(f(p, xy[:, :-1], xy[:, 1:]))[: len(st)]
        for j, s in enumerate(st):
            out.append(l[j] if s == 0 else l[j, S:])
    return np.concatenate(out)  # entry i corresponds to data position i+1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--d", type=int, required=True); ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--K", type=int, default=7); ap.add_argument("--ng-bytes", type=int, default=80_000_000)
    ap.add_argument("--log", default="results/runs/pilot_ngram.jsonl")
    a = ap.parse_args()
    z = np.load(a.ckpt)
    p = lm.init(jax.random.PRNGKey(0), a.d, a.layers, a.heads)
    flat, tdef = jax.tree_util.tree_flatten_with_path(p)
    p = jax.tree_util.tree_unflatten(tdef, [jnp.asarray(z[jax.tree_util.keystr(k)]) for k, _ in flat])
    full = np.fromfile(os.path.join(lm.RAW, "enwik8"), dtype=np.uint8)
    off = 90_000_000
    ev = full[off: off + 1_000_000]
    nn = per_byte_nll(p, ev, a.T, a.heads) / math.log(2)          # bits, positions 1..
    orders = ng.build(full[: a.ng_bytes], a.K)
    ngb = -ng.score(orders, full[off - a.K: off + 1_000_000], a.K)  # bits, positions 0..
    ngb = ngb[1: 1 + len(nn)]
    pos = np.arange(1, 1 + len(nn))
    rep, tune = pos <= 499968, pos >= 500_000
    lams = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    def mix(l, m):
        return -np.log2(l * 2.0 ** -ngb[m] + (1 - l) * 2.0 ** -nn[m]).mean()
    tune_bpb = {float(l): float(mix(l, tune)) for l in lams}
    lam = min(tune_bpb, key=tune_bpb.get)
    res = dict(tag="pilot_ngM", model="mixture", ckpt=a.ckpt, d=a.d, K=a.K, ng_bytes=a.ng_bytes, lam=lam,
               bpb=round(float(mix(lam, rep)), 5), bpb_nn=round(float(nn[rep].mean()), 5),
               bpb_ng=round(float(ngb[rep].mean()), 5), eval_positions=int(rep.sum()), tune_curve=tune_bpb)
    print(json.dumps(res))
    with open(a.log, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
