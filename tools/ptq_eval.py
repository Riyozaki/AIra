"""Strong size baseline: post-training quantization of a trained fp model (no retraining).

Block linears -> symmetric absmax groupwise int-b (group 32 along input dim), like llama.cpp Q8_0/Q4_0.
Storage incl. one fp16 scale per group: b + 16/32 bits/weight. Embedding/head stay fp (same as ternary runs).
Usage: .venv/bin/python tools/ptq_eval.py ckpt.npz --d 128 --layers 4 --bits 4
"""
import argparse, json, os, sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(__file__))
import lm


def qgroup(w, bits, g=32):
    K, M = w.shape
    wg = w.reshape(K // g, g, M)
    qmax = 2 ** (bits - 1) - 1
    s = np.abs(wg).max(1, keepdims=True) / qmax + 1e-12
    s = s.astype(np.float16).astype(np.float32)
    return (np.clip(np.round(wg / s), -qmax - 1, qmax) * s).reshape(K, M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--d", type=int, required=True); ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--bits", type=int, nargs="+", default=[8, 4, 3]); ap.add_argument("--eval-bytes", type=int, default=1_000_000)
    ap.add_argument("--log", default="results/runs/ptq.jsonl"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    z = np.load(a.ckpt)
    p = lm.init(jax.random.PRNGKey(0), a.d, a.layers, a.heads)
    flat, tdef = jax.tree_util.tree_flatten_with_path(p)
    p = jax.tree_util.tree_unflatten(tdef, [jnp.asarray(z[jax.tree_util.keystr(k)]) for k, _ in flat])
    full = np.memmap(os.path.join(lm.RAW, "enwik8"), dtype=np.uint8, mode="r")
    ev = np.asarray(full[90_000_000: 90_000_000 + a.eval_bytes])
    for bits in [None] + a.bits:
        q = jax.tree_util.tree_map(lambda x: x, p)
        if bits:
            q["blocks"] = [{k: (jnp.asarray(qgroup(np.asarray(v), bits)) if k in ("qkv", "o", "up", "down") else v)
                            for k, v in b.items()} for b in p["blocks"]]
        bpb, n = lm.evaluate(q, ev, a.T, a.heads, "none")
        r = dict(tag=a.tag, ckpt=a.ckpt, d=a.d, L=a.layers, ptq_bits=bits or 32,
                 bits_per_lin_weight=(bits + 0.5) if bits else 16, bpb=round(bpb, 5), eval_bytes=n)
        print(json.dumps(r))
        with open(a.log, "a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
