#!/usr/bin/env python3
"""QAT на собственном nano-фреймворке: STE + shadow-fp32.
Каждый шаг: матричные веса подменяются tern(W) (честный квантованный форвард),
градиенты (вычисленные сквозь квантованный форвард) применяются к теневым fp32-весам.
Запуск: python3 np_qat.py ckpt.npz ema 0.5 300"""
import os, sys, json, math, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nano_lc
from nano_lc import NanoGPT, AdamW, MuonW, softmax_ce, layernorm

ROOT = os.path.dirname(os.path.abspath(__file__))
PATS = (".fc1", ".fc2", ".Wm", ".qkv", ".Wqkv", ".Wo")

def tern(W):
    s = np.abs(W).mean(-1, keepdims=True).clip(1e-5)
    return (np.clip(np.rint(W / s), -1, 1) * s).astype(W.dtype)

ckpt, kind = sys.argv[1], sys.argv[2]
adr = None if sys.argv[3] == "none" else float(sys.argv[3])
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 300
if "--actq8" in sys.argv:
    nano_lc.ACTQ8 = True
    print("activation-int8 QAT включён", flush=True)
tag = os.path.basename(ckpt).replace("ckpt_", "").replace(".npz", "") + f"_qat{steps}"

d = dict(np.load(ckpt))
m = NanoGPT(8000, kind=kind, adr_kf=adr)
for k in m.p.d:
    if k in d: m.p.d[k][...] = np.asarray(d[k], dtype=m.p.d[k].dtype)
targets = [k for k in m.p.d if any(p in k for p in PATS)]
shadow = {k: m.p.d[k].copy() for k in targets}
print(f"QAT targets: {len(targets)} matrices", flush=True)

tr = np.load(f"{ROOT}/data/prep/train.npy").astype(np.int64)
va = np.load(f"{ROOT}/data/prep/val.npy").astype(np.int64)
rng = np.random.default_rng(123)
opt = MuonW(m.p, lr=1.5e-4, mulr=0.0075) if "--muon" in sys.argv else AdamW(m.p, lr=1.5e-4)
teacher = None
if "--kl" in sys.argv:                       # Gemma-QAT рецепт: KL к fp32-исходнику
    teacher = NanoGPT(8000, kind=kind, adr_kf=adr)
    for k in teacher.p.d:
        if k in d: teacher.p.d[k][...] = np.asarray(d[k], dtype=teacher.p.d[k].dtype)
    print("QAT-KL: цель = fp32-учитель (T=2)", flush=True)

def batch(arr, rr, B=24, T=96):
    st = rr.integers(0, len(arr) - T - 1, size=B)
    return (np.stack([arr[s:s+T] for s in st]), np.stack([arr[s+1:s+T+1] for s in st]))

def apply_q():
    for k in targets:
        m.p.d[k][...] = tern(shadow[k])

def vloss(iters=6):
    rr = np.random.default_rng(1234); tot = 0.0
    for _ in range(iters):
        x, y = batch(va, rr)
        H, _ = m.forward(x); l, _ = softmax_ce(m.logits(H), y); tot += l
    return tot / iters

apply_q()
print(json.dumps(dict(step=-1, val_loss=round(float(vloss()), 4), note="zero-shot-quantized")), flush=True)
logf = open(f"{ROOT}/results/run_{tag}.jsonl", "w")
t0 = time.time()
for step in range(steps):
    x, y = batch(tr, rng)
    apply_q()
    H, c = m.forward(x); loss, dz = softmax_ce(m.logits(H), y)
    if teacher is not None:
        Ht, _ = teacher.forward(x)
        lgt = Ht @ teacher.p.d["E"].T
        zt = lgt / 2.0; q = np.exp(zt - zt.max(-1, keepdims=True)); q /= q.sum(-1, keepdims=True)
        zs = m.logits(H) / 2.0; ps = np.exp(zs - zs.max(-1, keepdims=True)); ps /= ps.sum(-1, keepdims=True)
        n = lg_n = y.size
        kl = float((q * (np.log(q + 1e-12) - np.log(ps + 1e-12))).sum() / n)
        dz = dz + 1.0 * 4.0 * (ps - q)
        loss = loss + kl
    m.backward(H, c, dz)
    opt.step(1.5e-4)
    for k in targets:
        shadow[k] += m.p.d[k] - tern(shadow[k])
        m.p.d[k][...] = shadow[k]
    if step % 50 == 0 or step == steps - 1:
        apply_q()
        vl = float(vloss())
        rec = dict(step=step, train_loss=round(float(loss), 4), val_loss=round(vl, 4),
                   val_ppl=round(math.exp(vl), 2), wall=round(time.time() - t0, 1))
        print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
apply_q()
np.savez(f"{ROOT}/results/ckpt_{tag}.npz", **m.p.d)
print("SUMMARY " + json.dumps(dict(tag=tag, params=sum(v.size for v in m.p.d.values()),
      final_val_ppl=round(float(math.exp(vloss(10))), 2), wall_s=round(time.time() - t0, 1))), flush=True)
