#!/usr/bin/env python3
"""Профиль одного шага обучения nano-EMA(+ADR): где уходит время.
Декомпозиция: forward-тело | голова+CE | backward-тело | backward-голова | оптимизатор."""
import os, sys, json, time, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nano_lc import NanoGPT, AdamW, MuonW, softmax_ce

ROOT = os.path.dirname(os.path.abspath(__file__))
tr = np.load(f"{ROOT}/data/prep/train.npy").astype(np.int64)
V = 8000
m = NanoGPT(V, kind="ema", adr_kf=0.5)
opt = MuonW(m.p)
rng = np.random.default_rng(0)
B, T = 24, 96

def batch():
    st = rng.integers(0, len(tr) - T - 1, size=B)
    return np.stack([tr[s:s+T] for s in st]), np.stack([tr[s+1:s+T+1] for s in st])

# прогрев
for _ in range(3):
    x, y = batch(); H, c = m.forward(x); lg = m.logits(H)
    loss, dz = softmax_ce(lg, y); m.backward(H, c, dz); opt.step(6e-4)

acc = dict(fwd=0.0, head=0.0, bwd_head=0.0, bwd_body=0.0, opt=0.0)
N = 12
for _ in range(N):
    x, y = batch()
    t0 = time.perf_counter(); H, c = m.forward(x); t1 = time.perf_counter()
    lg = m.logits(H); loss, dz = softmax_ce(lg, y); t2 = time.perf_counter()
    # backward головы вручную (как в NanoGPT.backward до layernorm_bwd)
    m.p.g["E"] += dz.reshape(-1, V).T @ H.reshape(-1, m.D)
    dH = dz @ m.p.d["E"]; t3 = time.perf_counter()
    dH2, dg, db = __import__("nano_lc").layernorm_bwd(c[3], dH)
    m.p.g["lnfg"] += dg; m.p.g["lnfb"] += db
    for i in reversed(range(m.L)):
        bc = c[2][i]; b = m.blocks[i]
        if b["routed"]:
            Xin, s, idx, gsel, k, arm_c = bc["route"]
            bi = np.arange(B)[:, None]
            dH_sel = dH2[bi, idx]
            delta = arm_c["delta"]
            dg_ = (dH_sel * delta).sum(-1); ds_ = dg_ * gsel * (1 - gsel)
            m.p.g[f"b{i}.rw"] += (ds_[..., None] * Xin[bi, idx]).sum((0, 1))
            m.p.g[f"b{i}.rb"] += ds_.sum()
            dx_arm = m._arm_backward(arm_c, dH_sel * gsel[..., None], i, b)
            dH2[bi, idx] = dH_sel + dx_arm
            dH2[bi, idx] += ds_[..., None] * m.p.d[f"b{i}.rw"][None, None, :]
        else:
            dX = m._arm_backward(bc["arm"], dH2, i, b)
            dH2 = dH2 + dX
    np.add.at(m.p.g["E"], x, dH2); m.p.g["pos"] += dH2.sum(0)
    t4 = time.perf_counter()
    opt.step(6e-4); t5 = time.perf_counter()
    acc["fwd"] += t1-t0; acc["head"] += t2-t1; acc["bwd_head"] += t3-t2
    acc["bwd_body"] += t4-t3; acc["opt"] += t5-t4

tot = sum(acc.values())
print(json.dumps({k: round(v/N*1000, 2) for k, v in acc.items()}, indent=1))
print(f"итого мс/шаг: {tot/N*1000:.2f}; ток/с экв.: {B*T/(tot/N):.0f}")
print("доли: " + ", ".join(f"{k}={v/tot*100:.1f}%" for k, v in acc.items()))
