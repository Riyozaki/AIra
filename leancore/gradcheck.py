#!/usr/bin/env python3
"""gradcheck.py — конечно-разностная верификация всех backward-выводов nano_lc.
float64, ε=1e-6; шумовая полка: пропуск записей |num|+|ana|<1e-5; порог rel-err 5e-3.
Ожидание (уже было достигнуто в сессии): все опы ≤ 2e-8, сквозные модели ≤ 1e-4."""
import sys, numpy as np
sys.path.insert(0, '.')
from nano_lc import (gelu, gelu_bwd, layernorm, layernorm_bwd, linear, linear_bwd,
                     attention, attention_bwd, ema_mix, ema_mix_bwd, softmax_ce, NanoGPT,
                     delta_mix, delta_mix_bwd)

abs_tol = 1e-8
rel_tol = 5e-3
results = []

def fd_check(name, fwd, bwd_ana, shapes, nsample=40, eps=1e-6):
    """fwd(*xs)->(y,cache); bwd как в nano. Проверяем dL/dx и параметры."""
    rng = np.random.default_rng(0)
    xs = [rng.normal(0, 0.5, s).astype(np.float64) for s in shapes]
    w = rng.normal(0, 1, None)
    y, c = fwd(*xs)
    w = rng.normal(0, 1, y.shape)
    ana = bwd_ana(c, w)
    def fn(*xx):
        yy, _ = fwd(*xx); return float((yy * w).sum())
    worst, where = 0.0, ""
    for xi, (x, an) in enumerate(zip(xs, ana)):
        flat_a = np.asarray(an).reshape(-1)
        idxs = rng.choice(x.size, min(nsample, x.size), replace=False)
        for idx in idxs:
            flat = x.reshape(-1); o = flat[idx]
            flat[idx] = o + eps; lp = fn(*xs)
            flat[idx] = o - eps; lm = fn(*xs)
            flat[idx] = o
            num = (lp - lm) / (2 * eps); ana_v = flat_a[idx]
            if abs(num) + abs(ana_v) < 1e-5: continue
            err = abs(num - ana_v) / max(abs(num), abs(ana_v), abs_tol)
            if err > worst: worst, where = err, f"{name}@{idx}"
    results.append((name, worst, where, worst < rel_tol))
    return

def check_op(name, fwd, bwd_wrap, shapes):
    rng = np.random.default_rng(1)
    xs = [rng.normal(0, 0.6, s).astype(np.float64) for s in shapes]
    y, c = fwd(*xs)
    wy = rng.normal(0, 1, y.shape)
    ana = bwd_wrap(c, wy)
    def fn(*xx):
        yy, _ = fwd(*xx); return float((yy * wy).sum())
    worst = 0.0
    eps = 1e-6
    for x, an in zip(xs, ana):
        flat_a = np.asarray(an).reshape(-1)
        idxs = np.random.default_rng(2).choice(x.size, min(64, x.size), replace=False)
        for idx in idxs:
            flat = x.reshape(-1); o = flat[idx]
            flat[idx] = o + eps; lp = fn(*xs)
            flat[idx] = o - eps; lm = fn(*xs)
            flat[idx] = o
            num = (lp - lm) / (2*eps); ana_v = flat_a[idx]
            if abs(num) + abs(ana_v) < 1e-5: continue
            worst = max(worst, abs(num - ana_v) / max(abs(num), abs(ana_v)))
    return worst

worst = check_op("gelu", lambda x: gelu(x), lambda c, dy: (gelu_bwd(c, dy),), [(3, 4, 8)])
results.append(("gelu", worst, f"gelu@{worst:.0e}", worst < rel_tol))

worst = check_op("layernorm", lambda x, g, b: layernorm(x, g, b),
                 lambda c, dy: layernorm_bwd(c, dy), [(2, 5, 8), (8,), (8,)])
results.append(("layernorm", worst, "", worst < rel_tol))

worst = check_op("linear", lambda x, w: linear(x, w), lambda c, dy: linear_bwd(c, dy),
                 [(2, 3, 8), (8, 6)])
results.append(("linear", worst, "", worst < rel_tol))

worst = check_op("attention", lambda x, w1, w2: attention(x, w1, w2, 2),
                 lambda c, dy: attention_bwd(c, dy), [(2, 5, 8), (8, 24), (8, 8)])
results.append(("attention", worst, "", worst < rel_tol))

worst = check_op("ema_mix", lambda x, th, sc: ema_mix(x, th, sc),
                 lambda c, dy: ema_mix_bwd(c, dy), [(2, 6, 8), (8,), (8,)])
results.append(("ema_mix", worst, "", worst < rel_tol))

# softmax_ce
rng = np.random.default_rng(3)
lg = rng.normal(0, 1, (2, 6, 11)).astype(np.float64)
yy = rng.integers(0, 11, (2, 6))
_, dz = softmax_ce(lg, yy)
worst = 0.0; eps = 1e-6
for idx in np.random.default_rng(4).choice(lg.size, 80, replace=False):
    flat = lg.reshape(-1); o = flat[idx]
    flat[idx] = o + eps; lp = softmax_ce(lg, yy)[0]
    flat[idx] = o - eps; lm = softmax_ce(lg, yy)[0]
    flat[idx] = o
    num = (lp - lm) / (2*eps); ana = dz.reshape(-1)[idx]
    if abs(num) + abs(ana) < 1e-5: continue
    worst = max(worst, abs(num - ana) / max(abs(num), abs(ana)))
results.append(("softmax_ce", worst, "", worst < rel_tol))


def model_fd(tag, make):
    m = make()
    for k in m.p.d: m.p.d[k] = m.p.d[k].astype(np.float64)
    for k in m.p.g: m.p.g[k] = m.p.g[k].astype(np.float64)
    rng = np.random.default_rng(0)
    ids = rng.integers(0, m.V, (3, 8)); ytrue = rng.integers(0, m.V, (3, 8))
    def fn():
        H, _ = m.forward(ids); l, _ = softmax_ce(m.logits(H), ytrue); return float(l)
    H, c = m.forward(ids); l, dz = softmax_ce(m.logits(H), ytrue)
    m.p.zero(); m.backward(H, c, dz)
    worst, where = 0.0, ""
    eps = 1e-6
    for name in m.p.d:
        arr = m.p.d[name]; ga = m.p.g[name]
        if arr.size > 512:
            idxs = rng.choice(arr.size, 40, replace=False)
        else:
            idxs = np.arange(arr.size)
        for idx in idxs:
            flat = arr.reshape(-1); o = flat[idx]
            flat[idx] = o + eps; lp = fn()
            flat[idx] = o - eps; lm = fn()
            flat[idx] = o
            num = (lp - lm) / (2*eps); ana = ga.reshape(-1)[idx]
            if abs(num) + abs(ana) < 1e-5: continue
            err = abs(num - ana) / max(abs(num), abs(ana))
            if err > worst: worst, where = err, f"{tag}:{name}@{idx}"
    return worst, where

# op-level: delta_mix (KDA-lite)
worst = check_op("delta_mix", lambda x, th, sc, br: delta_mix(x, th, sc, br),
                 lambda c, dy: delta_mix_bwd(c, dy), [(2, 6, 8), (8,), (8,), ()])
results.append(("delta_mix", worst, "", worst < rel_tol))
w, wh = model_fd("hybrid", lambda: NanoGPT(40, D=16, L=3, h=2, ff=32, T=8, kind="hybrid"))
results.append(("nanoGPT-hybrid", w, wh, w < 5e-4))
w, wh = model_fd("ema+ADR", lambda: NanoGPT(40, D=16, L=3, h=2, ff=32, T=8, kind="ema", adr_kf=0.5))
results.append(("ema+ADR", w, wh, w < 5e-4))
w, wh = model_fd("delta", lambda: NanoGPT(40, D=16, L=3, h=2, ff=32, T=8, kind="delta"))
results.append(("kda-delta", w, wh, w < 5e-4))
w, wh = model_fd("ema+moe4", lambda: NanoGPT(40, D=16, L=3, h=2, ff=32, T=8, kind="ema", moe_e=4))
results.append(("ema+moe4", w, wh, w < 5e-4))

ok_all = True
for name, worst, where, ok in results:
    print(f"{'OK ' if ok else 'FAIL'} {name:14s} worst_rel_err={worst:.2e} {where}")
    ok_all &= ok
print("\n== ВСЕ ГРАДИЕНТЫ ВЕРНЫ ==" if ok_all else "\n== ЕСТЬ ОШИБКИ, СМ. ВЫШЕ ==")
sys.exit(0 if ok_all else 1)
