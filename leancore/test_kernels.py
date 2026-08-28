#!/usr/bin/env python3
"""Численная сверка ctypes-ядер (lc_kernels.so) против numpy-опорного кода в f32."""
import numpy as np, ctypes as ct, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
KS = ct.CDLL(os.path.join(ROOT, "lc_kernels.so"))
FP = ct.POINTER(ct.c_float); IP = ct.POINTER(ct.c_int64)
F=FP
KS.k_ln_fwd.argtypes=[F,F,F,F,F,F,F,ct.c_int64,ct.c_int,ct.c_float]
KS.k_ln_bwd.argtypes=[F,F,F,F,F,F,F,ct.c_int64,ct.c_int]
KS.k_sce.argtypes=[F,IP,ct.c_int64,ct.c_int64,ct.c_float]; KS.k_sce.restype=ct.c_double
def fp(a): return a.ctypes.data_as(FP)

rng = np.random.default_rng(7)
fails = 0
def chk(name, a, b, tol):
    global fails
    e = float(np.abs(a - b).max())
    ok = e <= tol
    if not ok: fails += 1
    print(f"{'OK ' if ok else 'FAIL'} {name}: max|Δ|={e:.3g} (tol {tol})")

# --- gelu fwd/bwd ---
x = (rng.normal(0, 1.5, (24, 96, 576))).astype(np.float32)
c0, c1 = np.float32(0.7978845608028654), np.float32(0.044715)
u = c0 * (x + c1 * x * x * x); t_ref = np.tanh(u); y_ref = np.float32(0.5) * x * (1 + t_ref)
y_c = np.empty_like(x); t_c = np.empty_like(x)
KS.k_gelu_fwd(fp(np.ascontiguousarray(x)), fp(y_c), fp(t_c), x.size)
chk("gelu_fwd y", y_ref, y_c, 1e-6); chk("gelu_fwd t", t_ref, t_c, 1e-6)
dy = rng.normal(0, 1, x.shape).astype(np.float32)
du = c0 * (1 + 3 * c1 * x * x)
dx_ref = dy * (np.float32(0.5) * (1 + t_ref) + np.float32(0.5) * x * (1 - t_ref * t_ref) * du)
dx_c = np.empty_like(x)
KS.k_gelu_bwd(fp(np.ascontiguousarray(x)), fp(t_c), fp(np.ascontiguousarray(dy)), fp(dx_c), x.size)
chk("gelu_bwd", dx_ref, dx_c, 1e-5)

# --- layernorm fwd/bwd ---
xl = rng.normal(0, 1, (24, 96, 192)).astype(np.float32)
g = rng.normal(1, .1, 192).astype(np.float32); b = rng.normal(0, .1, 192).astype(np.float32)
mu = xl.mean(-1, keepdims=True); xcr = xl - mu
var = (xcr**2).mean(-1, keepdims=True); istd = np.float32(1/np.sqrt(var + 1e-5))
xh = xcr * istd; y_ref = g * xh + b
R, N = xl.size // 192, 192
y_c = np.empty_like(xl); xh_c = np.empty_like(xl); mu_c = np.empty(R, np.float32); rs_c = np.empty(R, np.float32)
KS.k_ln_fwd(fp(np.ascontiguousarray(xl)), fp(g), fp(b), fp(y_c), fp(xh_c), fp(mu_c), fp(rs_c), R, N, np.float32(1e-5))
chk("ln_fwd y", y_ref, y_c, 1e-5); chk("ln_fwd xhat", xh, xh_c, 4e-6)
dyl = rng.normal(0, 1, xl.shape).astype(np.float32)
dg_ref = (dyl * xh).sum((0, 1)); db_ref = dyl.sum((0, 1))
v = dyl * g
dx_ref = (istd / N) * (N * v - v.sum(-1, keepdims=True) - xh * (v * xh).sum(-1, keepdims=True))
dx_c = np.empty_like(dyl); dg_c = np.empty(N, np.float32); db_c = np.empty(N, np.float32)
KS.k_ln_bwd(fp(np.ascontiguousarray(g)), fp(np.ascontiguousarray(dyl)), fp(np.ascontiguousarray(xh)),
            fp(np.ascontiguousarray(rs_c)), fp(dx_c), fp(dg_c), fp(db_c), R, N)
chk("ln_bwd dx", dx_ref, dx_c, 2e-4); chk("ln_bwd dg", dg_ref, dg_c, 1e-3); chk("ln_bwd db", db_ref, db_c, 1e-4)

# --- softmax+CE ---
V = 8000; Rt = 24 * 96
lg = rng.normal(0, 3, (Rt, V)).astype(np.float32)
yidx = rng.integers(0, V, Rt).astype(np.int64)
z = lg - lg.max(-1, keepdims=True); e = np.exp(z); p = e / e.sum(-1, keepdims=True)
nll_ref = float(-np.log(p[np.arange(Rt), yidx] + 1e-12).mean())
dz_ref = p.copy(); dz_ref[np.arange(Rt), yidx] -= 1.0; dz_ref /= Rt
dz_c = lg.copy()
nll_c = KS.k_sce(fp(dz_c), yidx.ctypes.data_as(IP), Rt, V, np.float32(1.0 / Rt))
print(f"{'OK ' if abs(nll_ref-nll_c) < 1e-4 else 'FAIL'} sce loss: ref {nll_ref:.8f} vs C {nll_c:.8f}")
if abs(nll_ref - nll_c) >= 1e-4: fails += 1
chk("sce dz", dz_ref, dz_c, 5e-6)

# --- ema fwd/bwd (C vs numpy-einsum опора) ---
import importlib
sys.path.insert(0, ROOT)
import nano_lc as NL
B_, T_, D_ = 4, 96, 192
Xe = rng.normal(0, 1, (B_, T_, D_)).astype(np.float32)
th = rng.normal(0, .8, D_).astype(np.float32); scv = rng.normal(1, .2, D_).astype(np.float32)
dYe = rng.normal(0, 1, (B_, T_, D_)).astype(np.float32)
KS_holder = NL._KS
NL._KS = None
Y_ref, cch = NL.ema_mix(Xe, th, scv)                       # einsum-опора
dX_ref, dth_ref, dsc_ref = NL.ema_mix_bwd(cch, dYe)
NL._KS = KS_holder
Y_c, cc_h = NL.ema_mix(Xe, th, scv)                        # C-путь
dX_c, dth_c, dsc_c = NL.ema_mix_bwd(cc_h, dYe)
chk("ema_fwd", Y_ref, Y_c, 1e-5)
chk("ema_bwd dX", dX_ref, dX_c, 2e-4)
chk("ema_bwd dth", dth_ref, dth_c, 2e-5)
chk("ema_bwd dsc", dsc_ref, dsc_c, 2e-4)

print("== C-KERNELS " + ("ALL OK ==" if fails == 0 else f"{fails} FAIL =="))
sys.exit(1 if fails else 0)
