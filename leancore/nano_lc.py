#!/usr/bin/env python3
"""nano_lc — собственный минимальный DL-фреймворк на чистом NumPy (без torch/TF).
Выводы backward — ручные (см. MATH.md), все верифицированы FD-проверкой gradcheck.py.

Модели: kind = attn | ema | hybrid (блок0 attn, остальные ema).
adr_kf ∈ (0,1]: блоки 1..L-1 маршрутизируют top-k=round(adr_kf·T) токенов через плечо
mixer+FFN с гейтом σ(s_t) (ADR, mixture-of-depths стиль); k делится на T окна.

CLI: python3 nano_lc.py --kind ema --tag run --steps 500 [--adr 0.55] [--initckpt x.npz]
"""
import os, sys, json, math, time, argparse
import numpy as np

PROF = {}      # in-situ профайлер: NANOLC_PROF=1 → медианы по секциям в конце
def _tick(sec, t0):
    PROF.setdefault(sec, []).append(time.process_time_ns() - t0)
    return time.process_time_ns()

ROOT = os.path.dirname(os.path.abspath(__file__))
f32 = np.float32

def scatter_add_rows(dst, ids, src):
    """dst[ids] += src с ДУБЛИКАТАМИ в ids (векторно: сорт + reduceat; замена медленному np.add.at).
    ids (N,) int, src (N,D) — суммируется по одинаковым id и пишется одной fancy-записью."""
    ids = ids.reshape(-1)
    src2 = src.reshape(ids.shape[0], -1)
    order = np.argsort(ids, kind="stable")
    sids = ids[order]
    bounds = np.flatnonzero(np.r_[True, sids[1:] != sids[:-1]])
    dst[sids[bounds]] += np.add.reduceat(src2[order], bounds)

class Params:
    def __init__(self): self.d, self.g = {}, {}
    def add(self, name, shape, std=0.02, init=None):
        rng = np.random.default_rng(abs(hash(name)) % (2**32))
        self.d[name] = (rng.normal(0, std, shape).astype(f32) if init is None else init(shape))
        self.g[name] = np.zeros_like(self.d[name])
    def zero(self):
        for k in self.g: self.g[k][...] = 0

# ---------------------------------------------------------------- core ops (fwd + hand-derived bwd)

# ---- ctypes-ядра (lc_kernels.so): gelu / layernorm / softmax+CE — с numpy-фолбэком ----
import ctypes as _ct
_KS = None
try:
    _KS = _ct.CDLL(os.path.join(ROOT, "lc_kernels.so"))
    _F = _ct.POINTER(_ct.c_float); _I = _ct.POINTER(_ct.c_int64)
    _KS.k_gelu_fwd.argtypes = [_F, _F, _F, _ct.c_int64]
    _KS.k_gelu_bwd.argtypes = [_F, _F, _F, _F, _ct.c_int64]
    _KS.k_ln_fwd.argtypes = [_F, _F, _F, _F, _F, _F, _F, _ct.c_int64, _ct.c_int, _ct.c_float]
    _KS.k_ln_bwd.argtypes = [_F, _F, _F, _F, _F, _F, _F, _ct.c_int64, _ct.c_int]
    _KS.k_sce.argtypes = [_F, _I, _ct.c_int64, _ct.c_int64, _ct.c_float]
    _KS.k_sce.restype = _ct.c_double
    _KS.k_ema_fwd.argtypes = [_F, _F, _F, _F, _F, _ct.c_int, _ct.c_int, _ct.c_int]
    _KS.k_ema_bwd.argtypes = [_F, _F, _F, _F, _F, _F, _F, _F, _ct.c_int, _ct.c_int, _ct.c_int]
except OSError:
    _KS = None


def _fp(a): return a.ctypes.data_as(_ct.POINTER(_ct.c_float))

def gelu(x):                                   # tanh-approx gelu: ctypes-ядро или numpy
    if _KS is not None and x.dtype == np.float32:
        xc = np.ascontiguousarray(x, np.float32); n = xc.size
        y = np.empty_like(xc); t = np.empty_like(xc)
        _KS.k_gelu_fwd(_fp(xc), _fp(y), _fp(t), n)
        return y, (xc, t)
    c1 = np.float32(0.044715); c0 = np.float32(0.7978845608028654)
    x2 = x * x
    u = x2 * x; u *= c1; u += x; u *= c0
    t = np.tanh(u)
    y = t.copy(); y += np.float32(1.0); y *= x; y *= np.float32(0.5)
    return y, (x, t)
def gelu_bwd(c, dy):                           # dx = dy·(½(1+t) + ½x(1−t²)·k0(1+3c₁x²))
    x, t = c
    if _KS is not None and dy.dtype == np.float32:
        dyc = np.ascontiguousarray(dy, np.float32); n = x.size
        dx = np.empty_like(x)
        _KS.k_gelu_bwd(_fp(x), _fp(t), _fp(dyc), _fp(dx), n)
        return dx
    q = t.copy(); q *= t; q *= np.float32(-1.0); q += np.float32(1.0)   # 1−t²
    x2 = x * x; x2 *= np.float32(3.0 * 0.044715); x2 += np.float32(1.0) # 1+3c₁x²
    q *= x2; q *= np.float32(0.5 * 0.7978845608028654); q *= x          # ½k0·x·(1−t²)(1+3c₁x²)
    r = t.copy(); r += np.float32(1.0); r *= np.float32(0.5); r += q
    r *= dy
    return r

def layernorm(x, g, b, eps=1e-5):              # x: (...,N)
    N = x.shape[-1]
    if _KS is not None and x.dtype == np.float32:
        xc = np.ascontiguousarray(x, np.float32)
        R = xc.size // N
        y = np.empty_like(xc); xhat = np.empty_like(xc)
        mu = np.empty(R, np.float32); istd = np.empty(R, np.float32)
        _KS.k_ln_fwd(_fp(xc), _fp(g), _fp(b), _fp(y), _fp(xhat), _fp(mu), _fp(istd), R, N, np.float32(eps))
        return y, (xhat, istd.reshape(x.shape[:-1] + (1,)), g, N)
    mu = x.mean(-1, keepdims=True); xc = x - mu
    var = (xc**2).mean(-1, keepdims=True); istd = 1.0 / np.sqrt(var + eps)
    xhat = xc * istd
    return g * xhat + b, (xhat, istd, g, N)
def layernorm_bwd(c, dy):
    xhat, istd, g, N = c
    if _KS is not None and dy.dtype == np.float32:
        dyc = np.ascontiguousarray(dy, np.float32)
        R = dyc.size // N
        dx = np.empty_like(dyc); dg = np.empty(N, np.float32); db = np.empty(N, np.float32)
        _KS.k_ln_bwd(_fp(np.ascontiguousarray(g, np.float32)), _fp(dyc),
                     _fp(np.ascontiguousarray(xhat, np.float32)),
                     _fp(np.ascontiguousarray(istd.reshape(-1), np.float32)),
                     _fp(dx), _fp(dg), _fp(db), R, N)
        return dx, dg, db
    dg = (dy * xhat).sum(axis=tuple(range(dy.ndim - 1)))
    db = dy.sum(axis=tuple(range(dy.ndim - 1)))
    v = dy * g                                                   # γ ДО редукций
    dx = (istd / N) * (N * v - v.sum(-1, keepdims=True) - xhat * (v * xhat).sum(-1, keepdims=True))
    return dx, dg, db

# ---------------- sampled softmax (batch-candidates + bias-correction Q_c = 1−(1−q)^K) -------------
# Голова H@E[cand].T по m ≪ V кандидатам; логиты корректируются −logQ_c (RAW/Jean-style):
# E[∂L'/∂z] ≈ ∂L/∂z при общем наборе кандидатов на батч. Вал — всегда полный softmax.

def sampled_ce(H, y, E, cand, logcor, Ec=None):
    """H (B,T,D), y (B,T), cand (m,) sorted unique ⊇ targets, logcor (m,) = logQ.
    → (loss, (dz, cand, Ec)); dz — градиент по скорректированным логитам (p−onehot)/(B·T).
    Ec можно передать посчитанным (низкоранговый случай: U[cand]@P)."""
    if Ec is None:
        Ec = E[cand]                                 # (m,D)
    lg = H @ Ec.T                                    # (B,T,m)
    lg = lg - logcor[None, None, :]
    lbl = np.searchsorted(cand, y)
    loss, dz = softmax_ce(lg, lbl, destroy=True)
    return loss, (dz, cand, Ec)

def sampled_ce_bwd(c, H):
    dz, cand, Ec = c
    dH = dz @ Ec                                     # (B,T,D)
    dEc = dz.reshape(-1, dz.shape[-1]).T @ H.reshape(-1, H.shape[-1])   # (m,D)
    return dH, dEc

ACTQ8 = False   # QAT-режим: квантование активаций int8 (динамический absmax на строку) перед матмулом

def _actq(x):
    am = np.abs(x).max(-1, keepdims=True)
    am = np.where(am < 1e-8, np.ones_like(am), am)
    return np.rint(x / am * 127) * (am / 127)

def linear(x, W):                              # (..., in) @ (in,out)
    xq = _actq(x) if ACTQ8 else x
    return xq @ W, (xq, W)                     # градиент по квантованному входу (STE-стандарт)
def linear_bwd(c, dy):
    x, W = c
    dw = x.reshape(-1, x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
    return dy @ W.T, dw


def kron_pair_facs(in_d, out_d):
    """Факторы in=n1·n2 (n1>=n2), out=m1·m2 (m1<=m2), стоимость m1·n2·(n1+m2) минимальна."""
    import math as _m
    def facs(n, want_min_first):
        best, bd = (n, 1), 1e9
        for a in range(1, int(_m.sqrt(n)) + 1):
            if n % a == 0:
                b = n // a
                d = (a - b) if want_min_first else (b - a)
                if abs(a - b) < bd:
                    bd = abs(a - b); best = (a, b) if want_min_first else (b, a)
        return best
    n2, n1 = facs(in_d, False)     # n1 — больший фактор входа
    m1, m2 = facs(out_d, True)     # m1 — меньший фактор выхода
    return n1, n2, m1, m2

def kron_fwd(x2d, As, Bs):
    """x2d (P, n1·n2) → (y (P, m1·m2), cache). Y = Σ_k A_k X B_kᵀ, vec row-major с обеих сторон."""
    P = x2d.shape[0]
    n1, n2 = As[0].shape[1], Bs[0].shape[1]
    m1, m2 = As[0].shape[0], Bs[0].shape[0]
    X = x2d.reshape(P, n1, n2)
    Y = np.zeros((P, m1, m2), x2d.dtype)
    for A, B in zip(As, Bs):
        t = X.transpose(0, 2, 1).reshape(P * n2, n1) @ A.T
        t = t.reshape(P, n2, m1).transpose(0, 2, 1).reshape(P * m1, n2) @ B.T
        Y += t.reshape(P, m1, m2)
    return Y.reshape(P, m1 * m2), (X, As, Bs)

def kron_bwd(c, dy2d):
    """dy2d (P, m1·m2) → (dX (P, n1·n2), [dA_k], [dB_k]).
    dA_k = Σ_p G B Xᵀ;  dB_k = Σ_p Gᵀ A X;  dX = Σ_k Aᵀ G B."""
    X, As, Bs = c
    P = X.shape[0]
    m1, m2 = As[0].shape[0], Bs[0].shape[0]
    G = dy2d.reshape(P, m1, m2)
    Gt = G.transpose(0, 2, 1)
    dX = np.zeros_like(X)
    dAs, dBs = [], []
    for A, B in zip(As, Bs):
        dAs.append(np.einsum('pmr,pnr->mn', G @ B, X))
        dBs.append(np.einsum('pmn,pnq->mq', Gt @ A, X))
        dX += A.T @ G @ B
    return dX.reshape(P, X.shape[1] * X.shape[2]), dAs, dBs

def softmax_ce(logits, y, destroy=False):      # logits (B,T,V), y (B,T) -> (loss, dlogits); destroy=True → пишем dz поверх logits
    B, T, V = logits.shape
    if _KS is not None and logits.dtype == np.float32:
        dz = logits if (destroy and logits.flags['C_CONTIGUOUS']) else np.ascontiguousarray(logits, np.float32).copy()
        yy = np.ascontiguousarray(y, np.int64).reshape(-1)
        nll = _KS.k_sce(_fp(dz), _ct.cast(yy.ctypes.data, _ct.POINTER(_ct.c_int64)),
                        B * T, V, np.float32(1.0 / (B * T)))
        return nll, dz
    z = logits - logits.max(-1, keepdims=True)
    e = np.exp(z); p = e / e.sum(-1, keepdims=True)
    nll = -np.log(p.reshape(-1, V)[np.arange(B*T), y.reshape(-1)] + 1e-12).mean()
    dz = p
    dz.reshape(-1, V)[np.arange(B*T), y.reshape(-1)] -= 1.0
    dz /= (B * T)
    return nll, dz

def attention(X, Wqkv, Wo, h):                 # X (B,T,D), causal MHA
    B, T, D = X.shape; dh = D // h
    Z = X @ Wqkv                                # (B,T,3D)
    Q = Z[..., :D].reshape(B, T, h, dh).transpose(0, 2, 1, 3)
    K = Z[..., D:2*D].reshape(B, T, h, dh).transpose(0, 2, 1, 3)
    V = Z[..., 2*D:].reshape(B, T, h, dh).transpose(0, 2, 1, 3)
    S = (Q @ K.transpose(0, 1, 3, 2)) / math.sqrt(dh)
    mask = np.triu(np.ones((T, T), bool), 1)
    S = np.where(mask, -1e30, S)
    A = np.exp(S - S.max(-1, keepdims=True)); A /= A.sum(-1, keepdims=True)
    Y = A @ V                                   # (B,h,T,dh)
    Yc = Y.transpose(0, 2, 1, 3).reshape(B, T, D)
    out = Yc @ Wo
    return out, (X, Wqkv, Wo, Q, K, V, A, Yc)
def attention_bwd(c, dY):
    X, Wqkv, Wo, Q, K, V, A, Yc = c
    B, h, T, dh = Q.shape; D = Wo.shape[1]
    dWo = Yc.reshape(-1, D).T @ dY.reshape(-1, D)
    dYc = dY @ Wo.T
    dY2 = dYc.reshape(B, T, h, dh).transpose(0, 2, 1, 3)
    dA = dY2 @ V.transpose(0, 1, 3, 2)
    dV = A.transpose(0, 1, 3, 2) @ dY2
    dS = A * (dA - (A * dA).sum(-1, keepdims=True)) / math.sqrt(dh)
    dQ = dS @ K
    dK = dS.transpose(0, 1, 3, 2) @ Q
    dQ_ = dQ.transpose(0, 2, 1, 3).reshape(B, T, D)
    dK_ = dK.transpose(0, 2, 1, 3).reshape(B, T, D)
    dV_ = dV.transpose(0, 2, 1, 3).reshape(B, T, D)
    dZ = np.concatenate([dQ_, dK_, dV_], axis=-1)
    dWqkv = X.reshape(-1, D).T @ dZ.reshape(-1, 3 * D)
    return dZ @ Wqkv.T, dWqkv, dWo

_EMA_CACHE = {}                                   # (T,D,a) → M_forward, L_backward

def _ema_mats(a, T, D, dt):
    key = (T, D, a.tobytes(), dt)
    got = _EMA_CACHE.get(key)
    if got is not None: return got
    if len(_EMA_CACHE) > 12: _EMA_CACHE.clear()   # a меняется каждый шаг → не копим
    tt = np.arange(T)
    dd = tt[:, None] - tt[None, :]                # t−k
    mf = dd >= 0
    ad = np.abs(dd)[:, :, None].astype(dt)
    Pf = a[None, None, :] ** ad
    M = Pf * mf[:, :, None] * (1.0 - a)[None, None, :]         # h_t = Σ_k a^{t−k}(1−a)x_k, k≤t
    Lb = Pf * (dd <= 0)[:, :, None]                             # lam_t = Σ_s a^{s−t}·dH_s, s≥t
    _EMA_CACHE[key] = (M, Lb)
    return M, Lb

def ema_mix(X, th, sc):                        # h_t = a·h_{t-1} + (1-a)·x_t; y = sc⊙h
    a = (1.0 / (1.0 + np.exp(-th))).astype(X.dtype)
    B, T, D = X.shape
    if _KS is not None and X.dtype == np.float32:
        Xc = np.ascontiguousarray(X)
        H = np.empty_like(Xc); Y = np.empty_like(Xc)
        ac = np.ascontiguousarray(a); scc = np.ascontiguousarray(sc)
        _KS.k_ema_fwd(_fp(Xc), _fp(ac), _fp(scc), _fp(Y), _fp(H), B, T, D)
        return Y, (Xc, H, a, sc)
    M, _ = _ema_mats(a, T, D, X.dtype)
    H = np.einsum('tkd,bkd->btd', M, X, optimize=True)
    return H * sc, (X, H, a, sc)

def ema_mix_bwd(c, dY):
    X, H, a, sc = c
    B, T, D = X.shape
    if _KS is not None and dY.dtype == np.float32:
        dYc = np.ascontiguousarray(dY)
        dX = np.empty_like(Xc_ := np.ascontiguousarray(X))
        dth = np.empty(D, np.float32); dsc = np.empty(D, np.float32)
        _KS.k_ema_bwd(_fp(Xc_), _fp(np.ascontiguousarray(H)), _fp(np.ascontiguousarray(a)),
                      _fp(np.ascontiguousarray(sc)), _fp(dYc), _fp(dX), _fp(dth), _fp(dsc), B, T, D)
        return dX, dth, dsc
    dH = dY * sc
    dsc = (dY * H).sum(axis=(0, 1))
    M, Lb = _ema_mats(a, T, D, X.dtype)
    lam = np.einsum('tsd,bsd->btd', Lb, dH, optimize=True)       # lam_t = Σ_{s≥t} a^{s-t} dH_s
    dX = (1.0 - a)[None, None, :] * lam
    hp = np.zeros_like(H); hp[:, 1:] = H[:, :-1]                 # h_{t-1}
    da = (lam * (hp - X)).sum((0, 1))
    dth = da * a * (1.0 - a)
    return dX, dth, dsc

# ---------------------------------------------------------------- model
class NanoGPT:
    def __init__(self, V, D=192, L=4, h=6, ff=576, T=96, kind="attn", adr_kf=None, moe_e=1, mtp_w=0.0, erank=0, kronfc=0):  # noqa
        self.V, self.D, self.L, self.h, self.T, self.ff, self.moe_e, self.mtp_w = V, D, L, h, T, ff, moe_e, mtp_w
        self.kind, self.adr_kf, self.erank, self.kronfc = kind, adr_kf, erank, kronfc
        p = self.p = Params()
        if erank > 0:                                  # низкоранговый tied E = U(V,r) @ P(r,D)
            sg = (0.02 ** 2 / erank) ** 0.25           # Var(E_ij) = r·σ⁴ = 0.02² → честный init
            p.add("U", (V, erank), std=sg); p.add("P", (erank, D), std=sg)
        else:
            p.add("E", (V, D))
        p.add("pos", (T, D))
        self.blocks = []
        for i in range(L):
            kind_i = "attn" if (kind == "attn" or (kind == "hybrid" and i == 0)) else ("delta" if kind == "delta" else "ema")
            b = {"kind": kind_i, "routed": bool(adr_kf) and i > 0}
            b["kind"] = kind_i
            self.blocks.append(b)
            p.add(f"b{i}.ln1g", (D,), init=lambda s: np.ones(s, f32))
            p.add(f"b{i}.ln1b", (D,), init=lambda s: np.zeros(s, f32))
            if kind_i == "attn":
                p.add(f"b{i}.Wqkv", (D, 3 * D)); p.add(f"b{i}.Wo", (D, D))
            else:
                p.add(f"b{i}.th", (D,), init=lambda s: np.zeros(s, f32))     # a=σ(0)=0.5
                p.add(f"b{i}.sc", (D,), init=lambda s: np.ones(s, f32))
                if kind_i == "delta":
                    p.add(f"b{i}.braw", (), init=lambda s: np.array(-2.0, f32))   # β(0)=0.119 — мягкая коррекция
                    p.add(f"b{i}.Wg", (D, D))                                      # KDA output gate
                p.add(f"b{i}.Wm", (D, D))
            p.add(f"b{i}.ln2g", (D,), init=lambda s: np.ones(s, f32))
            p.add(f"b{i}.ln2b", (D,), init=lambda s: np.zeros(s, f32))
            if self.moe_e > 1:
                E, F2 = self.moe_e, ff // 2
                p.add(f"b{i}.wr", (D, E))
                p.add(f"b{i}.be", (E,), init=lambda s: np.zeros(s, f32))
                p.add(f"b{i}.w1", (E, D, F2))
                p.add(f"b{i}.w2", (E, F2, D))
            else:
                if self.kronfc > 0:
                    n1, n2, m1, m2 = kron_pair_facs(D, ff)
                    o1, o2, q1, q2 = kron_pair_facs(ff, D)
                    def kinit(shape):
                        return np.random.default_rng(1).normal(0, (2.0 / (shape[0]*shape[1]))**0.25, shape).astype(f32)
                    for k_ in range(self.kronfc):
                        p.add(f"b{i}.fc1A{k_}", (m1, n1)); p.add(f"b{i}.fc1B{k_}", (m2, n2))
                        p.add(f"b{i}.fc2A{k_}", (q1, o1)); p.add(f"b{i}.fc2B{k_}", (q2, o2))
                    self._kfacs = (n1, n2, m1, m2, o1, o2, q1, q2)
                else:
                    p.add(f"b{i}.fc1", (D, ff)); p.add(f"b{i}.fc2", (ff, D))
            if b["routed"]:
                p.add(f"b{i}.rw", (D,))
                p.add(f"b{i}.rb", (), init=lambda s: np.array(1.5, f32))
        p.add("lnfg", (D,), init=lambda s: np.ones(s, f32))
        p.add("lnfb", (D,), init=lambda s: np.zeros(s, f32))
        if mtp_w > 0:
            p.add("lnm_g", (D,), init=lambda s: np.ones(s, f32))
            p.add("lnm_b", (D,), init=lambda s: np.zeros(s, f32))
            p.add("Wmtp", (D, D))

    def _arm_forward(self, X2, i, b):
        """плечо блока: ln1 → mixer(+Wm/attn) → +resid → ln2 → ffn → delta=mix+o2"""
        p = self.p.d; c = {}
        ln_out, c["ln1"] = layernorm(X2, p[f"b{i}.ln1g"], p[f"b{i}.ln1b"])
        if b["kind"] == "attn":
            mix, c["attn"] = attention(ln_out, p[f"b{i}.Wqkv"], p[f"b{i}.Wo"], self.h)
        elif b["kind"] == "delta":
            em, c["ema"] = delta_mix(ln_out, p[f"b{i}.th"], p[f"b{i}.sc"], p[f"b{i}.braw"], p[f"b{i}.Wg"])
            mix, c["wm"] = linear(em, p[f"b{i}.Wm"])
        else:
            em, c["ema"] = ema_mix(ln_out, p[f"b{i}.th"], p[f"b{i}.sc"])
            mix, c["wm"] = linear(em, p[f"b{i}.Wm"])
        H2 = X2 + mix
        ln_out2, c["ln2"] = layernorm(H2, p[f"b{i}.ln2g"], p[f"b{i}.ln2b"])
        c["fc1_in"] = ln_out2
        if self.moe_e > 1:
            c[f"moe_be_{i}"] = p[f"b{i}.be"].copy()
            o2, c["moe"] = moe_ffn(ln_out2, p[f"b{i}.wr"], p[f"b{i}.be"], p[f"b{i}.w1"], p[f"b{i}.w2"], i)
        elif self.kronfc > 0:
            P_, B_, T_ = ln_out2.shape[0] * ln_out2.shape[1], ln_out2.shape[0], ln_out2.shape[1]
            n1, n2, m1, m2, o1, o2, q1, q2 = self._kfacs
            z1, c["kron1"] = kron_fwd(ln_out2.reshape(P_, self.D),
                                      [p[f"b{i}.fc1A{k_}"] for k_ in range(self.kronfc)],
                                      [p[f"b{i}.fc1B{k_}"] for k_ in range(self.kronfc)])
            g1, c["gelu"] = gelu(z1.reshape(B_, T_, self.ff))
            z2, c["kron2"] = kron_fwd(g1.reshape(P_, self.ff),
                                      [p[f"b{i}.fc2A{k_}"] for k_ in range(self.kronfc)],
                                      [p[f"b{i}.fc2B{k_}"] for k_ in range(self.kronfc)])
            o2 = z2.reshape(B_, T_, self.D)          # fc2-cache не нужен — kron bwd свой
        else:
            g1, c["gelu"] = gelu(ln_out2 @ p[f"b{i}.fc1"])
            o2, c["fc2"] = linear(g1, p[f"b{i}.fc2"])
        c["delta"] = mix + o2
        return mix + o2, c

    def _arm_backward(self, c, dD, i, b):
        """dD = dL/dDelta → возвращает dL/dX2 (вход плеча)."""
        p, g = self.p.d, self.p.g
        if self.moe_e > 1:
            dxm, dwr, dbe, dw1, dw2 = moe_ffn_bwd(c["moe"], dD)
            g[f"b{i}.wr"] += dwr; g[f"b{i}.w1"] += dw1; g[f"b{i}.w2"] += dw2
            dln2 = dxm
        elif self.kronfc > 0:
            P_ = dD.shape[0] * dD.shape[1]
            n1, n2, m1, m2, o1, o2, q1, q2 = self._kfacs
            dX2f, dA2s, dB2s = kron_bwd(c["kron2"], dD.reshape(P_, self.D))
            for k_ in range(self.kronfc):
                g[f"b{i}.fc2A{k_}"] += dA2s[k_]; g[f"b{i}.fc2B{k_}"] += dB2s[k_]
            dpre = gelu_bwd(c["gelu"], dX2f.reshape(dD.shape[0], dD.shape[1], self.ff))
            dln24, dA1s, dB1s = kron_bwd(c["kron1"], dpre.reshape(P_, self.ff))
            for k_ in range(self.kronfc):
                g[f"b{i}.fc1A{k_}"] += dA1s[k_]; g[f"b{i}.fc1B{k_}"] += dB1s[k_]
            dln2 = dln24.reshape(dD.shape[0], dD.shape[1], self.D)
        else:
            dX2f, dW2 = linear_bwd(c["fc2"], dD); g[f"b{i}.fc2"] += dW2
            dpre = gelu_bwd(c["gelu"], dX2f)
            g[f"b{i}.fc1"] += c["fc1_in"].reshape(-1, self.D).T @ dpre.reshape(-1, self.ff)
            dln2 = dpre @ p[f"b{i}.fc1"].T
        dx2, dg2, db2 = layernorm_bwd(c["ln2"], dln2); g[f"b{i}.ln2g"] += dg2; g[f"b{i}.ln2b"] += db2
        dmix = dD + dx2                            # mix питает Δ и H2 (вход ln2)
        if b["kind"] == "attn":
            dxm, dWqkv, dWo = attention_bwd(c["attn"], dmix)
            g[f"b{i}.Wqkv"] += dWqkv; g[f"b{i}.Wo"] += dWo
        elif b["kind"] == "delta":
            dxe, dWm = linear_bwd(c["wm"], dmix); g[f"b{i}.Wm"] += dWm
            dxm, dth, dsc, dbraw, dWg = delta_mix_bwd(c["ema"], dxe)
            g[f"b{i}.th"] += dth; g[f"b{i}.sc"] += dsc; g[f"b{i}.braw"] += dbraw; g[f"b{i}.Wg"] += dWg
        else:
            dxe, dWm = linear_bwd(c["wm"], dmix); g[f"b{i}.Wm"] += dWm
            dxm, dth, dsc = ema_mix_bwd(c["ema"], dxe)
            g[f"b{i}.th"] += dth; g[f"b{i}.sc"] += dsc
        dx1, dg1, db1 = layernorm_bwd(c["ln1"], dxm); g[f"b{i}.ln1g"] += dg1; g[f"b{i}.ln1b"] += db1
        return dx2 + dx1

    def forward(self, ids):
        """→ (H после финального LN, caches)"""
        p = self.p.d
        B, T = ids.shape
        if self.erank > 0:
            H = p["U"][ids] @ p["P"] + p["pos"][None, :T]
        else:
            H = p["E"][ids] + p["pos"][None, :T]
        blocks_c = []
        for i, b in enumerate(self.blocks):
            bc = {}
            if b["routed"]:
                Xin = H
                s = Xin @ p[f"b{i}.rw"] + p[f"b{i}.rb"]                # (B,T)
                k = max(1, int(round(self.adr_kf * T)))
                idx = np.argsort(-s, axis=1)[:, :k]; idx.sort(axis=1)
                bi = np.arange(B)[:, None]
                xs = Xin[bi, idx]                                     # (B,k,D)
                g = 1.0 / (1.0 + np.exp(-s[bi, idx]))                 # (B,k)
                delta, arm_c = self._arm_forward(xs, i, b)
                H = H.copy()
                H[bi, idx] = H[bi, idx] + delta * g[..., None]
                bc["route"] = (Xin, s, idx, g, k, arm_c)
            else:
                bc["Xin"] = H
                delta, arm_c = self._arm_forward(H, i, b)
                bc["arm"] = arm_c
                H = H + delta
            blocks_c.append(bc)
        out, c_lnf = layernorm(H, p["lnfg"], p["lnfb"])
        return out, (ids, T, blocks_c, c_lnf)

    def logits(self, H):
        if self.erank > 0:
            return (H @ self.p.d["P"].T) @ self.p.d["U"].T            # tied lowrank head
        return H @ self.p.d["E"].T                   # tied head

    def backward(self, H, caches, dlogits=None, dH_aux=None, head_cache=None):
        p, g = self.p.d, self.p.g
        ids, T, blocks_c, c_lnf = caches
        _t = time.process_time_ns()
        if head_cache is not None:                      # sampled softmax: разреженный путь
            cand, dz, Ec = head_cache
            if self.erank > 0:                          # Ec = U[cand]@P был форвард-копией; градиент по U,P
                PH = H @ p["P"].T                       # (B,T,r)
                Uc = p["U"][cand]                       # (m,r)
                g["U"][cand] += dz.reshape(-1, dz.shape[-1]).T @ PH.reshape(-1, self.erank)
                dPH = dz @ Uc                           # (B,T,r)
                g["P"] += dPH.reshape(-1, self.erank).T @ H.reshape(-1, self.D)
                dH = dPH @ p["P"]
            else:
                g["E"][cand] += dz.reshape(-1, dz.shape[-1]).T @ H.reshape(-1, self.D)   # cand уникален → fancy-add
                dH = dz @ Ec
        elif self.erank > 0:                            # полная голова, низкий ранг
            PH = H @ p["P"].T
            g["U"] += dlogits.reshape(-1, self.V).T @ PH.reshape(-1, self.erank)
            dPH = dlogits @ p["U"]                      # (B,T,r)
            g["P"] += dPH.reshape(-1, self.erank).T @ H.reshape(-1, self.D)
            dH = dPH @ p["P"]
        else:
            g["E"] += dlogits.reshape(-1, self.V).T @ H.reshape(-1, self.D)
            dH = dlogits @ p["E"]
        _t = _tick("bwd_head", _t)
        if dH_aux is not None: dH = dH + dH_aux            # MTP: aux-градиент уровня H (после lnf)
        dH, dg, db = layernorm_bwd(c_lnf, dH); g["lnfg"] += dg; g["lnfb"] += db
        _t = _tick("bwd_body", _t)
        for i in reversed(range(self.L)):
            bc = blocks_c[i]; b = self.blocks[i]
            if b["routed"]:
                Xin, s, idx, gsel, k, arm_c = bc["route"]
                B = Xin.shape[0]; bi = np.arange(B)[:, None]
                dH_sel = dH[bi, idx]                                  # (B,k,D)
                delta = arm_c["delta"]
                # dL/dg = <dH_sel, delta>; dL/ds = ·g(1−g)
                dg_ = (dH_sel * delta).sum(-1)
                ds_ = dg_ * gsel * (1 - gsel)
                g[f"b{i}.rw"] += (ds_[..., None] * Xin[bi, idx]).sum((0, 1))
                g[f"b{i}.rb"] += ds_.sum()
                # dL/dx_sel = dH_sel + J_armᵀ(g⊙dH_sel)   [dDelta = g⊙dH]
                dx_arm = self._arm_backward(arm_c, dH_sel * gsel[..., None], i, b)
                dH[bi, idx] = dH_sel + dx_arm
                dH[bi, idx] += ds_[..., None] * p[f"b{i}.rw"][None, None, :]   # путь через скор
                continue
            dX = self._arm_backward(bc["arm"], dH, i, b)
            dH = dH + dX
        _t = _tick("bwd_blocks", _t)
        if self.erank > 0:                              # lookup-путь: x0 = U[ids]@P
            Ui = p["U"][ids].reshape(-1, self.erank)
            dHf = dH.reshape(-1, self.D)
            scatter_add_rows(g["U"], ids, dHf @ p["P"].T)
            g["P"] += Ui.T @ dHf
        else:
            scatter_add_rows(g["E"], ids, dH)   # градиент эмбеддинга (lookup-путь, векторный)
        g["pos"] += dH.sum(0)
        _t = _tick("bwd_embscat", _t)
        return H

class AdamW:
    def __init__(self, params, lr=6e-4, b1=0.9, b2=0.95, wd=0.1, eps=1e-8, wdmask=False):
        self.p = params; self.b1, self.b2, self.wd, self.eps = b1, b2, wd, eps
        # wdmask: decoupled-wd только на скрытых 2D-матрицах; НЕ на tied-эмбеддинге "E",
        # LN-gains/биасах/скалярах (ndim<2). Классическая правка AdamW → обычно +стабильность.
        self.wdmask = wdmask
        self.m = {k: np.zeros_like(v) for k, v in params.d.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.d.items()}
        self.t = 0
    def step(self, lr):
        self.t += 1
        for k in self.p.d:
            g = self.p.g[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            wd = 0.0 if (self.wdmask and (self.p.d[k].ndim < 2 or k in ("E", "U", "P"))) else self.wd
            self.p.d[k] -= lr * (mh / (np.sqrt(vh) + self.eps) + wd * self.p.d[k])
        self.p.zero()

class MuonW:
    """Muon (Keller Jordan): momentum + Newton–Schulz ортогонализация для 2D-матриц (~35% к скорости NanoGPT).
    1D/скаляры/эмбеддинг/pos остаются на Adam (стандартная практика)."""
    NS_COEF = (3.4445, -4.7750, 2.0315)
    def __init__(self, params, lr=6e-4, mulr=0.02, mom=0.95, wd=0.1, ns=5, eps=1e-8):
        self.p = params; self.lr0, self.mulr, self.mom, self.wd, self.ns, self.eps = lr, mulr, mom, wd, ns, eps
        self.muon_keys = {k for k, v in params.d.items()
                          if v.ndim == 2 and min(v.shape) >= 16 and k not in ("E", "U", "pos")}
        self.adam_keys = set(params.d) - self.muon_keys
        self.mb = {k: np.zeros_like(v) for k, v in params.d.items() if k in self.muon_keys}
        self.m = {k: np.zeros_like(params.d[k]) for k in self.adam_keys}
        self.v = {k: np.zeros_like(params.d[k]) for k in self.adam_keys}
        self.t = 0
        print(f"[MuonW] muon: {sorted(self.muon_keys)[:4]}... ({len(self.muon_keys)} шт), adam: {len(self.adam_keys)} шт", flush=True)
    @staticmethod
    def _ns5(G, steps, eps):
        a, b, c = MuonW.NS_COEF
        X = G / (np.linalg.norm(G.astype(np.float64)) + eps).astype(np.float32)
        tr = X.shape[0] > X.shape[1]
        if tr: X = X.T.copy()
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        if tr: X = X.T
        return X
    def step(self, lr):
        self.t += 1
        for k in self.muon_keys:
            g = self.p.g[k]
            self.mb[k] = self.mom * self.mb[k] + (1 - self.mom) * g
            gu = g * (1 - self.mom) + self.mom * self.mb[k]          # nesterov
            o = self._ns5(gu, self.ns, self.eps)
            scale = max(1.0, o.shape[0] / o.shape[1]) ** 0.5
            self.p.d[k] -= self.mulr * scale * o + self.wd * lr * self.p.d[k]
        for k in self.adam_keys:
            g = self.p.g[k]
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.95 * self.v[k] + 0.05 * g * g
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.95 ** self.t)
            self.p.d[k] -= lr * (mh / (np.sqrt(vh) + self.eps) + self.wd * self.p.d[k])
        self.p.zero()


# ---------------------------------------------------------------- KDA-lite delta-миксер
# S_0 = 0;  p = S·k;  u = v − p;  S ← Diag(a)·S + β·u⊗k;  o = a⊙p + β·u·‖k‖²   [k ≡ v ≡ y]
def delta_mix(X, th, sc, braw, Wg):            # X (B,T,D) → (Y, cache); k = x/‖x‖, v = x
    a = 1.0 / (1.0 + np.exp(-th)); beta = 1.0 / (1.0 + np.exp(-braw))
    B, T, D = X.shape
    S = np.zeros((B, D, D), X.dtype)
    Y = np.empty_like(X)
    KN = np.empty((B, T), X.dtype)                # ‖x_t‖
    GS = np.empty((B, T, D), X.dtype)
    P = np.empty_like(X); U = np.empty_like(X); N2 = np.empty((B, T), X.dtype)
    KK = np.empty_like(X)
    for t in range(T):                            # KDA-update, состояние не складируем (backward replay)
        v = X[:, t]
        kn = np.sqrt((v * v).sum(-1)) + 1e-8
        k = v / kn[:, None]
        p_ = np.einsum('bij,bj->bi', S, k)
        u = v - p_
        n2 = (k * k).sum(-1)                      # ≈1
        S = a[None, :, None] * S + beta * u[:, :, None] * k[:, None, :]
        o = a[None, :] * p_ + beta * u * n2[:, None]
        P[:, t] = p_; U[:, t] = u; N2[:, t] = n2; KK[:, t] = k; KN[:, t] = kn
        Gt = 1.0 / (1.0 + np.exp(-(v @ Wg)))       # KDA output gate σ(xW_g), канальный
        Y[:, t] = o * sc * Gt
        GS[:, t] = Gt
    return Y, (X, KK, KN, P, U, N2, S, a, beta, sc, GS, Wg)

def delta_mix_bwd(c, dY):
    X, KK, KN, P, U, N2, Sfin, a, beta, sc, GS, Wg = c
    B, T, D = X.shape
    dX = np.zeros_like(X); dth = np.zeros(D, X.dtype); dbeta = 0.0
    dWg = np.zeros_like(Wg)
    dO_full = dY * sc * GS
    dsc = (dY * (a[None, None, :] * P + beta * U * N2[:, :, None]) * GS).sum((0, 1))
    dO = dO_full
    G = np.zeros((B, D, D), X.dtype)              # dL/dS_t
    da = np.zeros(D, X.dtype)
    Scur = Sfin
    for t in range(T - 1, -1, -1):
        v = X[:, t]; k = KK[:, t]; kn = KN[:, t]
        u = U[:, t]
        S = (Scur - beta * u[:, :, None] * k[:, None, :]) / a[None, :, None]   # S_{t-1} точно
        p = P[:, t]; n2 = N2[:, t]
        dOt = dO[:, t]
        da += (dOt * p).sum(0)
        dbeta += float((dOt * u * n2[:, None]).sum())
        dn2 = (dOt * (beta * u)).sum(-1)
        da += (G * S).sum((0, 2))
        dbeta += float((G * (u[:, :, None] * k[:, None, :])).sum())
        du = dOt * beta * n2[:, None] + beta * np.einsum('bij,bj->bi', G, k)
        dp = dOt * a[None, :] - du
        dk = 2.0 * k * dn2[:, None] + beta * np.einsum('bij,bi->bj', G, u)
        dk_tot = dk + np.einsum('bij,bi->bj', S, dp)
        # выходной гейт: d/dy_raw = yō·sc·o·g(1−g); в X-токен через Wg
        o_t = a[None, :] * p + beta * u * n2[:, None]
        dgr = dY[:, t] * sc * o_t * GS[:, t] * (1 - GS[:, t])
        dWg += (v.T @ dgr)                        # v = x_t сырой (B,D)·(B,D)
        dX[:, t] += dgr @ Wg.T
        dX[:, t] += du + (dk_tot - k * (dk_tot * k).sum(-1, keepdims=True)) / kn[:, None]
        G = a[None, :, None] * G + np.einsum('bi,bj->bij', dp, k)
        Scur = S
    dth = da * a * (1 - a)
    dbraw = np.array(dbeta * beta * (1 - beta), X.dtype)
    return dX, dth, dsc, dbraw, dWg

# ---------------------------------------------------------------- MoE-FFN (DeepSeek-style, aux-loss-free)
MOE_STATS = {}                                   # running loads для bias-апдейта (не градиентно)

def moe_ffn(X, Wr, be, W1, W2, key):           # X (B,T,D); W1 (E,D,F2); W2 (E,F2,D)
    B, T, D = X.shape; E = Wr.shape[1]; F2 = W1.shape[2]
    s = X @ Wr                                   # (B,T,E) affinity (сырые)
    sel = (s + be).argmax(-1)                    # hard top-1 с bias-коррекцией (aux-loss-free)
    sg = 1.0 / (1.0 + np.exp(-s))
    gv = np.take_along_axis(sg, sel[..., None], -1)[..., 0]      # gate = σ(s_sel), bias в выбор НЕ в значение
    with np.errstate(over='ignore'):
        pass
    out = np.zeros_like(X)
    for e in range(E):
        m = sel == e
        if not m.any(): continue
        xe = X[m]
        h = gelu(xe @ W1[e])[0] @ W2[e]
        out[m] = gv[m, None] * h
    loads = np.bincount(sel.reshape(-1), minlength=E) / float(B * T)
    MOE_STATS.setdefault(key, np.full(E, 1.0 / E))
    MOE_STATS[key] = 0.99 * MOE_STATS[key] + 0.01 * loads
    return out, (X, Wr, be, W1, W2, sel, sg, gv, s)

def moe_ffn_bwd(c, dOut):
    X, Wr, be, W1, W2, sel, sg, gv, s = c
    B, T, D = X.shape; E = Wr.shape[1]; F2 = W1.shape[2]
    dX = np.zeros_like(X); dWr = np.zeros_like(Wr)
    dW1 = np.zeros_like(W1); dW2 = np.zeros_like(W2)
    ds = np.zeros_like(s)
    for e in range(E):
        m = sel == e
        if not m.any(): continue
        xe = X[m]; dO = dOut[m]
        dgv = (dO * (gelu(xe @ W1[e])[0] @ W2[e])).sum(-1)         # (N,)
        dh = dO * gv[m, None]
        z1, cg = gelu(xe @ W1[e])
        dz1 = gelu_bwd(cg, dh @ W2[e].T)
        dW2[e] += z1.T @ dh
        dW1[e] += xe.T @ dz1
        dX[m] += dz1 @ W1[e].T
        dsel = dgv * gv[m] * (1 - gv[m])
        ds[..., e][m] += dsel
        # gate σ(s_sel): вклад в остальные эксперты = 0 (top-1 hard, как и выбор)
    dWr = X.reshape(-1, D).T @ ds.reshape(-1, E)
    dX += ds @ Wr.T
    dbe = ds.sum((0, 1)) * 0.0                     # bias обновляется running-статистикой
    return dX, dWr, dbe, dW1, dW2

def main():
    args = ap_parse()
    tr = np.load(f"{ROOT}/{args.data}/train.npy").astype(np.int64)
    va = np.load(f"{ROOT}/{args.data}/val.npy").astype(np.int64)
    V = json.load(open(f"{ROOT}/{args.data}/meta.json"))["vocab"]
    rng = np.random.default_rng(args.seed)
    assert not (args.erank > 0 and args.mtp > 0), "erank+mtp пока не комбинируются"
    model = NanoGPT(V, D=args.dim, L=args.layers, ff=args.ff, kind=args.kind, adr_kf=args.adr, moe_e=args.moe, mtp_w=args.mtp, erank=args.erank, kronfc=args.kronfc)
    teacher = None
    if args.distill:
        teacher = NanoGPT(V)
        td = np.load(args.distill)
        for k in teacher.p.d:
            if k in td.files: teacher.p.d[k][...] = td[k]
        print(f"[{args.tag}] учитель {args.distill} загружен", flush=True)
    if args.initckpt:
        d0 = np.load(args.initckpt)
        for k in model.p.d:
            if k in d0.files: model.p.d[k][...] = d0[k]
        print(f"[{args.tag}] init from {args.initckpt}", flush=True)
    nparams = sum(v.size for v in model.p.d.values())
    print(f"[{args.tag}] nano-{args.kind} params={nparams:,}", flush=True)
    if args.opt == "muon": opt = MuonW(model.p, lr=args.lr, mulr=args.mulr)
    else: opt = AdamW(model.p, lr=args.lr, wdmask=bool(args.wdmask))

    ew = None
    if args.wema > 0:
        ew = {k: np.zeros_like(v) for k, v in model.p.d.items()}
        ew_corr = 1.0
        print(f"[{args.tag}] weight-EMA decay={args.wema}: двойная оценка val (live + ema)", flush=True)

    # ---- sampled softmax: предложение q = unigram(train), коррекция Q_c = 1−(1−q)^K (with-replacement)
    ssq = None
    if args.ssk > 0:
        assert teacher is None, "ssk: distill пока не комбинируется"
        assert args.mtp == 0, "ssk: mtp пока не комбинируется"
        cnt = np.bincount(tr, minlength=V).astype(np.float64) + 1.0
        ssq = cnt ** args.ssalpha
        ssq = ssq / ssq.sum()
        print(f"[{args.tag}] sampled softmax: K={args.ssk} негативов/шаг, unigram-предложение, "
              f"logQ-коррекция; вал — полный softmax", flush=True)

    def batch(ids, rr):
        st = rr.integers(0, len(ids) - args.ctx - 1, size=args.batch)
        x = np.stack([ids[s:s + args.ctx] for s in st]); y = np.stack([ids[s + 1:s + args.ctx + 1] for s in st])
        return x, y

    def vloss(iters=6):
        rr = np.random.default_rng(1234); tot = 0.0
        for _ in range(iters):
            x, y = batch(va, rr)
            H, _ = model.forward(x)
            l, _ = softmax_ce(model.logits(H), y, destroy=True)
            tot += l
        return tot / iters

    os.makedirs(f"{ROOT}/results", exist_ok=True)
    logf = open(f"{ROOT}/results/run_{args.tag}.jsonl", "w")
    t0 = time.time(); toks = 0
    for step in range(args.steps):
        lr = args.lr * min((step + 1) / args.warmup,
              0.1 + 0.45 * (1 + math.cos(math.pi * max(0, step - args.warmup) / max(1, args.steps - args.warmup))))
        _t = time.process_time_ns()
        x, y = batch(tr, rng)
        _t = _tick("batch", _t)
        H, caches = model.forward(x)
        _t = _tick("fwd", _t)
        head_cache = None
        use_ss = ssq is not None and step < args.steps * (1.0 - args.ssfull)
        if use_ss:
            neg = rng.choice(V, size=args.ssk, replace=True, p=ssq)
            cand = np.union1d(np.unique(y), neg).astype(np.int64)
            logQ = np.log1p(-np.power(1.0 - ssq[cand], args.ssk))      # log(1−(1−q)^K)
            logQ[np.isin(cand, y)] = 0.0                               # цели всегда в наборе
            Ec0 = (model.p.d["U"][cand] @ model.p.d["P"]) if model.erank > 0 else None
            loss, (dz, cand, Ec) = sampled_ce(H, y, model.p.d["E"] if model.erank == 0 else Ec0, cand, logQ, Ec=Ec0)
            head_cache = (cand, dz, Ec)
            dz = None
        else:
            lg = model.logits(H)
            loss, dz = softmax_ce(lg, y, destroy=(teacher is None))
        _t = _tick("head_fwd", _t)
        if teacher is not None:
            Ht, _ = teacher.forward(x)
            lgt = Ht @ teacher.p.d["E"].T
            zt = lgt / args.dtemp; q = np.exp(zt - zt.max(-1, keepdims=True)); q /= q.sum(-1, keepdims=True)
            zs = lg / args.dtemp; ps = np.exp(zs - zs.max(-1, keepdims=True)); ps /= ps.sum(-1, keepdims=True)
            n = lg.shape[0] * lg.shape[1]
            kl = float((q * (np.log(q + 1e-12) - np.log(ps + 1e-12))).sum() / n)
            dz = dz + args.dkl * (args.dtemp ** 2) * (ps - q) / n
            loss = loss + args.dkl * kl
        dH_aux = None
        if args.mtp > 0:
            y2 = np.concatenate([y[:, 1:], x[:, -1:]], axis=1)     # t+2 цель (последний — фиктив, вес 0)
            zn, clnm = layernorm(H, model.p.d["lnm_g"], model.p.d["lnm_b"])
            zm, cwm = linear(zn, model.p.d["Wmtp"])
            lg2 = model.logits(zm)
            loss2, dz2 = softmax_ce(lg2, y2)
            dz2[:, -1] = 0                                        # фиктивная цель
            loss = loss + args.mtp * loss2
            dz_m = dz2 @ model.p.d["E"]                           # через tied-E
            # точнее: lg2 = zm @ E.T → dE += dz2ᵀ·zm; dzm = dz2 @ E
            model.p.g["E"] += args.mtp * dz2.reshape(-1, model.V).T @ zm.reshape(-1, model.D)
            dzm = dz2 @ model.p.d["E"]
            dzn, dWmtp = linear_bwd(cwm, args.mtp * dzm); model.p.g["Wmtp"] += dWmtp
            dH_aux, dgm, dbm = layernorm_bwd(clnm, dzn)
            model.p.g["lnm_g"] += dgm; model.p.g["lnm_b"] += dbm
        model.backward(H, caches, dz, dH_aux=dH_aux, head_cache=head_cache)
        _t = _tick("bwd_total", _t)
        if args.moe > 1:
            for i in range(model.L):
                be = model.p.d[f"b{i}.be"]
                be += 0.02 * (0.25 - MOE_STATS[i])         # bias ↑ недогруженным экспертам
        opt.step(lr)
        _t = _tick("opt", _t)
        if ew is not None:
            d = args.wema
            for k in ew:
                ew[k] = d * ew[k] + (1.0 - d) * model.p.d[k]
            ew_corr = 1.0 - d ** (step + 1)
        toks += x.size
        if step % args.eval_every == 0 or step == args.steps - 1:
            vl = vloss(); el = time.time() - t0
            rec = dict(step=step, train_loss=round(float(loss), 4), val_loss=round(float(vl), 4),
                       val_ppl=round(float(math.exp(vl)), 2), wall=round(el, 1),
                       tok_s=round(toks / el, 1), tokens=toks)
            if ew is not None:
                bk = {k: model.p.d[k].copy() for k in ew}
                for k in ew: model.p.d[k][...] = ew[k] / ew_corr      # bias-коррекция EMA
                vle = vloss()
                for k in ew: model.p.d[k][...] = bk[k]
                rec["val_ppl_ema"] = round(float(math.exp(vle)), 2)
            print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
    np.savez(f"{ROOT}/results/ckpt_{args.tag}.npz", **model.p.d)
    if ew is not None:
        np.savez(f"{ROOT}/results/ckpt_{args.tag}_ema.npz", **{k: ew[k] / ew_corr for k in ew})
    print("SUMMARY " + json.dumps(dict(tag=args.tag, kind=args.kind, params=nparams,
        tokens=toks, wall_s=round(time.time() - t0, 1), tok_s=round(toks / (time.time() - t0), 1),
        final_val_loss=rec["val_loss"], final_val_ppl=rec["val_ppl"])), flush=True)
    if os.environ.get("NANOLC_PROF"):
        tot = sum(np.median(v) for v in PROF.values()) / 1e6
        print("PROF медианы ms/шаг:", flush=True)
        for k in sorted(PROF, key=lambda k: -np.median(PROF[k])):
            v = np.median(PROF[k]) / 1e6
            print(f"  {k:12s} {v:8.1f} ({100*v/tot:.1f}%)", flush=True)
        print(f"  {'ИТОГО':12s} {tot:8.1f}", flush=True)

def ap_parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["attn", "ema", "hybrid", "delta"], required=True)
    ap.add_argument("--opt", choices=["adam", "muon"], default="adam")
    ap.add_argument("--wdmask", type=int, default=0, help="AdamW: wd только на скрытых 2D-матрицах")
    ap.add_argument("--wema", type=float, default=0.0, help="decay EMA весов для оценки (0=выкл)")
    ap.add_argument("--mulr", type=float, default=0.02)
    ap.add_argument("--moe", type=int, default=1, help="E экспертов top-1 (DeepSeek aux-loss-free); 1 = dense FFN")
    ap.add_argument("--kronfc", type=int, default=0, help="K кронекер-членов в FFN fc1/fc2 (0=dense)")
    ap.add_argument("--mtp", type=float, default=0.0, help="вес MTP-aux головы t+2 (DeepSeek); 0 — выкл")
    ap.add_argument("--dim", type=int, default=192); ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ff", type=int, default=576)
    ap.add_argument("--distill", default=None, help="ckpt учителя (fp32 ema 192x4) для KL-дистилляции")
    ap.add_argument("--dtemp", type=float, default=2.0); ap.add_argument("--dkl", type=float, default=0.5)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--ctx", type=int, default=96)
    ap.add_argument("--adr", type=float, default=None)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--initckpt", default=None)
    ap.add_argument("--data", default="data/prep")
    ap.add_argument("--ssk", type=int, default=0, help="sampled softmax: K негативов/шаг (0 = полный CE)")
    ap.add_argument("--ssfull", type=float, default=0.0, help="доля финальных шагов с полным CE (аннил)")
    ap.add_argument("--ssalpha", type=float, default=1.0, help="степень сглаживания unigram-предложения (word2vec 0.75)")
    ap.add_argument("--erank", type=int, default=0, help="низкоранговый tied-эмбеддинг U(V,r)@P(r,D); 0 = выкл")
    return ap.parse_args()

if __name__ == "__main__":
    main()
