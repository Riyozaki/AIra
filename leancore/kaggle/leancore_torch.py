# -*- coding: utf-8 -*-
"""leancore_torch.py — верный GPU-порт LeanCore (nano_lc/np_qat) под Kaggle.

Математика 1:1 с numpy/C версией (проверено gradcheck'ом там же):
  блок:  x += g·( mix + ffn2(gelu(ffn1(ln2(x+mix)))) )  при ADR-роутинге top-k по σ(x·rw+rb)
  миксер EMA:  a = σ(th);  h_t = a·h_{t−1} + (1−a)·x_t;  y = sc⊙h;  mix = y @ Wm
        (на GPU строится треугольная матрица M[t,k,d] = (1−a_d)a_d^{t−k} и einsum)
  head:  logits = H @ Eᵀ  (tied)
QAT-STE:  тернаризация meanabs per-out для fc1/fc2/Wm через detach-трюк.
Потоковый экспорт: npz → (в отдельной ячейке) LCW2-share8 для C-движка из репо.
"""
import math, os, json, time
import torch, torch.nn as nn, torch.nn.functional as F

f32 = torch.float32


# ---------------------------------------------------------------- Muon (Keller Jordan)
@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps=5):
    """Ортогонализация обновления, bf16-хвост допустим; форма (r,c)."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    transposed = G.size(0) > G.size(1)
    if transposed: X = X.mT
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed: X = X.mT
    return X


class MuonW:
    """Muon для 2D скрытых матриц (по белому списку имён), AdamW для остального."""
    def __init__(self, named_params, muon_keys=(".Wm", ".fc1", ".fc2"), lr=6e-4, mulr=0.02,
                 adam_betas=(0.85, 0.95), adam_eps=1e-8, wd=0.0):
        self.lr, self.mulr = lr, mulr
        self.mu, self.ad = [], []          # (name, param, kind)
        for n, p in named_params:
            kind = "muon" if (p.ndim == 2 and any(k in n for k in muon_keys)) else "adam"
            (self.mu if kind == "muon" else self.ad).append((n, p))
            if kind == "muon":
                st = {"buf": torch.zeros_like(p)}
            else:
                st = {"m": torch.zeros_like(p), "v": torch.zeros_like(p), "t": 0}
            self.state = getattr(self, "state", {})
            self.state[p] = st
        self.b1, self.b2, self.eps, self.wd = adam_betas[0], adam_betas[1], adam_eps, wd
        self.nesterov = True
        print(f"[MuonW] muon: {len(self.mu)} шт, adam: {len(self.ad)} шт", flush=True)

    @torch.no_grad()
    def step(self, lr=None):
        lr = self.lr if lr is None else lr
        for n, p in self.mu:
            g = p.grad
            st = self.state[p]; buf = st["buf"]
            buf.lerp_(g, 1 - 0.95)                     # momentum 0.95
            u = (g.lerp(buf, 0.95) if self.nesterov else buf)
            O = zeropower_via_newtonschulz5(u).to(p.dtype)
            scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
            p.add_(O, alpha=-lr * self.mulr * scale)   # mulr=0.02 как в numpy-версии
        for n, p in self.ad:
            g = p.grad; st = self.state[p]
            st["t"] += 1; t = st["t"]
            st["m"].lerp_(g, 1 - self.b1); st["v"].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            mh = st["m"] / (1 - self.b1 ** t); vh = st["v"] / (1 - self.b2 ** t)
            p.addcdiv_(mh, vh.sqrt().add_(self.eps), value=-lr)
            if self.wd: p.mul_(1 - lr * self.wd)


# ---------------------------------------------------------------- EMA-миксер (треугольная Σ-форма)
def ema_mix(X, th, sc):
    """X (B,T,D). h_t = a⊙h_{t−1} + (1−a)⊙x_t; y = sc⊙h. Возвращает (y)."""
    B, T, D = X.shape
    a = torch.sigmoid(th)                                   # (D,)
    tt = torch.arange(T, device=X.device)
    dd = (tt[:, None] - tt[None, :]).clamp(min=0).to(f32)   # t−k, k≤t
    alog = torch.log(a.clamp_min(1e-20))
    P = torch.exp(dd[:, :, None] * alog[None, None, :])     # a^{t−k}
    mask = (tt[:, None] >= tt[None, :]).to(f32)
    M = P * mask[:, :, None] * (1 - a)[None, None, :]       # h_t = Σ_k M·x_k
    H = torch.einsum('tkd,bkd->btd', M, X)
    return H * sc


class Block(nn.Module):
    def __init__(self, D, ff, routed=False):
        super().__init__()
        self.routed = routed
        self.ln1 = nn.LayerNorm(D); self.ln2 = nn.LayerNorm(D)
        self.th = nn.Parameter(torch.zeros(D)); self.sc = nn.Parameter(torch.ones(D))
        self.Wm = nn.Parameter(torch.empty(D, D)); nn.init.normal_(self.Wm, std=0.02)
        self.fc1 = nn.Parameter(torch.empty(D, ff)); nn.init.normal_(self.fc1, std=0.02)
        self.fc2 = nn.Parameter(torch.empty(ff, D)); nn.init.normal_(self.fc2, std=0.02)
        if routed:
            self.rw = nn.Parameter(torch.empty(D)); nn.init.normal_(self.rw, std=0.02)
            self.rb = nn.Parameter(torch.tensor(1.5))
        # LayerNorm как {w=1,b=0} по умолчанию — совпадает с numpy-инициализацией

    def arm(self, x):
        """x (B,k,D) → delta (B,k,D): ln1→EMA→Wm→+resid→ln2→FFN; вернуть mix+ffn_out."""
        ln1 = self.ln1(x)
        mix = ema_mix(ln1, self.th, self.sc) @ self.Wm      # (B,k,D)
        h2 = x + mix
        z = self.ln2(h2)
        o2 = F.gelu(z @ self.fc1, approximate='tanh') @ self.fc2
        return mix + o2

    def forward(self, x, kfrac=None):
        """x (B,T,D). Если routed: top-k позиций по σ(x·rw+rb), gate σ, остальные наследуют вход."""
        if not self.routed or kfrac is None:
            return x + self.arm(x)
        B, T, D = x.shape
        s = (x * self.rw).sum(-1) + self.rb                 # (B,T)
        k = max(1, int(round(kfrac * T)))
        idx = s.topk(k, dim=1).indices.sort(dim=1).values   # (B,k)
        xs = torch.gather(x, 1, idx[:, :, None].expand(B, k, D))
        gs = torch.sigmoid(torch.gather(s, 1, idx))         # (B,k)
        delta = self.arm(xs) * gs[:, :, None]
        return x.scatter_add(1, idx[:, :, None].expand(B, k, D), delta)


class LeanCore(nn.Module):
    def __init__(self, V, D=192, L=4, ff=576, T=96, adr_kf=0.5):
        super().__init__()
        self.V, self.D, self.L, self.T, self.adr_kf = V, D, L, T, adr_kf
        self.E = nn.Parameter(torch.empty(V, D)); nn.init.normal_(self.E, std=0.02)
        self.pos = nn.Parameter(torch.empty(T, D)); nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(D, ff, routed=(adr_kf is not None and i > 0))
                                     for i in range(L)])
        self.lnf = nn.LayerNorm(D)

    def forward(self, ids):
        B, T = ids.shape
        h = self.E[ids] + self.pos[:T][None]
        for b in self.blocks:
            h = b(h, kfrac=self.adr_kf)
        return self.lnf(h)

    def logits(self, h):
        return h @ self.E.t()                               # tied head


# ---------------------------------------------------------------- QAT: тернарный STE (np_qat-протокол)
TERN_KEYS = (".Wm", ".fc1", ".fc2")

def tern_meanabs(w):
    s = w.abs().mean(dim=-1, keepdim=True).clamp_min(1e-5)
    return (torch.clamp(torch.round(w / s), -1, 1) * s)

class QAT:
    """np_qat-протокол 1:1. Цикл:  quantize_() → fwd/bwd → opt.step() → absorb_()  (eval: apply_())."""
    def __init__(self, model):
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.ndim == 2 and any(k in n for k in TERN_KEYS):
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def quantize_(self, model):                             # p ← tern(shadow)  (точка измерения градиента)
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.copy_(tern_meanabs(self.shadow[n]))

    @torch.no_grad()
    def absorb_(self, model):                               # shadow += (p_after_opt − tern(shadow)); p ← shadow
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].add_(p - tern_meanabs(self.shadow[n]))
                p.copy_(self.shadow[n])

    @torch.no_grad()
    def apply_(self, model):                                # для eval/экспорта: p ← tern(shadow)
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.copy_(tern_meanabs(self.shadow[n]))


# ---------------------------------------------------------------- экспорт в npz (совместимо с репо)
def to_numpy_sd(model):
    sd = model.state_dict()
    out = {}
    out["E"] = model.E.detach().cpu().numpy().astype("float32")
    out["pos"] = model.pos.detach().cpu().numpy().astype("float32")
    for i, b in enumerate(model.blocks):
        pr = f"b{i}."
        out[pr + "ln1g"] = b.ln1.weight.detach().cpu().numpy().astype("float32")
        out[pr + "ln1b"] = b.ln1.bias.detach().cpu().numpy().astype("float32")
        out[pr + "th"] = b.th.detach().cpu().numpy().astype("float32")
        out[pr + "sc"] = b.sc.detach().cpu().numpy().astype("float32")
        # наши W в torch хранятся как (in,out) — numpy-версия тоже (in,out): совпадает
        out[pr + "Wm"] = b.Wm.detach().cpu().numpy().astype("float32")
        out[pr + "ln2g"] = b.ln2.weight.detach().cpu().numpy().astype("float32")
        out[pr + "ln2b"] = b.ln2.bias.detach().cpu().numpy().astype("float32")
        out[pr + "fc1"] = b.fc1.detach().cpu().numpy().astype("float32")
        out[pr + "fc2"] = b.fc2.detach().cpu().numpy().astype("float32")
        if b.routed:
            out[pr + "rw"] = b.rw.detach().cpu().numpy().astype("float32")
            out[pr + "rb"] = b.rb.detach().cpu().numpy().astype("float32")
    out["lnfg"] = model.lnf.weight.detach().cpu().numpy().astype("float32")
    out["lnfb"] = model.lnf.bias.detach().cpu().numpy().astype("float32")
    return out

def save_npz(model, path):
    import numpy as np
    np.savez(path, **to_numpy_sd(model))
    print("saved", path, flush=True)
