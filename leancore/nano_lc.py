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

ROOT = os.path.dirname(os.path.abspath(__file__))
f32 = np.float32

class Params:
    def __init__(self): self.d, self.g = {}, {}
    def add(self, name, shape, std=0.02, init=None):
        rng = np.random.default_rng(abs(hash(name)) % (2**32))
        self.d[name] = (rng.normal(0, std, shape).astype(f32) if init is None else init(shape))
        self.g[name] = np.zeros_like(self.d[name])
    def zero(self):
        for k in self.g: self.g[k][...] = 0

# ---------------------------------------------------------------- core ops (fwd + hand-derived bwd)
def gelu(x):                                   # tanh-approx gelu
    k0 = 0.7978845608028654
    u = k0 * (x + 0.044715 * x**3)
    t = np.tanh(u)
    return 0.5 * x * (1 + t), (x, t)
def gelu_bwd(c, dy):
    x, t = c; k0 = 0.7978845608028654
    u = k0 * (x + 0.044715 * x**3)
    du = k0 * (1 + 3 * 0.044715 * x**2)
    return dy * (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * du)

def layernorm(x, g, b, eps=1e-5):              # x: (...,N)
    N = x.shape[-1]
    mu = x.mean(-1, keepdims=True); xc = x - mu
    var = (xc**2).mean(-1, keepdims=True); istd = 1.0 / np.sqrt(var + eps)
    xhat = xc * istd
    return g * xhat + b, (xhat, istd, g, N)
def layernorm_bwd(c, dy):
    xhat, istd, g, N = c
    dg = (dy * xhat).sum(axis=tuple(range(dy.ndim - 1)))
    db = dy.sum(axis=tuple(range(dy.ndim - 1)))
    v = dy * g                                                   # γ ДО редукций
    dx = (istd / N) * (N * v - v.sum(-1, keepdims=True) - xhat * (v * xhat).sum(-1, keepdims=True))
    return dx, dg, db

def linear(x, W):                              # (..., in) @ (in,out)
    return x @ W, (x, W)
def linear_bwd(c, dy):
    x, W = c
    dw = x.reshape(-1, x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
    return dy @ W.T, dw

def softmax_ce(logits, y):                     # logits (B,T,V), y (B,T) -> (loss, dlogits)
    z = logits - logits.max(-1, keepdims=True)
    e = np.exp(z); p = e / e.sum(-1, keepdims=True)
    B, T, V = logits.shape
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

def ema_mix(X, th, sc):                        # h_t = a·h_{t-1} + (1-a)·x_t; y = sc⊙h
    a = 1.0 / (1.0 + np.exp(-th))
    B, T, D = X.shape
    H = np.empty_like(X)
    hprev = np.zeros((B, D), X.dtype)
    for t in range(T):
        hprev = a * hprev + (1.0 - a) * X[:, t]
        H[:, t] = hprev
    return H * sc, (X, H, a, sc)
def ema_mix_bwd(c, dY):
    X, H, a, sc = c
    dH = dY * sc
    dsc = (dY * H).sum(axis=(0, 1))
    dX = np.empty_like(X); da = np.zeros(X.shape[2], X.dtype)
    lam = np.zeros((X.shape[0], X.shape[2]), X.dtype)
    for t in range(X.shape[1] - 1, -1, -1):
        lam = dH[:, t] + a * lam
        dX[:, t] = (1.0 - a) * lam
        hprev = H[:, t - 1] if t > 0 else np.zeros_like(lam)
        da += (lam * (hprev - X[:, t])).sum(0)
    dth = da * a * (1.0 - a)
    return dX, dth, dsc

# ---------------------------------------------------------------- model
class NanoGPT:
    def __init__(self, V, D=192, L=4, h=6, ff=576, T=96, kind="attn", adr_kf=None):
        self.V, self.D, self.L, self.h, self.T, self.ff = V, D, L, h, T, ff
        self.kind, self.adr_kf = kind, adr_kf
        p = self.p = Params()
        p.add("E", (V, D)); p.add("pos", (T, D))
        self.blocks = []
        for i in range(L):
            kind_i = "attn" if (kind == "attn" or (kind == "hybrid" and i == 0)) else "ema"
            b = {"kind": kind_i, "routed": bool(adr_kf) and i > 0}
            self.blocks.append(b)
            p.add(f"b{i}.ln1g", (D,), init=lambda s: np.ones(s, f32))
            p.add(f"b{i}.ln1b", (D,), init=lambda s: np.zeros(s, f32))
            if kind_i == "attn":
                p.add(f"b{i}.Wqkv", (D, 3 * D)); p.add(f"b{i}.Wo", (D, D))
            else:
                p.add(f"b{i}.th", (D,), init=lambda s: np.zeros(s, f32))     # a=σ(0)=0.5
                p.add(f"b{i}.sc", (D,), init=lambda s: np.ones(s, f32))
                p.add(f"b{i}.Wm", (D, D))
            p.add(f"b{i}.ln2g", (D,), init=lambda s: np.ones(s, f32))
            p.add(f"b{i}.ln2b", (D,), init=lambda s: np.zeros(s, f32))
            p.add(f"b{i}.fc1", (D, ff)); p.add(f"b{i}.fc2", (ff, D))
            if b["routed"]:
                p.add(f"b{i}.rw", (D,))
                p.add(f"b{i}.rb", (), init=lambda s: np.array(1.5, f32))
        p.add("lnfg", (D,), init=lambda s: np.ones(s, f32))
        p.add("lnfb", (D,), init=lambda s: np.zeros(s, f32))

    def _arm_forward(self, X2, i, b):
        """плечо блока: ln1 → mixer(+Wm/attn) → +resid → ln2 → ffn → delta=mix+o2"""
        p = self.p.d; c = {}
        ln_out, c["ln1"] = layernorm(X2, p[f"b{i}.ln1g"], p[f"b{i}.ln1b"])
        if b["kind"] == "attn":
            mix, c["attn"] = attention(ln_out, p[f"b{i}.Wqkv"], p[f"b{i}.Wo"], self.h)
        else:
            em, c["ema"] = ema_mix(ln_out, p[f"b{i}.th"], p[f"b{i}.sc"])
            mix, c["wm"] = linear(em, p[f"b{i}.Wm"])
        H2 = X2 + mix
        ln_out2, c["ln2"] = layernorm(H2, p[f"b{i}.ln2g"], p[f"b{i}.ln2b"])
        c["fc1_in"] = ln_out2
        g1, c["gelu"] = gelu(ln_out2 @ p[f"b{i}.fc1"])
        o2, c["fc2"] = linear(g1, p[f"b{i}.fc2"])
        c["delta"] = mix + o2
        return mix + o2, c

    def _arm_backward(self, c, dD, i, b):
        """dD = dL/dDelta → возвращает dL/dX2 (вход плеча)."""
        p, g = self.p.d, self.p.g
        dX2f, dW2 = linear_bwd(c["fc2"], dD); g[f"b{i}.fc2"] += dW2
        dpre = gelu_bwd(c["gelu"], dX2f)
        g[f"b{i}.fc1"] += c["fc1_in"].reshape(-1, self.D).T @ dpre.reshape(-1, self.ff)
        dln2 = dpre @ p[f"b{i}.fc1"].T
        dx2, dg2, db2 = layernorm_bwd(c["ln2"], dln2); g[f"b{i}.ln2g"] += dg2; g[f"b{i}.ln2b"] += db2
        dmix = dD + dx2                            # mix питает Δ и H2 (вход ln2)
        if b["kind"] == "attn":
            dxm, dWqkv, dWo = attention_bwd(c["attn"], dmix)
            g[f"b{i}.Wqkv"] += dWqkv; g[f"b{i}.Wo"] += dWo
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

    def logits(self, H): return H @ self.p.d["E"].T                   # tied head

    def backward(self, H, caches, dlogits):
        p, g = self.p.d, self.p.g
        ids, T, blocks_c, c_lnf = caches
        g["E"] += dlogits.reshape(-1, self.V).T @ H.reshape(-1, self.D)
        dH = dlogits @ p["E"]
        dH, dg, db = layernorm_bwd(c_lnf, dH); g["lnfg"] += dg; g["lnfb"] += db
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
        np.add.at(g["E"], ids, dH)          # градиент эмбеддинга (lookup-путь)
        g["pos"] += dH.sum(0)
        return H

class AdamW:
    def __init__(self, params, lr=6e-4, b1=0.9, b2=0.95, wd=0.1, eps=1e-8):
        self.p = params; self.b1, self.b2, self.wd, self.eps = b1, b2, wd, eps
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
            self.p.d[k] -= lr * (mh / (np.sqrt(vh) + self.eps) + self.wd * self.p.d[k])
        self.p.zero()

def main():
    args = ap_parse()
    tr = np.load(f"{ROOT}/{args.data}/train.npy").astype(np.int64)
    va = np.load(f"{ROOT}/{args.data}/val.npy").astype(np.int64)
    V = json.load(open(f"{ROOT}/{args.data}/meta.json"))["vocab"]
    rng = np.random.default_rng(42)
    model = NanoGPT(V, kind=args.kind, adr_kf=args.adr)
    if args.initckpt:
        d0 = np.load(args.initckpt)
        for k in model.p.d:
            if k in d0.files: model.p.d[k][...] = d0[k]
        print(f"[{args.tag}] init from {args.initckpt}", flush=True)
    nparams = sum(v.size for v in model.p.d.values())
    print(f"[{args.tag}] nano-{args.kind} params={nparams:,}", flush=True)
    opt = AdamW(model.p, lr=args.lr)

    def batch(ids, rr):
        st = rr.integers(0, len(ids) - args.ctx - 1, size=args.batch)
        x = np.stack([ids[s:s + args.ctx] for s in st]); y = np.stack([ids[s + 1:s + args.ctx + 1] for s in st])
        return x, y

    def vloss(iters=6):
        rr = np.random.default_rng(1234); tot = 0.0
        for _ in range(iters):
            x, y = batch(va, rr)
            H, _ = model.forward(x)
            l, _ = softmax_ce(model.logits(H), y)
            tot += l
        return tot / iters

    os.makedirs(f"{ROOT}/results", exist_ok=True)
    logf = open(f"{ROOT}/results/run_{args.tag}.jsonl", "w")
    t0 = time.time(); toks = 0
    for step in range(args.steps):
        lr = args.lr * min((step + 1) / args.warmup,
              0.1 + 0.45 * (1 + math.cos(math.pi * max(0, step - args.warmup) / max(1, args.steps - args.warmup))))
        x, y = batch(tr, rng)
        H, caches = model.forward(x)
        lg = model.logits(H)
        loss, dz = softmax_ce(lg, y)
        model.backward(H, caches, dz)
        opt.step(lr)
        toks += x.size
        if step % args.eval_every == 0 or step == args.steps - 1:
            vl = vloss(); el = time.time() - t0
            rec = dict(step=step, train_loss=round(float(loss), 4), val_loss=round(float(vl), 4),
                       val_ppl=round(float(math.exp(vl)), 2), wall=round(el, 1),
                       tok_s=round(toks / el, 1), tokens=toks)
            print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
    np.savez(f"{ROOT}/results/ckpt_{args.tag}.npz", **model.p.d)
    print("SUMMARY " + json.dumps(dict(tag=args.tag, kind=args.kind, params=nparams,
        tokens=toks, wall_s=round(time.time() - t0, 1), tok_s=round(toks / (time.time() - t0), 1),
        final_val_loss=rec["val_loss"], final_val_ppl=rec["val_ppl"])), flush=True)

def ap_parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["attn", "ema", "hybrid"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--ctx", type=int, default=96)
    ap.add_argument("--adr", type=float, default=None)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--initckpt", default=None)
    ap.add_argument("--data", default="data/prep")
    return ap.parse_args()

if __name__ == "__main__":
    main()
