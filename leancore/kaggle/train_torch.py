#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_torch.py — torch-движок турнира с семантикой nano_lc 1:1 (насколько позволяет fp32-порядок редукций).
Семантика (намеренно продублирована, НЕ импортируется из leancore_torch.py — там lr-семантика
и беты MuonW расходятся с оригиналом):
  блок (ema):  x += ln1→EMA(a=σ(th), y=sc·h)@Wm → +x → ln2 → fc1→gelu(tanh)→fc2  [ADR off]
  затем lnf; голова tied E. MuonW: muon 2D-маршруты (.Wm/.fc1/.fc2, ndim2, min≥16, кроме E/U/pos),
  обновление p·=(1−wd·lr); p−=mulr·scale·NS5(u) (mulr БЕЗ умножения на lr — как nano_lc);
  adam-ветка: b1=0.9, b2=0.95, eps=1e-8, wd=0.1·lr·p.
  LR:  lr·min((s+1)/warmup, 0.1+0.45(1+cos(π·max(0,s−warmup)/(steps−warmup)))).
  SSK:  негативы unigram cnt**ssalpha (host numpy!), кандидаты = union(цели, neg), logQ-коррекция
        log(1−(1−q)^K), цели обнуляют коррекцию; ssfull — доля финальных шагов с полным CE.
  RNG:  данные — host numpy default_rng(seed); негативы — negrng=1 → отдельный поток
        default_rng(seed·1000003+17); вал — default_rng(1234) фиксирован.
  ИНИТ: torch.manual_seed(0x1EA7) — ФИКСИРОВАН, не зависит от --seed (зеркало crc32-инициализации;
        парность конфигов = одинаковый инит у всех).
Поток: host-numpy данные → gpu-тензоры; autograd через E[cand] сам разбрасывает градиент головы.
"""
import os, sys, json, math, time, argparse
import numpy as hnp
import torch
import torch.nn as nn
import torch.nn.functional as F

INIT_SEED = 0x1EA7          # фиксированный инит для ВСЕХ конфигов (парность брекета)
f32 = torch.float32


def np_paired_init(model, hnp, zlib):
    """Битово-парный инит с numpy-тренером: тот же default_rng(crc32(имя)) и те же
    спец-иниты (ln=ones/zeros, th=0, sc=1). Без этого torch и numpy стартовали из
    РАЗНЫХ случайных точек и GPU-гейт измерял шум инита, а не расхождение движков
    [измерено на Kaggle: relΔ@40шагов = 2.0044% при пороге 2.0% — внебраковочно]."""
    import re
    TABLE = {"ln1.weight": "ln1g", "ln1.bias": "ln1b", "ln2.weight": "ln2g", "ln2.bias": "ln2b",
             "th": "th", "sc": "sc", "Wm": "Wm", "fc1": "fc1", "fc2": "fc2"}
    CONST = {"th": 0.0, "sc": 1.0, "ln1g": 1.0, "ln1b": 0.0, "ln2g": 1.0, "ln2b": 0.0,
             "lnfg": 1.0, "lnfb": 0.0}

    def mapped(tname):
        m = re.fullmatch(r"blocks\.(\d+)\.(.+)", tname)
        if m:
            return f"b{m.group(1)}." + TABLE[m.group(2)]
        return {"lnf.weight": "lnfg", "lnf.bias": "lnfb"}.get(tname, tname)

    with torch.no_grad():
        for tname, p in model.named_parameters():
            nname = mapped(tname)
            if nname in CONST:
                p.fill_(CONST[nname])
                continue
            rng = hnp.random.default_rng(zlib.crc32(nname.encode()) & 0xFFFFFFFF)
            arr = rng.normal(0.0, 0.02, tuple(p.shape)).astype(hnp.float32)
            p.copy_(torch.from_numpy(arr).to(p.dtype))


# ---------------------------------------------------------------- Muon: NS5 fp32-для-sm75/sm_60
def _ns_dtype():
    if torch.cuda.is_available():
        try:
            if torch.cuda.get_device_capability(0) < (8, 0):
                return torch.float32      # T4/P100 аппаратно bf16 не умеют
        except Exception:
            pass
    return torch.bfloat16


def ns5(G, steps=5):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(_ns_dtype())
    X = X / (X.norm() + 1e-7)
    tr = G.size(0) > G.size(1)
    if tr:
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if tr:
        X = X.mT
    return X.to(G.dtype)


# ---------------------------------------------------------------- EMA-миксер (замкнутая Σ-форма)
def ema_mix(X, th, sc):
    B, T, D = X.shape
    a = torch.sigmoid(th)
    tt = torch.arange(T, device=X.device)
    dd = (tt[:, None] - tt[None, :]).clamp(min=0).to(f32)
    alog = torch.log(a.clamp_min(1e-20))
    P = torch.exp(dd[:, :, None] * alog[None, None, :])
    mask = (tt[:, None] >= tt[None, :]).to(f32)
    M = P * mask[:, :, None] * (1 - a)[None, None, :]
    H = torch.einsum('tkd,bkd->btd', M, X)
    return H * sc


class Block(nn.Module):
    def __init__(self, D, ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(D); self.ln2 = nn.LayerNorm(D)
        self.th = nn.Parameter(torch.zeros(D)); self.sc = nn.Parameter(torch.ones(D))
        self.Wm = nn.Parameter(torch.empty(D, D)); nn.init.normal_(self.Wm, std=0.02)
        self.fc1 = nn.Parameter(torch.empty(D, ff)); nn.init.normal_(self.fc1, std=0.02)
        self.fc2 = nn.Parameter(torch.empty(ff, D)); nn.init.normal_(self.fc2, std=0.02)

    def forward(self, x):
        mix = ema_mix(self.ln1(x), self.th, self.sc) @ self.Wm
        z = self.ln2(x + mix)
        o2 = F.gelu(z @ self.fc1, approximate='tanh') @ self.fc2
        return x + mix + o2


class LeanCore(nn.Module):
    def __init__(self, V, D=192, L=4, ff=576):
        super().__init__()
        self.V, self.D = V, D
        self.E = nn.Parameter(torch.empty(V, D)); nn.init.normal_(self.E, std=0.02)
        self.pos = nn.Parameter(torch.empty(96, D)); nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(D, ff) for _ in range(L)])
        self.lnf = nn.LayerNorm(D)

    def forward(self, ids):
        h = self.E[ids] + self.pos[:ids.shape[1]][None]
        for b in self.blocks:
            h = b(h)
        return self.lnf(h)

    def logits(self, h):
        return h @ self.E.t()


MUON_SKIP = ("E", "U", "pos")


def muon_split(model):
    mu, ad = [], []
    for n, p in model.named_parameters():
        short = n.split(".")[-1]
        if p.ndim == 2 and min(p.shape) >= 16 and short not in MUON_SKIP:
            mu.append((n, p))
        else:
            ad.append((n, p))
    return mu, ad


def trunk_ratio(model, n0):
    tot = 0.0
    for n, p in model.named_parameters():
        short = n.split(".")[-1]
        if p.ndim == 2 and min(p.shape) >= 16 and short not in MUON_SKIP:
            v = p.detach().double().cpu()
            tot += float((v * v).sum())
    return tot ** 0.5 / n0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--mulr", type=float, default=0.02)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--ssk", type=int, default=0)
    ap.add_argument("--ssfull", type=float, default=0.0)
    ap.add_argument("--ssalpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--negrng", type=int, default=0)
    ap.add_argument("--trunknorm", type=int, default=0)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--ctx", type=int, default=96)
    ap.add_argument("--data", default="data/prep")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--saveckpt", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(INIT_SEED)                     # фиксированный инит — НЕ args.seed
    root = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(f"{root}/{args.outdir}", exist_ok=True)

    tr = hnp.load(f"{root}/{args.data}/train.npy").astype(hnp.int64)
    va = hnp.load(f"{root}/{args.data}/val.npy").astype(hnp.int64)
    V = json.load(open(f"{root}/{args.data}/meta.json"))["vocab"]
    rng = hnp.random.default_rng(args.seed)
    rng_neg = hnp.random.default_rng(args.seed * 1000003 + 17) if args.negrng else rng

    model = LeanCore(V).to(dev)
    import zlib as _zlib
    np_paired_init(model, hnp, _zlib)   # инит битово = numpy-тренер (иначе гейт меряет шум инита)
    nparams = sum(p.numel() for p in model.parameters())
    mu, ad = muon_split(model)
    mbuf = {n: torch.zeros_like(p) for n, p in mu}
    am = {n: torch.zeros_like(p) for n, p in ad}
    av = {n: torch.zeros_like(p) for n, p in ad}
    at = 0
    print(f"[{args.tag}] torch nano-ema params={nparams:,} dev={dev} muon={len(mu)} adam={len(ad)}", flush=True)

    ssq = None
    if args.ssk > 0:
        cnt = hnp.bincount(tr, minlength=V).astype(hnp.float64) + 1.0
        ssq = cnt ** args.ssalpha
        ssq = ssq / ssq.sum()

    def batch(ids, rr):
        st = rr.integers(0, len(ids) - args.ctx - 1, size=args.batch)
        x = hnp.stack([ids[s:s + args.ctx] for s in st])
        y = hnp.stack([ids[s + 1:s + args.ctx + 1] for s in st])
        return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)

    @torch.no_grad()
    def vloss(iters=6):
        model.eval()
        rr = hnp.random.default_rng(1234); tot = 0.0
        for _ in range(iters):
            x, y = batch(va, rr)
            lg = model.logits(model(x))
            tot += F.cross_entropy(lg.reshape(-1, V), y.reshape(-1)).item()
        model.train()
        return tot / iters

    def trunk_n0():
        tot = 0.0
        for n, p in mu:
            v = p.detach().double().cpu()
            tot += float((v * v).sum())
        return tot ** 0.5

    N0 = trunk_n0() if args.trunknorm else None
    logf = open(f"{root}/{args.outdir}/run_{args.tag}.jsonl", "w")
    t0 = time.time(); toks = 0; rec = {}
    for step in range(args.steps):
        lr = args.lr * min((step + 1) / args.warmup,
              0.1 + 0.45 * (1 + math.cos(math.pi * max(0, step - args.warmup) / max(1, args.steps - args.warmup))))
        x, y = batch(tr, rng)
        h = model(x)
        use_ss = ssq is not None and step < args.steps * (1.0 - args.ssfull)
        if use_ss:
            neg = rng_neg.choice(V, size=args.ssk, replace=True, p=ssq)
            yh = y.cpu().numpy()
            cand = hnp.union1d(hnp.unique(yh), neg).astype(hnp.int64)
            logQ = hnp.log1p(-hnp.power(1.0 - ssq[cand], args.ssk))
            logQ[hnp.isin(cand, yh)] = 0.0
            lbl = hnp.searchsorted(cand, yh)
            cand_t = torch.from_numpy(cand).to(dev)
            logQ_t = torch.from_numpy(logQ).to(dev).to(f32)
            lbl_t = torch.from_numpy(lbl).to(dev)
            lg = h @ model.E[cand_t].t() - logQ_t[None, None, :]
            loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), lbl_t.reshape(-1))
        else:
            lg = model.logits(h)
            loss = F.cross_entropy(lg.reshape(-1, V), y.reshape(-1))
        model.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            at += 1
            for n, p in mu:
                g = p.grad
                mbuf[n].lerp_(g, 1 - 0.95)
                u = g.lerp(mbuf[n], 0.95)          # nesterov, как nano_lc
                O = ns5(u)
                scale = max(1.0, O.shape[0] / O.shape[1]) ** 0.5
                p.mul_(1 - args.wd * lr)
                p.add_(O, alpha=-args.mulr * scale)
            for n, p in ad:
                g = p.grad
                am[n].lerp_(g, 0.1)
                av[n].mul_(0.95).addcmul_(g, g, value=0.05)
                mh = am[n] / (1 - 0.9 ** at); vh = av[n] / (1 - 0.95 ** at)
                p.mul_(1 - args.wd * lr)
                p.addcdiv_(mh, vh.sqrt().add_(1e-8), value=-lr)
        toks += x.numel()
        if step % args.eval_every == 0 or step == args.steps - 1:
            vl = vloss(); el = time.time() - t0
            rec = dict(step=step, train_loss=round(float(loss.item()), 4), val_loss=round(vl, 4),
                       val_ppl=round(float(math.exp(vl)), 2), wall=round(el, 1),
                       tok_s=round(toks / el, 1), tokens=toks)
            if N0 is not None:
                rec["wn"] = round(trunk_ratio(model, N0), 3)
            print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
    if args.saveckpt:
        m = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
        hnp.savez(f"{root}/{args.outdir}/ckpt_{args.tag}.npz", **m)
    print("SUMMARY " + json.dumps(dict(tag=args.tag, kind="ema", params=nparams,
        tokens=toks, wall_s=round(time.time() - t0, 1), tok_s=round(toks / (time.time() - t0), 1),
        final_val_loss=rec.get("val_loss"), final_val_ppl=rec.get("val_ppl"))), flush=True)


if __name__ == "__main__":
    main()
