"""CDEMO stand R4: register iteration sigma^h(x) — ground-truth sequential depth.

Protocol: docs/CDEMO_PROTOCOL.md + ADDENDA 1/2/3. Primitive sigma = one FIXED
permutation of 32 values (bigram, trivially learnable); the TASK = compose it
h times. Anti-memorization: 20% of (x,h) cells held out of training, val_core
consists ONLY of held-out cells; extension eval h in 13..20.

Format: seq = [x, BLK x R]; slot i target = sigma^i(x) for i<=h else masked.

Arms: L (D-002 LN kernel, lognorm K-diet) / Lsum (sum kernel beta=.25, no LN) /
D8 (dense 8 blocks) / D2 (dense 2 blocks).
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.models import TinyLoopLM, count_params  # noqa: E402
from lab.telemetry import lambda_from_displacement  # noqa: E402

N = 32                 # value domain size
BLK = 33
VOCAB = 34
R = 20                 # readout slots (set_task overrides per armset)
HEAD = 1               # prefix: [x]
T = HEAD + R
SIGMA = np.random.default_rng(999).permutation(N)          # fixed bigram (add.3)
_SIT = np.zeros(N + 1, dtype=np.int64)
_SIT[:N] = SIGMA[: N]
_HOLDCACHE = {}


def set_task(R_new):
    """Reconfigure slot count (called by armset before building sources)."""
    global R, T
    R, T = R_new, HEAD + R_new


def hold_cells(h_hi, frac=0.2, seed=4242):
    """20% holdout of (x,h) cells for h in 1..h_hi (cached)."""
    key = (h_hi, frac, seed)
    if key not in _HOLDCACHE:
        cells = [(x, h) for x in range(N) for h in range(1, h_hi + 1)]
        idx = set(np.random.default_rng(seed).choice(
            len(cells), size=int(len(cells) * frac), replace=False).tolist())
        _HOLDCACHE[key] = {cells[i] for i in idx}
    return _HOLDCACHE[key]
D, DFF, NH = 96, 384, 4
MAX_LEN = 96
VAL_SEED = 1234
IGN = -100


class ChainSource:
    """(x,h) cells. partition: 'train' = 80% of in-range cells (holdout excluded),
    'holdout' = the held-out 20% (only for h<=12), 'all' = no partition."""

    def __init__(self, seed=0, h_lo=1, h_hi=12, partition="train", hold_hi=12):
        self.rng = np.random.default_rng(seed)
        cells = [(x, h) for h in range(h_lo, h_hi + 1) for x in range(N)]
        if partition != "all":
            hold = hold_cells(hold_hi)
            cells = [c for c in cells
                     if (c in hold) == (partition == "holdout")]
        self.cells = cells
        assert cells, f"empty cell list for {partition} h in {h_lo}..{h_hi}"

    def sample(self, B):
        idx = self.rng.integers(0, len(self.cells), B)
        xs = np.array([self.cells[i][0] for i in idx])
        hs = np.array([self.cells[i][1] for i in idx])
        xb = np.full((B, T), BLK, dtype=np.int64)
        xb[:, 0] = xs
        yt = np.full((B, R), IGN, dtype=np.int64)
        cur = SIGMA[xs]
        for i in range(R):
            write = (i + 1) <= hs
            yt[write, i] = cur[write]
            cur = SIGMA[cur]
        return (torch.from_numpy(xb), torch.from_numpy(yt),
                torch.from_numpy(hs.astype(np.int64)))


class ValSet:
    def __init__(self, n, seed, h_lo, h_hi, partition="all", hold_hi=12):
        src = ChainSource(seed=seed, h_lo=h_lo, h_hi=h_hi, partition=partition,
                          hold_hi=hold_hi)
        reps = int(np.ceil(n / len(src.cells)))
        xs, ys, hs_ = [], [], []
        for _ in range(reps):
            a, b, c = src.sample(len(src.cells))
            xs.append(a); ys.append(b); hs_.append(c)
        self.x = torch.cat(xs)[:n]
        self.yt = torch.cat(ys)[:n]
        self.h = torch.cat(hs_)[:n]


class DenseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(nn.Linear(D, DFF), nn.GELU(), nn.Linear(DFF, D))

    def forward(self, h, block_mask=None):
        B, T2, Dh = h.shape
        u = self.ln1(h)
        q, k, v = self.qkv(u).chunk(3, dim=-1)
        hd = Dh // NH
        q = q.view(B, T2, NH, hd).transpose(1, 2)
        k = k.view(B, T2, NH, hd).transpose(1, 2)
        v = v.view(B, T2, NH, hd).transpose(1, 2)
        s = q @ k.transpose(-2, -1) / math.sqrt(hd)
        mask = torch.triu(torch.ones(T2, T2, dtype=torch.bool, device=h.device), 1)
        if block_mask is not None:
            mask = mask | block_mask
        s = s.masked_fill(mask, float("-inf"))
        a = torch.softmax(s, dim=-1) @ v
        h = h + self.proj(a.transpose(1, 2).reshape(B, T2, Dh))
        h = h + self.mlp(self.ln2(h))
        return h


class DenseLM(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D)
        self.pos = nn.Embedding(MAX_LEN, D)
        self.blocks = nn.ModuleList(DenseBlock() for _ in range(layers))
        self.ln_out = nn.LayerNorm(D)
        for emb in (self.tok, self.pos):
            nn.init.normal_(emb.weight, std=0.02)

    def forward(self, x, K=None, block_mask=None):
        h = self.tok(x) + self.pos.weight[:x.shape[1]].unsqueeze(0)
        for b in self.blocks:
            h = b(h, block_mask=block_mask)
        return F.linear(self.ln_out(h), self.tok.weight)


def make_ab_mask():
    """Block slot->slot attention (queries and keys with index >= HEAD)."""
    m = torch.zeros(T, T, dtype=torch.bool)
    m[HEAD:, HEAD:] = True
    return m


def build_loop(state_ln=True, beta=1.0):
    m = TinyLoopLM(vocab=VOCAB, d=D, n_head=NH, d_ff=DFF, max_len=MAX_LEN,
                   max_loops=64, beta=beta, state_ln=state_ln)
    for emb in (m.tok, m.pos):
        nn.init.normal_(emb.weight, std=0.02)
    return m


_AB = {"mask": None}

def _abm(ab):
    if ab and _AB["mask"] is None:
        _AB["mask"] = make_ab_mask()
    return _AB["mask"] if ab else None


def slot_logits(model, kind, x, K, ab=False):
    if kind == "dense":
        return model(x, block_mask=_abm(ab))[:, HEAD:]     # [B,R,V]
    out = model(x, K, block_mask=_abm(ab))
    return out["logits"][:, :, HEAD:]                  # [K,B,R,V]


def acc_on(model, vset, kind, K=16, batch=128, ab=False):
    """acc reading slot h-1 (per-sample), plus per-h buckets."""
    model.eval()
    per_h = {}
    with torch.no_grad():
        for i in range(0, vset.x.shape[0], batch):
            x = vset.x[i:i + batch]
            h = vset.h[i:i + batch]
            lg = slot_logits(model, kind, x, K, ab=ab)
            if kind != "dense":
                lg = lg[-1]
            pred = lg.argmax(-1)                       # [B,R]
            ans = pred.gather(1, (h - 1).unsqueeze(1)).squeeze(1)
            y = vset.yt[i:i + batch].gather(1, (h - 1).unsqueeze(1)).squeeze(1)
            ok = (ans == y)
            for hh, o in zip(h.tolist(), ok.tolist()):
                s = per_h.setdefault(hh, [0, 0])
                s[0] += int(o)
                s[1] += 1
    tot = [sum(v[j] for v in per_h.values()) for j in range(2)]
    acc_h = {k: v[0] / max(v[1], 1) for k, v in sorted(per_h.items())}
    return tot[0] / tot[1], acc_h


def acc_oracle_on(model, vset, kind, K=32, batch=128, ab=False):
    """Existence read: fraction of samples where ANY slot argmax == sigma^h(x)."""
    model.eval()
    oks = []
    with torch.no_grad():
        for i in range(0, vset.x.shape[0], batch):
            x = vset.x[i:i + batch]
            h = vset.h[i:i + batch]
            lg = slot_logits(model, kind, x, K, ab=ab)
            if kind != "dense":
                lg = lg[-1]
            pred = lg.argmax(-1)                       # [B,R]
            y = vset.yt[i:i + batch].gather(1, (h - 1).unsqueeze(1)).squeeze(1)
            oks += (pred == y.unsqueeze(1)).any(-1).tolist()
    return float(np.mean(oks))


def _spearman(a, b):
    def rank(v):
        order = np.argsort(np.asarray(v, dtype=np.float64))
        r = np.empty(len(v)); r[order] = np.arange(len(v))
        # average ties
        vv = np.asarray(v, dtype=np.float64)
        for u in np.unique(vv):
            idx = vv == u
            r[idx] = r[idx].mean()
        return r
    ra, rb = rank(a), rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def stop_rule_stats(model, vset, Kmax=16, eps=0.05, batch=128, ab=False):
    model.eval()
    Ks, oks, hs_all = [], [], []
    with torch.no_grad():
        for i in range(0, vset.x.shape[0], batch):
            x = vset.x[i:i + batch]
            h = vset.h[i:i + batch]
            out = model(x, Kmax, want_hiddens=True, block_mask=_abm(ab))
            H = out["hiddens"]                        # [K+1,B,T,D]
            dkb = (H[1:] - H[:-1]).pow(2).sum(-1).sqrt().mean(-1)   # [K,B]
            d = dkb.transpose(0, 1)                   # [B,K]
            thr = eps * d[:, :1].clamp_min(1e-9)
            kstop = torch.full((x.shape[0],), Kmax, dtype=torch.long)
            for k in range(Kmax):
                hit = d[:, k] < thr[:, 0]
                kstop = torch.where((kstop == Kmax) & hit,
                                    torch.full_like(kstop, k + 1), kstop)
            lg = out["logits"][:, :, HEAD:]           # [K,B,R,V]
            pred = lg.argmax(-1)                       # [K,B,R]
            idx_b = torch.arange(x.shape[0])
            ans = pred[kstop - 1, idx_b].gather(1, (h - 1).unsqueeze(1)).squeeze(1)
            y = vset.yt[i:i + batch].gather(1, (h - 1).unsqueeze(1)).squeeze(1)
            Ks += kstop.tolist()
            oks += (ans == y).tolist()
            hs_all += h.tolist()
    return {"K_mean": float(np.mean(Ks)), "acc_at_stop": float(np.mean(oks)),
            "spearman_K_h": _spearman(Ks, hs_all),
            "K_by_h": {str(hh): round(float(np.mean(
                [k for k, hv in zip(Ks, hs_all) if hv == hh])), 2)
                for hh in sorted(set(hs_all))}}


def lambda_report(model, vset, K=32, ab=False):
    model.eval()
    with torch.no_grad():
        out = model(vset.x[:128], K, want_hiddens=True, block_mask=_abm(ab))
        H = out["hiddens"]
        d = (H[1:] - H[:-1]).pow(2).mean(dim=(1, 2, 3)).sqrt()
        hmax = float(H.norm(dim=-1).max())
    fit = lambda_from_displacement(d, floor_ratio=1e-4)
    return {"lambda": fit["lambda"], "regime": fit["regime"], "r2": fit["r2"],
            "window": fit["window"], "max_h_norm": hmax}


def train_arm(name, kind, model, steps, B=64, lr=2e-3, seed=0, warmup=100,
              eval_every=250, thr=0.80, ab=False, diet=("lognorm", 8.0, 0.8, 4, 48),
              fixed_k=None, probe_hi=8, train_hi=48):
    torch.manual_seed(seed)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    src = ChainSource(seed=seed + 1, partition="train", h_hi=train_hi,
                      hold_hi=train_hi)
    g = torch.Generator().manual_seed(seed + 2)
    probe = ValSet(256, seed=VAL_SEED, h_lo=1, h_hi=probe_hi, partition="holdout",
                   hold_hi=train_hi)
    flops_blocks = 0
    t0 = time.time()
    first_hit = None
    hist = []
    for step in range(1, steps + 1):
        x, yt, h = src.sample(B)
        if kind == "dense":
            le = slot_logits(model, kind, x, 0, ab=ab)
            loss = F.cross_entropy(le.reshape(-1, VOCAB).float(),
                                   yt.reshape(-1), ignore_index=IGN)
            flops_blocks += len(model.blocks) * B
        else:
            if fixed_k is not None:
                K = fixed_k
            else:
                _, mu, sg, klo, khi = diet
                u = torch.randn((), generator=g) * sg + math.log(mu)
                K = int(min(khi, max(klo, round(float(u.exp())))))
            le = slot_logits(model, kind, x, K, ab=ab)  # [K,B,R,V]
            loss = torch.stack([F.cross_entropy(
                le[k].reshape(-1, VOCAB).float(), yt.reshape(-1),
                ignore_index=IGN) for k in range(K)]).mean()
            flops_blocks += K * B
        for pg in opt.param_groups:
            pg["lr"] = lr * min(1.0, step / warmup) * \
                0.5 * (1 + math.cos(math.pi * step / steps))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % eval_every == 0 or step == 1:
            acc_probe, _ = acc_on(model, probe, kind, K=16, ab=ab)
            el = time.time() - t0
            hist.append({"step": step, "loss": float(loss), "probe_acc": acc_probe,
                         "flops_blocks": flops_blocks, "sec": el})
            if first_hit is None and acc_probe >= thr:
                first_hit = {"step": step, "flops_blocks": flops_blocks, "sec": el}
            print(f"  [{name}] step {step}/{steps} loss {float(loss):.4f} "
                  f"probe {acc_probe:.3f} ({el/step:.3f}s/step)", flush=True)
    return {"hist": hist, "first_hit": first_hit,
            "flops_blocks_total": flops_blocks, "sec_total": time.time() - t0}


class StopCritic(nn.Module):
    """Learned halt critic (A1_PLAN §4 toy prototype): features of loop state k
    -> P(answer already correct at k). Trained on a FROZEN loop model."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 64), nn.GELU(),
                                 nn.Linear(64, 64), nn.GELU(),
                                 nn.Linear(64, 1))

    def forward(self, f):
        return self.net(f).squeeze(-1)


def collect_critic_data(model, src, n_samples, Kmax=48, batch=128, ab=False):
    """Features per (sample,k): d_k, cos_k, k/K, d1-log, logit-margin; label correct_k."""
    model.eval()
    F_all, Y_all, H_all = [], [], []
    with torch.no_grad():
        seen = 0
        while seen < n_samples:
            x, yt, h = src.sample(batch)
            out = model(x, Kmax, want_hiddens=True, block_mask=_abm(ab))
            H = out["hiddens"]                        # [K+1,B,T,D]
            lg = out["logits"][:, :, HEAD:]           # [K,B,R,V]
            pred = lg.argmax(-1)                        # [K,B,R]
            idx = torch.arange(x.shape[0])
            y = yt.gather(1, (h - 1).unsqueeze(1)).squeeze(1)
            corr = (pred[:, idx, h - 1] == y.unsqueeze(0)).float()   # [K,B]
            dh = (H[1:] - H[:-1]).pow(2).sum(-1).sqrt().mean(-1)   # [K,B]
            cs = torch.cosine_similarity(H[1:], H[:-1], dim=-1).mean(-1)
            _t2 = lg.topk(2, dim=-1).values
            marg = (_t2[..., 0] - _t2[..., 1])[:, idx, h - 1]        # [K,B]
            kk = torch.arange(1, Kmax + 1, dtype=torch.float32).unsqueeze(1) / Kmax
            feats = torch.stack([dh, cs, kk.expand_as(dh),
                                 torch.log(dh[:, :1].clamp_min(1e-9)).expand_as(dh),
                                 marg / 5.0], dim=-1)              # [K,B,5]
            F_all.append(feats.transpose(0, 1).reshape(-1, 5))
            Y_all.append(corr.transpose(0, 1).reshape(-1))
            H_all.append(h.repeat(Kmax))
            seen += x.shape[0]
    return torch.cat(F_all), torch.cat(Y_all), torch.cat(H_all)


def train_critic(model, src, Kmax=48, steps=1500, B=256, ab=False, seed=0):
    torch.manual_seed(seed)
    Ftr, Ytr, _ = collect_critic_data(model, src, 8192, Kmax=Kmax, ab=ab)
    crit = StopCritic()
    opt = torch.optim.AdamW(crit.parameters(), lr=3e-3)
    n = Ftr.shape[0]
    pos = Ytr.mean()
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (B,))
        p = crit(Ftr[idx])
        w = torch.where(Ytr[idx] > 0.5, 1.0 / max(pos, 1e-3), 1.0 / max(1 - pos, 1e-3))
        loss = (F.binary_cross_entropy_with_logits(p, Ytr[idx], reduction="none") * w).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return crit


def policy_eval(model, vset, critic=None, Kmax=48, eps=0.05, batch=128, ab=False):
    """Policies on a (possibly skewed) val set: fixed Kmax / sigma-rule / critic / oracle."""
    model.eval()
    res = {k: {"Ks": [], "oks": []} for k in ("fixed", "sigma", "critic", "oracle")}
    with torch.no_grad():
        for i in range(0, vset.x.shape[0], batch):
            x = vset.x[i:i + batch]
            h = vset.h[i:i + batch]
            out = model(x, Kmax, want_hiddens=True, block_mask=_abm(ab))
            H = out["hiddens"]
            lg = out["logits"][:, :, HEAD:]
            pred = lg.argmax(-1)                       # [K,B,R]
            idx = torch.arange(x.shape[0])
            y = vset.yt[i:i + batch].gather(1, (h - 1).unsqueeze(1)).squeeze(1)
            corr = (pred[:, idx, h - 1] == y.unsqueeze(0))            # [K,B]
            Bsz = x.shape[0]
            kb = torch.arange(1, Kmax + 1)
            solvable_first = corr.float().argmax(0) + 1          # [B] first-correct (1-based)
            never = ~corr.any(0)
            # fixed
            res["fixed"]["Ks"] += [Kmax] * Bsz
            res["fixed"]["oks"] += corr[-1].tolist()
            # oracle
            res["oracle"]["Ks"] += torch.where(never, torch.tensor(Kmax),
                                               solvable_first).tolist()
            res["oracle"]["oks"] += corr[solvable_first - 1, idx].tolist()
            # sigma
            dkb = (H[1:] - H[:-1]).pow(2).sum(-1).sqrt().mean(-1).transpose(0, 1)
            thr = eps * dkb[:, :1].clamp_min(1e-9)
            ks_sig = torch.full((Bsz,), Kmax, dtype=torch.long)
            for k in range(Kmax):
                hit = dkb[:, k] < thr[:, 0]
                ks_sig = torch.where((ks_sig == Kmax) & hit,
                                     torch.full_like(ks_sig, k + 1), ks_sig)
            res["sigma"]["Ks"] += ks_sig.tolist()
            res["sigma"]["oks"] += corr[ks_sig - 1, idx].tolist()
            # critic
            if critic is not None:
                dh = (H[1:] - H[:-1]).pow(2).sum(-1).sqrt().mean(-1)
                cs = torch.cosine_similarity(H[1:], H[:-1], dim=-1).mean(-1)
                top2 = lg.topk(2, dim=-1).values
                marg = (top2[..., 0] - top2[..., 1])[:, idx, h - 1]   # [K,B]
                kk = kb.unsqueeze(1).float().expand(Kmax, Bsz) / Kmax
                feats = torch.stack([dh, cs, kk,
                                     torch.log(dh[:, :1].clamp_min(1e-9)).expand_as(dh),
                                     marg / 5.0], dim=-1)          # [K,B,5]
                with torch.no_grad():
                    p = torch.sigmoid(critic(feats.transpose(0, 1)))   # [B,K]
                ks_cr = torch.full((Bsz,), Kmax, dtype=torch.long)
                for k in range(Kmax):
                    hit = p[:, k] > 0.5
                    ks_cr = torch.where((ks_cr == Kmax) & hit,
                                        torch.full_like(ks_cr, k + 1), ks_cr)
                res["critic"]["Ks"] += ks_cr.tolist()
                res["critic"]["oks"] += corr[ks_cr - 1, idx].tolist()
    out = {}
    for k, v in res.items():
        if not v["Ks"]:
            continue
        out[k] = {"K_mean": float(np.mean(v["Ks"])), "acc": float(np.mean(v["oks"]))}
    for k in out:
        out[k]["saving_vs_fixed"] = Kmax / out[k]["K_mean"]
    out["drop_vs_fixed"] = {k: out["fixed"]["acc"] - out[k]["acc"] for k in out}
    return out


class SkewVal:
    """Skewed-difficulty val: weights over h-ranges, x uniform."""

    def __init__(self, n, seed, spec=((1, 2, 0.95), (4, 8, 0.045), (17, 48, 0.005))):
        rng = np.random.default_rng(seed)
        ranges = np.array([r[2] for r in spec])
        ranges /= ranges.sum()
        pick = rng.choice(len(spec), size=n, p=ranges)
        hs = np.array([rng.integers(spec[i][0], spec[i][1] + 1) for i in pick])
        xs = rng.integers(0, N, n)
        xb = np.full((n, T), BLK, dtype=np.int64)
        xb[:, 0] = xs
        yt = np.full((n, R), IGN, dtype=np.int64)
        cur = SIGMA[xs]
        for i in range(R):
            write = (i + 1) <= hs
            yt[write, i] = cur[write]
            cur = SIGMA[cur]
        self.x = torch.from_numpy(xb)
        self.yt = torch.from_numpy(yt)
        self.h = torch.from_numpy(hs.astype(np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arms", default="L,Lsum,D8,D2")
    ap.add_argument("--armset", default="r4", choices=["r4", "r5", "r6"])
    ap.add_argument("--out", default="results/cdemo/chain_r2.json")
    a = ap.parse_args()
    steps = 400 if a.smoke else a.steps

    val12 = ValSet(512, seed=VAL_SEED + 1, h_lo=1, h_hi=12, partition="holdout")
    val_ext = ValSet(384, seed=VAL_SEED + 2, h_lo=13, h_hi=20, partition="all")

    if a.armset == "r5":
        a.arms = "L_ab,Lsum_ab,D8_ab,D2_ab"
    if a.armset == "r6":
        set_task(48)
        a.arms = "L48d,L48f"
    builders = {
        "L": ("loop", build_loop(state_ln=True, beta=1.0), False),
        "Lsum": ("loop", build_loop(state_ln=False, beta=0.25), False),
        "D8": ("dense", DenseLM(8), False),
        "D2": ("dense", DenseLM(2), False),
        "L_ab": ("loop", build_loop(state_ln=True, beta=1.0), True),
        "Lsum_ab": ("loop", build_loop(state_ln=False, beta=0.25), True),
        "D8_ab": ("dense", DenseLM(8), True),
        "D2_ab": ("dense", DenseLM(2), True),
        "L48d": ("loop", build_loop(state_ln=True, beta=1.0), False),
        "L48f": ("loop", build_loop(state_ln=True, beta=1.0), False),
    }
    if a.armset == "r6":
        return main_r6(a, builders)
    res = {"config": {"N": N, "R": R, "D": D, "DFF": DFF, "steps": steps,
                      "diet": "lognorm(ln4.5,0.9) clip [2,16]", "B": 64, "lr": 2e-3,
                      "format": "R4/R5/R6 register task sigma^h(x) (addenda 1-5)"},
           "arms": {}}
    for name in a.arms.split(","):
        kind, model, ab = builders[name]
        print(f"[arm {name}] params={count_params(model):,} kind={kind} ab={ab}",
              flush=True)
        tr = train_arm(name, kind, model, steps, ab=ab,
                       diet=("lognorm", 4.5, 0.9, 2, 16), train_hi=12)
        rep = {"params": count_params(model), "train": tr, "ab": ab}
        a16, ah16 = acc_on(model, val12, kind, K=16, ab=ab)
        rep["acc12_at_K16"], rep["acc_by_h_K16"] = a16, ah16
        if kind != "dense":
            rep["acc12_at_K32"], rep["acc_by_h_K32"] = acc_on(model, val12, kind,
                                                              K=32, ab=ab)
            rep["acc_ext13_20_at_K32"], rep["acc_ext_by_h"] = \
                acc_on(model, val_ext, kind, K=32, ab=ab)
            rep["acc_ext13_20_oracle_K32"] = acc_oracle_on(model, val_ext, kind,
                                                           K=32, ab=ab)
            rep["lambda_K32"] = lambda_report(model, val12, K=32, ab=ab)
            rep["stop_rule"] = stop_rule_stats(model, val12, Kmax=16, ab=ab)
        else:
            rep["acc_ext13_20"], rep["acc_ext_by_h"] = acc_on(model, val_ext, kind,
                                                              K=0, ab=ab)
            rep["acc_ext13_20_oracle"] = acc_oracle_on(model, val_ext, kind, K=0,
                                                       ab=ab)
        res["arms"][name] = rep
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
        print(f"[arm {name} done] acc12@K16={a16:.3f}", flush=True)
    print("[saved]", a.out)




def main_r6(a, builders):
    """R6 protocol (ADDENDUM_5): diet vs fixed-K control, critic, skew policies."""
    steps_diet = 5000
    val48 = ValSet(512, seed=VAL_SEED + 1, h_lo=1, h_hi=48, partition="holdout",
                   hold_hi=48)
    res = {"config": {"format": "R6 (addendum 5)", "h_max": 48, "R": 48,
                      "diet": "lognorm(ln8,0.8) clip [4,48]", "steps_diet": steps_diet},
           "arms": {}}

    # --- diet arm ---
    kind, model, ab = builders["L48d"]
    print(f"[arm L48d] params={count_params(model):,}", flush=True)
    tr = train_arm("L48d", kind, model, steps_diet, train_hi=48, probe_hi=8)
    diet_blocks = tr["flops_blocks_total"]
    rep = {"params": count_params(model), "train": tr, "diet": True}
    a48, ah48 = acc_on(model, val48, kind, K=48)
    rep["acc48_at_K48"], rep["acc_by_h_K48"] = a48, ah48
    rep["lambda_K48"] = lambda_report(model, val48, K=48)
    res["arms"]["L48d"] = rep
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))

    # --- critic on frozen diet arm ---
    src_crit = ChainSource(seed=777, h_hi=48, partition="train", hold_hi=48)
    print("[critic] collecting + training ...", flush=True)
    critic = train_critic(model, src_crit, Kmax=48, steps=1500)
    skew = SkewVal(4096, seed=VAL_SEED + 5)
    res["critic_policies"] = policy_eval(model, skew, critic=critic, Kmax=48)
    res["critic_policies"]["skew_spec"] = "95% h1-2 / 4.5% h4-8 / 0.5% h17-48"
    res["critic_policies"]["oracle_bound"] = res["critic_policies"]["oracle"]
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("[policies]", json.dumps(res["critic_policies"], indent=1), flush=True)

    # --- fixed-K control arm, iso-blocks ---
    fixed_steps = max(250, round(diet_blocks / (48 * 64)))
    kind2, model2, _ = builders["L48f"]
    print(f"[arm L48f] fixed-K=48 control, {fixed_steps} steps (iso-blocks "
          f"{diet_blocks})", flush=True)
    tr2 = train_arm("L48f", kind2, model2, fixed_steps, fixed_k=48,
                    train_hi=48, probe_hi=8)
    rep2 = {"params": count_params(model2), "train": tr2, "diet": False}
    a48f, ah48f = acc_on(model2, val48, kind2, K=48)
    rep2["acc48_at_K48"], rep2["acc_by_h_K48"] = a48f, ah48f
    res["arms"]["L48f"] = rep2
    res["iso_blocks_comparison"] = {
        "blocks_each": diet_blocks,
        "acc_L48d": a48, "acc_L48f": a48f}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("[saved]", a.out, flush=True)

if __name__ == "__main__":
    main()
