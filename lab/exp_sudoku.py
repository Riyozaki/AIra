"""G0S stand: Sudoku 9x9 with ground-truth depth. Predictions S-1..S-7 (docs/G0S_PROTOCOL).

One training variant = one dict. Per-row telemetry at eval K=64:
acc_blank/exact per k, angular displacement per row, hitting times tau_stab/tau_eps,
per-row lambda slopes, oracle-critic savings, Spearman(tau, d*), bucket E(K).
Writes results/g0s/sudoku_results.json (+ pools cache npz).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import TinyLoopLM, count_params  # noqa: E402
from sudoku import SudokuConfig, SudokuSource, build_pool  # noqa: E402
from telemetry import geometric_floor_fit  # noqa: E402

torch.set_num_threads(2)
D, NH, DFF = 64, 4, 256
CFG = SudokuConfig(box=3)
S = CFG.S  # 81
TRAIN_POOL = "results/g0s/pool_train9.npz"
VAL_POOL = "results/g0s/pool_val9.npz"


def get_pools(n_train=1400, n_val=400):
    def load_or_build(path, n, seed):
        if os.path.exists(path):
            z = np.load(path)
            pool = [{"grid": z["grid"][i], "sol": z["sol"][i],
                     "depth": int(z["depth"][i])} for i in range(len(z["depth"]))]
            return pool
        pool, counts, att = build_pool(CFG, n, seed=seed)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path,
                            grid=np.array([p["grid"] for p in pool]),
                            sol=np.array([p["sol"] for p in pool]),
                            depth=np.array([p["depth"] for p in pool]))
        print(f"[pool] built {len(pool)} attempts={att} "
              f"depths={dict(sorted(counts.items()))}", flush=True)
        return pool
    return (load_or_build(TRAIN_POOL, n_train, 0),
            load_or_build(VAL_POOL, n_val, 10_000))


def build_model(beta=1.0, seed=0):
    torch.manual_seed(seed)
    return TinyLoopLM(vocab=CFG.N + 2, d=D, n_head=NH, d_ff=DFF,
                      max_len=2 * S + 2, max_loops=64, beta=beta,
                      depth_emb=False, state_ln=True)


def train(model, src, steps, K_min, K_max, B=12, lr=3e-3, warmup=100, device="cpu"):
    torch.manual_seed(1)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    g = torch.Generator().manual_seed(2)
    t0 = time.time()
    hist = []
    for step in range(steps):
        x, y, _, _, _ = src.batch_tensors(B, device)
        K = int(torch.randint(K_min, K_max + 1, (1,), generator=g))
        out = model(x, K)
        logits = out["logits"]
        loss = torch.stack([
            F.cross_entropy(logits[k].reshape(-1, logits.shape[-1]).float(),
                            y.reshape(-1)) for k in range(K)]).mean()
        for pg in opt.param_groups:
            pg["lr"] = lr * min(1.0, (step + 1) / warmup) * \
                (0.5 * (1 + np.cos(np.pi * step / max(steps, 1))))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % max(steps // 10, 5) == 0:
            hist.append({"step": step + 1, "loss": float(loss),
                         "sps": (time.time() - t0) / (step + 1)})
            print(f"  step {step+1}/{steps} loss {float(loss):.3f} K={K} "
                  f"({hist[-1]['sps']:.3f}s/step)", flush=True)
    if not hist:
        hist.append({"step": steps, "loss": float(loss), "sps": 0.0})
    return hist


def rankdata(a):
    a = np.asarray(a, dtype=float)
    order = a.argsort()
    ranks = np.empty(len(a))
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


@torch.no_grad()
def eval_rows(model, val_pool, K=64, batch=25, device="cpu"):
    """Per-row telemetry. Returns dict of [K, n] arrays + per-row scalar vectors."""
    model.eval()
    src = SudokuSource(CFG, val_pool, seed=555, shuffle=False)
    acc, exact, dang, preds, depths = [], [], [], [], []
    n = len(val_pool)
    done = 0
    while done < n:
        B = min(batch, n - done)
        x, y, blank, dep, items = src.batch_tensors(B, device)
        out = model(x, K)
        logits, hidd = out["logits"], out["hiddens"]  # [K,B,T,V], [K+1,B,T,Dm]
        # answer region in y: y indices S..2S-1 (0-based), cell i -> y idx S+i
        ans = torch.arange(S, 2 * S, device=device)
        sol = y[:, ans]                        # [B, S]
        pr = logits[:, :, ans].argmax(-1)      # [K, B, S]
        hit = (pr == sol.unsqueeze(0))         # [K, B, S]
        bl = blank.unsqueeze(0)                # [1, B, S]
        accb = (hit & bl).sum(-1).float() / bl.sum(-1).float()
        exr = ((hit | ~bl).all(-1))            # [K, B]
        u = F.normalize(hidd, dim=-1)
        du = (u[1:, :, ans] - u[:-1, :, ans]).pow(2).sum(-1).sqrt() / math.sqrt(2.0)
        acc.append(accb.cpu())
        exact.append(exr.float().cpu())
        dang.append(du.mean(-1).cpu())         # [K, B]
        preds.append(pr[:, :, :].cpu())
        depths.append(dep)
        done += B
    acc = torch.cat(acc, 1).numpy()            # [K, n]
    exact = torch.cat(exact, 1).numpy()
    dang = torch.cat(dang, 1).numpy()          # [K, n]
    pr = torch.cat(preds, 1).numpy()           # [K, n, S]
    dep = torch.cat(depths).numpy()
    blank_counts = np.array([(it["grid"] == 0).sum() for it in val_pool])

    Kmax = acc.shape[0]
    # per-row scalar summaries
    tau_stab = np.full(acc.shape[1], Kmax)
    for r in range(acc.shape[1]):
        target = pr[Kmax - 1, r]
        for k in range(Kmax):
            if (pr[k:, r] == target[None, :]).all():
                tau_stab[r] = k + 1  # 1-based loops count
                break
    tau_eps = np.full(acc.shape[1], Kmax, dtype=float)
    for r in range(dang.shape[1]):
        d = dang[:, r]
        for k in range(len(d) - 2):
            if (d[k:k + 3] < 0.02).all():
                tau_eps[r] = k + 1
                break
    lam_row = np.full(acc.shape[1], np.nan)
    win_lo, win_hi = 2, 12
    ks = np.arange(win_lo, win_hi)
    for r in range(dang.shape[1]):
        d = dang[win_lo:win_hi, r].astype(float)
        if (d > 0).all() and len(d) >= 8:
            A = np.vstack([ks, np.ones_like(ks)]).T
            coef, *_ = np.linalg.lstsq(A, np.log(d + 1e-9), rcond=None)
            lam_row[r] = math.exp(coef[0])
    return {"acc": acc, "exact": exact, "dang": dang, "tau_stab": tau_stab,
            "tau_eps": tau_eps, "lam_row": lam_row, "depth": dep,
            "blank_counts": blank_counts}


def analyze(ev, name, K=64):
    dep, acc = ev["depth"], ev["acc"]
    shallow = dep <= 5
    deep = dep >= 7
    out = {"name": name, "n": int(len(dep)),
           "depth_hist": {int(d): int((dep == d).sum()) for d in sorted(set(dep))},
           "K": K}
    for grp, mask in [("all", np.ones(len(dep), bool)),
                      ("shallow", shallow), ("deep", deep)]:
        if mask.sum() < 5:
            continue
        L = 1.0 - acc[:, mask].mean(1)          # error curve
        out[f"Lblank_{grp}"] = L.tolist()
        out[f"acc1_{grp}"] = float(acc[0, mask].mean())
        out[f"acc16_{grp}"] = float(acc[15, mask].mean())
        out[f"acc64_{grp}"] = float(acc[63, mask].mean())
        ks = np.arange(1, K + 1, dtype=float)
        fit = geometric_floor_fit(ks, L)
        out[f"Ek_{grp}"] = fit
    out["gap_deep_1_16"] = out.get("acc16_deep", float("nan")) - out.get("acc1_deep", float("nan"))
    out["gap_all_1_16"] = out.get("acc16_all", 0) - out.get("acc1_all", 0)
    out["deg_16_64_deep"] = out.get("acc16_deep", float("nan")) - out.get("acc64_deep", float("nan"))
    rho_stab = spearman(ev["tau_stab"], dep)
    rho_eps = spearman(ev["tau_eps"], dep)
    out["spearman_taustab_depth"] = rho_stab
    out["spearman_taueps_depth"] = rho_eps
    out["tau_stab_med_shallow"] = float(np.median(ev["tau_stab"][shallow])) if shallow.sum() else None
    out["tau_stab_med_deep"] = float(np.median(ev["tau_stab"][deep])) if deep.sum() else None
    lams = ev["lam_row"][~np.isnan(ev["lam_row"])]
    out["lam_row_med"] = float(np.median(lams))
    out["lam_row_q25"] = float(np.percentile(lams, 25))
    out["lam_row_q75"] = float(np.percentile(lams, 75))
    out["lam_row_qratio"] = out["lam_row_q75"] / max(out["lam_row_q25"], 1e-9)
    # oracle critic
    Kstar = ev["tau_stab"].astype(float)
    acc_or = np.array([acc[int(k) - 1, r] for r, k in enumerate(Kstar)])
    acc_final = acc[63]
    fixed_acc = acc.mean(1)
    acc_target = acc_or.mean()
    K_fixed = int(np.argmax(fixed_acc >= acc_target - 1e-9)) + 1 \
        if (fixed_acc >= acc_target - 1e-9).any() else 64
    out["oracle"] = {
        "mean_Kstar": float(Kstar.mean()),
        "acc_oracle": float(acc_or.mean()),
        "acc_final64": float(acc_final.mean()),
        "K_fixed_same_acc": K_fixed,
        "saving_vs_fixed": float((K_fixed - Kstar.mean()) / K_fixed),
        "saving_vs_64": float((64 - Kstar.mean()) / 64.0),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3500)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--variants", default="ln_b10,ln_b05")
    ap.add_argument("--out", default="results/g0s/sudoku_results.json")
    args = ap.parse_args()
    steps = 40 if args.smoke else args.steps

    print("[pools]", flush=True)
    train_pool, val_pool = get_pools(n_train=200 if args.smoke else 1400,
                                   n_val=120 if args.smoke else 400)
    vdep = np.array([p["depth"] for p in val_pool])
    print(f"[pools] train={len(train_pool)} val={len(val_pool)} "
          f"val depths: {dict(sorted(zip(*np.unique(vdep, return_counts=True))))}", flush=True)

    results = {"config": {"D": D, "DFF": DFF, "steps": steps,
                          "K_train": [2, 12], "K_eval": 64},
               "val_depth_hist": dict(zip(*[a.tolist() for a in np.unique(vdep, return_counts=True)])),
               "runs": {}}
    variants = {"ln_b10": dict(beta=1.0), "ln_b05": dict(beta=0.5)}
    for name in args.variants.split(","):
        if name not in variants:
            continue
        cfgv = variants[name]
        model = build_model(**cfgv)
        print(f"[train] {name} params={count_params(model)} steps={steps}", flush=True)
        t0 = time.time()
        src = SudokuSource(CFG, train_pool, seed=11)
        hist = train(model, src, steps, 2, 12)
        ev = eval_rows(model, val_pool, K=64)
        rep = analyze(ev, name)
        rep["train_loss_final"] = hist[-1]["loss"]
        rep["train_sec"] = time.time() - t0
        rep["n_params"] = count_params(model)
        rep["cfg"] = cfgv
        results["runs"][name] = rep
        print(f"[done] {name}: acc1/16/64 all={rep['acc1_all']:.3f}/"
              f"{rep['acc16_all']:.3f}/{rep['acc64_all']:.3f} "
              f"deep gap={rep['gap_deep_1_16']:.3f} rho_tau={rep['spearman_taustab_depth']:.2f} "
              f"lam_row={rep['lam_row_med']:.3f} save={rep['oracle']['saving_vs_fixed']:.2f}",
              flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
