"""T-week run A: lambda telemetry (TW-0, TW-1, TW-1b, TW-1c, TW-2).

Phases:
  0. init sweep over beta (untrained maps) — TW-1b control.
  1. trained runs (DE on/off, beta=1.0) — TW-1, TW-1c, TW-2.
Writes results/tweek/lambda_results.json
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
from data_synth import TwoTimeSource  # noqa: E402
from telemetry import (lambda_from_displacement, per_token_lambda,  # noqa: E402
                       perturbation_lambda, jacobian_spectral_radius,
                       geometric_floor_fit)
from train import train_model  # noqa: E402

torch.set_num_threads(2)
VOCAB, D, NH, DFF, T = 24, 64, 4, 256, 96


def build(beta=1.0, depth_emb=False, seed=0):
    torch.manual_seed(seed)
    return TinyLoopLM(vocab=VOCAB, d=D, n_head=NH, d_ff=DFF, max_len=T,
                      max_loops=64, beta=beta, depth_emb=depth_emb)


@torch.no_grad()
def rollout_eval(model, src, K, n_batches=4, B=12, T=T):
    model.eval()
    L = np.zeros(K)
    Lh, Ls = [], []
    d_per_seq = []
    tok_lam, tok_r2 = [], []
    for _ in range(n_batches):
        x, y, hard = src.batch(B, T)
        out = model(x, K)
        logits, hidd = out["logits"], out["hiddens"]
        for k in range(K):
            ce = F.cross_entropy(logits[k].reshape(-1, VOCAB).float(),
                                 y.reshape(-1), reduction="none").reshape(B, T)
            L[k] += float(ce.mean()) / n_batches
            Lh.append((k, float(ce[hard].mean()) if hard.any() else float("nan")))
            Ls.append((k, float(ce[~hard].mean())))
        dpb = (hidd[1:] - hidd[:-1]).pow(2).mean(dim=(2, 3)).sqrt()  # [K,B]
        d_per_seq.append(dpb)
        pt = per_token_lambda(hidd)
        tok_lam.append(pt["median"])
        # store full per-token q's by re-running light stats (medians suffice per batch)
        tok_r2.append(pt["r2_median"])
    d_per_seq = torch.cat(d_per_seq, dim=1)  # [K, N]
    agg_hard, agg_steady = {}, {}
    for k, v in Lh:
        agg_hard.setdefault(k, []).append(v)
    for k, v in Ls:
        agg_steady.setdefault(k, []).append(v)
    return {
        "L_per_k": L.tolist(),
        "L_hard": [float(np.nanmean(agg_hard[k])) for k in range(K)],
        "L_steady": [float(np.mean(agg_steady[k])) for k in range(K)],
        "d_per_seq": d_per_seq,
        "tok_lam_batch_medians": tok_lam,
        "tok_r2_batch_medians": tok_r2,
    }


def probe_lambdas(model, src, probe_k=4, T=T):
    """Perturbation + Jacobian spectral radius of the core map R at trajectory point h_{probe_k}."""
    model.eval()
    x, _, _ = src.batch(8, T)
    with torch.no_grad():
        h = model.embed(x)
        state = None
        for k in range(probe_k + 1):
            h = model.core(h, k) if not model.two_time else model.core(h, k, state)
    hk = h.detach()

    def step_fn(h):
        return model.core(h, probe_k)

    lam_pert = perturbation_lambda(step_fn, hk, r=0.01, n_dirs=6, seed=7)
    lam_J = jacobian_spectral_radius(step_fn, hk, iters=10, seed=7)
    return lam_pert, lam_J


def bootstrap_lambda(d_per_seq: torch.Tensor, n_boot=200, seed=0):
    """Resample sequences, refit mean-curve lambda each time -> percentile CI."""
    rng = np.random.default_rng(seed)
    K, N = d_per_seq.shape
    d_np = d_per_seq.numpy()
    lams = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        d_mean = d_np[:, idx].mean(axis=1)
        fit = lambda_from_displacement(torch.tensor(d_mean))
        if fit.get("regime") == "contraction" and not math.isnan(fit["lambda"]):
            lams.append(fit["lambda"])
    if not lams:
        return {"lambda_mean": float("nan"), "ci90": [float("nan")] * 2}
    lams = np.array(lams)
    return {"lambda_mean": float(lams.mean()),
            "ci90": [float(np.percentile(lams, 5)), float(np.percentile(lams, 95))],
            "n_boot_ok": len(lams)}


def eval_and_report(model, src_val, name, K=64):
    print(f"[eval] {name} @ K={K}", flush=True)
    ev = rollout_eval(model, src_val, K)
    d_mean = ev["d_per_seq"].mean(dim=1)
    fit = lambda_from_displacement(d_mean)
    boot = bootstrap_lambda(ev["d_per_seq"])
    lam_pert, lam_J = probe_lambdas(model, src_val)
    ks = np.arange(1, K + 1, dtype=np.float64)
    ek = geometric_floor_fit(ks, np.array(ev["L_per_k"]))
    pt_all = per_token_lambda_light(model, src_val)
    rep = {
        "name": name, "K_eval": K,
        "d_fit": fit, "d_boot": boot,
        "lam_pert": lam_pert, "lam_J": lam_J,
        "E_over_K_fit": ek,
        "L_per_k": ev["L_per_k"],
        "L_hard": ev["L_hard"], "L_steady": ev["L_steady"],
        "per_token": pt_all,
    }
    return rep


@torch.no_grad()
def per_token_lambda_light(model, src_val, K=24, B=12, T=T, n_batches=2):
    med, q25, q75, r2m, qr = [], [], [], [], []
    for _ in range(n_batches):
        x, _, _ = src_val.batch(B, T)
        out = model(x, K)
        pt = per_token_lambda(out["hiddens"])
        med.append(pt["median"]); q25.append(pt["q25"]); q75.append(pt["q75"])
        r2m.append(pt["r2_median"]); qr.append(pt["q_ratio"])
    return {"median": float(np.mean(med)), "q25": float(np.mean(q25)),
            "q75": float(np.mean(q75)), "r2_median": float(np.mean(r2m)),
            "q75_q25_ratio": float(np.mean(qr))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results/tweek/lambda_results.json")
    args = ap.parse_args()
    steps = 40 if args.smoke else args.steps

    src_train = TwoTimeSource(seed=0)
    src_val = TwoTimeSource(seed=999)
    results = {"config": {"vocab": VOCAB, "d": D, "d_ff": DFF, "T": T,
                          "steps": steps, "K_train": [2, 8], "K_eval": 64,
                          "tau": 10.0, "p_noise": 0.05},
               "init_sweep": {}, "runs": {}}

    # ---- Phase 0: init sweep (untrained), TW-1b control ----
    for beta in [0.125, 0.25, 0.5, 1.0]:
        t0 = time.time()
        model = build(beta=beta, seed=0)
        x, _, _ = src_val.batch(8, T)
        with torch.no_grad():
            out = model(x, 48)
        d_mean = (out["hiddens"][1:] - out["hiddens"][:-1]).pow(2) \
            .mean(dim=(1, 2, 3)).sqrt()
        fit = lambda_from_displacement(d_mean)
        lam_pert, lam_J = probe_lambdas(model, src_val)
        results["init_sweep"][str(beta)] = {
            "lam_disp": fit["lambda"], "r2": fit["r2"],
            "lam_pert": lam_pert, "lam_J": lam_J, "sec": time.time() - t0}
        print(f"[init] beta={beta}: lam_disp={fit['lambda']:.4f} (r2={fit['r2']:.3f}) "
              f"lam_pert={lam_pert:.4f} lam_J={lam_J:.4f}", flush=True)

    # ---- Phase 1: trained runs ----
    runs = {
        "trained_beta1": dict(beta=1.0, depth_emb=False),
        "trained_beta1_DE": dict(beta=1.0, depth_emb=True),
    }
    for name, cfg in runs.items():
        t0 = time.time()
        model = build(seed=0, **cfg)
        print(f"[train] {name} params={count_params(model)} steps={steps}", flush=True)
        hist = train_model(model, src_train, steps=steps, B=12, T=T,
                           K_min=2, K_max=8, lr=3e-3, log_every=max(steps // 6, 20))
        rep = eval_and_report(model, src_val, name)
        rep["train_hist"] = hist
        rep["train_sec"] = time.time() - t0
        rep["n_params"] = count_params(model)
        rep["cfg"] = cfg
        results["runs"][name] = rep
        print(f"[done] {name}: lam_disp={rep['d_fit']['lambda']:.4f} "
              f"CI90={rep['d_boot']['ci90']} pert={rep['lam_pert']:.4f} "
              f"J={rep['lam_J']:.4f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    def default(o):
        if isinstance(o, torch.Tensor):
            return o.tolist()
        return str(o)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=default)
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
