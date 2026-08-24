"""T-week phase A2: compute-depth task (CA rule 110, r=8), angular metric, state-LN kernel.

Runs (TWEEK_ADDENDUM.md predictions TW-A1..TW-A5):
  ca_sum       — sum kernel on CA task
  ca_ln        — state-LN kernel on CA task
  tt_sum_long  — sum kernel on two-timescale task, long budget (confound control)
Writes results/tweek/ca_results.json
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
from data_synth import TwoTimeSource, CASource  # noqa: E402
from telemetry import (lambda_from_displacement, per_token_lambda,  # noqa: E402
                       angular_curve_per_seq, perturbation_lambda,
                       jacobian_spectral_radius, geometric_floor_fit)
from train import train_model  # noqa: E402
from exp_lambda import bootstrap_lambda  # noqa: E402

torch.set_num_threads(2)
D, NH, DFF = 64, 4, 256


def build(kernel: str, vocab: int, seed: int = 0):
    torch.manual_seed(seed)
    return TinyLoopLM(vocab=vocab, d=D, n_head=NH, d_ff=DFF, max_len=160,
                      max_loops=64, beta=1.0, depth_emb=False,
                      state_ln=(kernel == "ln"))


def acc_and_loss(model, src, K: int, n_batches: int = 4, B: int = 12):
    """Finite-aware eval: L_ans/acc_ans per k, norm + angular displacement curves."""
    model.eval()
    L_ans = np.zeros(K)
    acc_ans = np.zeros(K)
    finite_cnt = np.zeros(K)
    d_norm_seq, d_ang_seq = [], []
    first_nan = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y, hard = src.batch(B, 96)
            out = model(x, K)
            logits, hidd = out["logits"], out["hiddens"]
            fin = torch.stack([torch.isfinite(hidd[k]).all() for k in range(K)])
            fn = int(torch.argmin(fin.float() + 0.0)) if not bool(fin.all()) else K
            first_nan.append(fn)
            for k in range(K):
                if not bool(fin[k]):
                    continue
                ce = F.cross_entropy(logits[k].reshape(-1, logits.shape[-1]).float(),
                                     y.reshape(-1), reduction="none").reshape(B, -1)
                if hard is not None and hard.any():
                    L_ans[k] += float(ce[hard].mean())
                    pred = logits[k].argmax(-1)
                    acc_ans[k] += float((pred[hard] == y[hard]).float().mean())
                finite_cnt[k] += 1
            hidd_f = torch.nan_to_num(hidd, nan=0.0, posinf=0.0, neginf=0.0)
            d_norm_seq.append(
                (hidd_f[1:] - hidd_f[:-1]).pow(2).mean(dim=(2, 3)).sqrt())
            d_ang_seq.append(angular_curve_per_seq(hidd_f))
    d_norm = torch.cat(d_norm_seq, 1)
    d_ang = torch.cat(d_ang_seq, 1)
    nz = np.maximum(finite_cnt, 1)
    return {
        "L_ans": (L_ans / nz).tolist(),
        "acc_ans": (acc_ans / nz).tolist(),
        "finite_frac": (finite_cnt / n_batches).tolist(),
        "mean_first_nan_k": float(np.mean(first_nan)),
        "d_norm_seq": d_norm, "d_ang_seq": d_ang,
    }


def probe(model, src, probe_k: int = 4):
    model.eval()
    x, _, _ = src.batch(8, 96)
    with torch.no_grad():
        h = model.embed(x)
        for k in range(probe_k + 1):
            h = model.core(h, k)
            if model.ln_state is not None:
                h = model.ln_state(h)
    hk = h.detach()

    def step_fn(h):
        return model.core(h, probe_k)
    lam_pert = perturbation_lambda(step_fn, hk, r=0.01, n_dirs=6, seed=7)
    lam_J = jacobian_spectral_radius(step_fn, hk, iters=10, seed=7)
    return lam_pert, lam_J


def is_conv_task(src):
    return isinstance(src, CASource)


def run_variant(name, kernel, task, steps, out):
    vocab = 3 if task == "ca" else 24
    if task == "ca":
        src_train = CASource(seed=0)
        src_val = CASource(seed=999)
    else:
        src_train = TwoTimeSource(seed=0)
        src_val = TwoTimeSource(seed=999)
    model = build(kernel, vocab)
    print(f"[train] {name} kernel={kernel} task={task} "
          f"params={count_params(model)} steps={steps}", flush=True)
    t0 = time.time()
    hist = train_model(model, src_train, steps=steps, B=12, T=96,
                       K_min=2, K_max=12, lr=3e-3, log_every=max(steps // 5, 25))
    print(f"[eval] {name}", flush=True)
    ev = acc_and_loss(model, src_val, K=64)

    fit_norm = lambda_from_displacement(ev["d_norm_seq"].mean(1))
    boot_norm = bootstrap_lambda(ev["d_norm_seq"])
    fit_ang = lambda_from_displacement(ev["d_ang_seq"].mean(1))
    boot_ang = bootstrap_lambda(ev["d_ang_seq"])
    lam_pert, lam_J = probe(model, src_val)

    # per-token angular slopes (2 batches, K=24)
    pt = {"norm": [], "ang": []}
    with torch.no_grad():
        for _ in range(2):
            x, _, _ = src_val.batch(12, 96)
            o = model(x, 24)
            hf = torch.nan_to_num(o["hiddens"], nan=0.0, posinf=0.0, neginf=0.0)
            pt["norm"].append(per_token_lambda(hf, angular=False))
            pt["ang"].append(per_token_lambda(hf, angular=True))
    agg = lambda rows: {k: float(np.mean([r[k] for r in rows])) for k in
                        ["median", "q25", "q75", "r2_median", "q_ratio"]}

    # E(K) on finite answer-loss region
    L_ans = np.array(ev["L_ans"])
    ff = np.array(ev["finite_frac"]) > 0.5
    ks = np.arange(1, 65)[ff]
    ek = geometric_floor_fit(ks.astype(float), L_ans[ff]) if ff.sum() >= 6 else None
    rep = {
        "name": name, "kernel": kernel, "task": task,
        "n_params": count_params(model), "train_sec": time.time() - t0,
        "train_loss_final": hist[-1]["loss"] if hist else None,
        "L_ans": ev["L_ans"], "acc_ans": ev["acc_ans"],
        "finite_frac": ev["finite_frac"], "mean_first_nan_k": ev["mean_first_nan_k"],
        "fit_norm": {k: v for k, v in fit_norm.items() if k != "d"},
        "boot_norm": boot_norm,
        "fit_ang": {k: v for k, v in fit_ang.items() if k != "d"},
        "boot_ang": boot_ang,
        "d_ang_curve": ev["d_ang_seq"].mean(1).tolist(),
        "d_norm_curve": ev["d_norm_seq"].mean(1).tolist(),
        "lam_pert": lam_pert, "lam_J": lam_J,
        "pt_norm": agg(pt["norm"]), "pt_ang": agg(pt["ang"]),
        "E_ans_K_fit": ek,
    }
    out["runs"][name] = rep
    print(f"[done] {name}: acc@16={ev['acc_ans'][15] if len(ev['acc_ans'])>15 else float('nan'):.3f} "
          f"lam_dir={fit_ang['lambda']:.4f}({fit_ang['regime']},r2={fit_ang['r2']:.2f}) "
          f"lam_norm={fit_norm['lambda']:.4f} PR/J={lam_pert:.3f}/{lam_J:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--variant", default="all")
    ap.add_argument("--out", default="results/tweek/ca_results.json")
    args = ap.parse_args()
    steps = 40 if args.smoke else args.steps
    variants = {
        "ca_sum": ("sum", "ca"),
        "ca_ln": ("ln", "ca"),
        "tt_sum_long": ("sum", "tt"),
    }
    out = {"config": {"D": D, "DFF": DFF, "steps": steps, "K_train": [2, 12],
                      "K_eval": 64, "ca": {"rule": 110, "L": 32, "r": 8}},
           "runs": {}}
    for name, (kernel, task) in variants.items():
        if args.variant not in ("all", name):
            continue
        run_variant(name, kernel, task, steps, out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
