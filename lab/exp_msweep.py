"""T-week run B: two-timescale m-sweep (TW-3, TW-3b).

Slow block applies every m loop iterations. Task tau=10 known.
Prediction: argmin_m val-loss inside [m*_pred/2, 2*m*_pred],
m*_pred = ln(tau) / ln(1/lam_L), lam_L = measured per-step slow contraction at m=1.
Writes results/tweek/msweep_results.json
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
from train import train_model  # noqa: E402

torch.set_num_threads(2)
VOCAB, D, NH, DFF, T = 24, 64, 4, 256, 96
TAU = 10.0


def build(m, use_slow=True, beta=1.0, seed=0, state_ln=False, slow_ln=False):
    torch.manual_seed(seed)
    return TinyLoopLM(vocab=VOCAB, d=D, n_head=NH, d_ff=DFF, max_len=T,
                      max_loops=64, beta=beta, depth_emb=False,
                      two_time=True, slow_m=m, use_slow=use_slow,
                      state_ln=state_ln, slow_ln=slow_ln)


@torch.no_grad()
def eval_m(model, src, K=64, n_batches=4, B=12, T=T):
    """Final-K loss + hard-position CE + per-step slow contraction."""
    model.eval()
    L_fin, Lh_fin, acc_hard = [], [], []
    s_updates = None
    lam_steps = []
    for _ in range(n_batches):
        x, y, hard = src.batch(B, T)
        out = model(x, K)
        logits = out["logits"]
        ce = F.cross_entropy(logits[K - 1].reshape(-1, VOCAB).float(),
                             y.reshape(-1), reduction="none").reshape(B, T)
        L_fin.append(float(ce.mean()))
        Lh_fin.append(float(ce[hard].mean()) if hard.any() else float("nan"))
        if hard.any():
            pred = logits[K - 1].argmax(-1)
            acc_hard.append(float((pred[hard] == y[hard]).float().mean()))
        st = out.get("slow_states")
        if st is not None:
            m = model.core.m
            upd = list(range(0, K, m))  # iteration indices where slow was applied
            ss = st[upd]  # [J, B, D]
            if ss.shape[0] >= 4:
                d = (ss[1:] - ss[:-1]).pow(2).mean(dim=(1, 2)).sqrt()  # [J-1]
                d = d[2:]  # drop zero-init transient
                ratios = (d[1:] / d[:-1].clamp_min(1e-12)) ** (1.0 / m)
                lam_steps.append(float(ratios.median()))
    res = {"L_final": float(np.mean(L_fin)),
           "L_hard_final": float(np.nanmean(Lh_fin)),
           "acc_hard_final": float(np.mean(acc_hard))}
    if lam_steps:
        res["lam_slow_step"] = float(np.mean(lam_steps))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--state_ln", action="store_true",
                    help="stabilized kernel: LN on state each iteration (TW-3 round 2)")
    ap.add_argument("--slow_ln", action="store_true",
                    help="stabilized slow loop: LN on slow state (TW-3 round 3)")
    ap.add_argument("--out", default="results/tweek/msweep_results.json")
    args = ap.parse_args()
    steps = 30 if args.smoke else args.steps

    src_train = TwoTimeSource(seed=0)
    src_val = TwoTimeSource(seed=999)
    results = {"config": {"vocab": VOCAB, "d": D, "T": T, "steps": steps,
                          "K_train": [2, 10], "K_eval": 64, "tau": TAU},
               "runs": {}}

    grid = [("m1", 1, True), ("m2", 2, True), ("m4", 4, True), ("m8", 8, True),
            ("m16", 16, True), ("noslow", 1, False)]
    for name, m, use_slow in grid:
        t0 = time.time()
        model = build(m=m, use_slow=use_slow, state_ln=args.state_ln,
                      slow_ln=args.slow_ln)
        print(f"[train] {name} (m={m}, slow={use_slow}) "
              f"params={count_params(model)} steps={steps}", flush=True)
        train_model(model, src_train, steps=steps, B=12, T=T,
                    K_min=2, K_max=10, lr=3e-3, log_every=max(steps // 4, 15))
        ev = eval_m(model, src_val)
        ev["train_sec"] = time.time() - t0
        ev["n_params"] = count_params(model)
        ev["m"] = m
        ev["use_slow"] = use_slow
        results["runs"][name] = ev
        print(f"[done] {name}: L_final={ev['L_final']:.4f} "
              f"hard={ev['L_hard_final']:.4f} accH={ev['acc_hard_final']:.3f} "
              f"lam_slow={ev.get('lam_slow_step', float('nan')):.4f}", flush=True)

    # ---- TW-3 derivations ----
    lam_L = results["runs"]["m1"].get("lam_slow_step")
    order = ["m1", "m2", "m4", "m8", "m16"]
    Ls = {n: results["runs"][n]["L_final"] for n in order}
    argmin_m = min(order, key=lambda n: Ls[n])
    m_emp = results["runs"][argmin_m]["m"]
    deriv = {"lam_L": lam_L, "argmin_run": argmin_m, "m_empirical": m_emp}
    if lam_L and 0 < lam_L < 1:
        m_pred = math.log(TAU) / math.log(1.0 / lam_L)
        deriv["m_star_pred"] = m_pred
        deriv["window"] = [m_pred / 2, 2 * m_pred]
        deriv["TW3_pass"] = bool(m_pred / 2 <= m_emp <= 2 * m_pred)
    else:
        deriv["m_star_pred"] = None
        deriv["TW3_pass"] = False
        deriv["note"] = "slow map does not contract (lam_L>=1) or missing: T4 premise failed"
    best_slow = min(Ls[n] for n in order)
    deriv["m1_loss"] = Ls["m1"]
    deriv["TW3b_pass"] = bool(Ls["m1"] > min(Ls[n] for n in order if n != "m1") + 1e-3)
    deriv["noslow_loss"] = results["runs"]["noslow"]["L_final"]
    results["derived"] = deriv

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
