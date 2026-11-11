"""Pilot analysis for docs/03: effective-parameter ratio c and size multiplier k_size of ternary vs fp(+PTQ).

Method (fixed before looking at numbers, docs/03 §6): piecewise-linear interpolation of bpb vs log(N) on the fp
curve (same steps/tokens/recipe). No extrapolation: outside the fp range we print a bound.
Storage: ternary = 1.6 bit * N_lin + 16 bit * (N_emb + N_other); fp-format F = b_F * N_lin + 16 * (N_emb + N_other).
"""
import json, math, sys
from collections import defaultdict

runs = [json.loads(l) for l in open("results/runs/pilot_size.jsonl")]
ptq = [json.loads(l) for l in open("results/runs/ptq.jsonl")]
fp = sorted([r for r in runs if r["quant"] == "none"], key=lambda r: r["N_lin"])
tern = sorted([r for r in runs if r["quant"] == "ternary"], key=lambda r: r["N_lin"])
# include the ternary d=128 point from the LR pilot (same config, best LR) - selected by valid, flagged
lr_runs = [json.loads(l) for l in open("results/runs/pilot_lr.jsonl")]
t128 = min([r for r in lr_runs if r["quant"] == "ternary"], key=lambda r: r["bpb"])
tern = sorted(tern + [t128], key=lambda r: r["N_lin"])
fp_meta = {r["d"]: r for r in fp}


def interp_logx(points, y):
    """points: [(x, y)] with y decreasing in x. returns x at y (log-linear), or None + bound side."""
    pts = sorted(points)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if (y0 - y) * (y1 - y) <= 0 and y0 != y1:
            f = (y0 - y) / (y0 - y1)
            return math.exp(math.log(x0) + f * (math.log(x1) - math.log(x0))), ""
    if y > max(p[1] for p in pts):
        return min(p[0] for p in pts), "<"  # worse than smallest fp: true value is below
    return max(p[0] for p in pts), ">"


def other(r):
    return r["N_emb"] + r["N_other"]


def storage(r, b):
    return b * r["N_lin"] + 16 * other(r)


print("| ternary d | N_lin | bpb_t | fp-equiv N_lin | c | k_size vs bf16 | vs int8-PTQ | vs int4-PTQ |")
print("|---|---|---|---|---|---|---|---|")
rows = []
for t in tern:
    Nfp, s = interp_logx([(r["N_lin"], r["bpb"]) for r in fp], t["bpb"])
    c = Nfp / t["N_lin"]
    St = storage(t, 1.6)
    ks = {}
    for name, bits, b_eff in (("bf16", None, 16), ("int8", 8, 8.5), ("int4", 4, 4.5)):
        pts = []
        for r in fp:
            if bits is None:
                pts.append((storage(r, 16), r["bpb"]))
            else:
                q = [p for p in ptq if p["d"] == r["d"] and p["ptq_bits"] == bits]
                if q:
                    pts.append((storage(r, b_eff), q[-1]["bpb"]))
        if len(pts) < 2:
            ks[name] = "n/a"; continue
        S, sb = interp_logx(pts, t["bpb"])
        ks[name] = f"{sb}{S / St:.2f}" + ("" if len(pts) == len(fp) else f" ({len(pts)} pts)")
    rows.append(dict(d=t["d"], c=c, bound=s))
    print(f"| {t['d']} | {t['N_lin']} | {t['bpb']:.4f} | {s}{Nfp:,.0f} | {s}{c:.3f} | {ks['bf16']} | {ks['int8']} | {ks['int4']} |")

print()
print("fp curve:", [(r["d"], r["N_lin"], r["bpb"]) for r in fp])
print("PTQ:", sorted({(p["d"], p["ptq_bits"], p["bpb"]) for p in ptq}))
print("train time ratio ternary/fp at equal d:",
      {t["d"]: round(t["train_s"] / fp_meta[t["d"]]["train_s"], 3) for t in tern if t["d"] in fp_meta})
