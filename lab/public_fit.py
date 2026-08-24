"""T-week run C: functional-form competition on public curves (TW-4 / P-12-dataset).

Reads lab/public_curves/*.json: {"name", "metric", "points": [{"r", "value"}],
"provenance": "...which table/figure of which arXiv id..."}.
Fits geometric+floor (ours) vs pure power law (Parcae-world) and compares by AICc.
Writes results/tweek/public_fits.json
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from telemetry import geometric_floor_fit, power_fit, aicc  # noqa: E402


def main():
    out = {"curves": {}, "verdicts": {}}
    for path in sorted(glob.glob("lab/public_curves/*.json")):
        with open(path) as f:
            data = json.load(f)
        pts = sorted(data["points"], key=lambda p: p["r"])
        if len(pts) < 5:
            out["curves"][data["name"]] = {"status": "too_few_points",
                                           "n": len(pts)}
            continue
        r = np.array([p["r"] for p in pts], dtype=np.float64)
        y = np.array([p["value"] for p in pts], dtype=np.float64)
        # loss-like curves should DECREASE with r for the geometric fit;
        # accuracy-like curves increase -> flip to loss = max - value + eps
        increasing = np.corrcoef(np.log(r), y)[0, 1] > 0
        yl = (y.max() - y + 1e-4) if increasing else y
        g = geometric_floor_fit(r, yl)
        pw = power_fit(r, yl)
        n = len(r)
        row = {"n_points": n, "metric": data.get("metric"),
               "provenance": data.get("provenance"),
               "transformed_from_accuracy": bool(increasing)}
        if g:
            row["geometric_floor"] = {**g, "aicc": aicc(g["sse"], n, 3)}
        row["power"] = {**pw, "aicc": aicc(pw["sse_log"], n, 2)}
        out["curves"][data["name"]] = row
        if g:
            win = "geometric+floor" if row["geometric_floor"]["aicc"] + 2 <= row["power"]["aicc"] \
                else ("power" if row["power"]["aicc"] + 2 <= row["geometric_floor"]["aicc"] else "tie")
            out["verdicts"][data["name"]] = {
                "winner": win,
                "delta_aicc_geometric_minus_power":
                    row["geometric_floor"]["aicc"] - row["power"]["aicc"]}
    os.makedirs("results/tweek", exist_ok=True)
    with open("results/tweek/public_fits.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out.get("verdicts", {}), indent=2))
    print("[saved] results/tweek/public_fits.json")


if __name__ == "__main__":
    main()
