"""Lab regression gate: run every stand in --smoke mode plus donor_patch tests.

Purpose: the retired toy stands (TWEEK/R3/G0S) are kept ONLY as core
regression tests (G0S_RESULTS §4), so this script is the project's
"does the kernel still behave" check after any models.py/telemetry.py edit.

HW-0 compliant: pure CPU, a few minutes total, no downloads.

Usage:  python3 lab/selftest.py            # full smoke battery
        python3 lab/selftest.py --quick    # 60-step runs only
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "selftest"
OUT.mkdir(parents=True, exist_ok=True)

PY = sys.executable


def run(name, cmd, timeout):
    t0 = time.time()
    print(f"[selftest] {name}: {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       timeout=timeout)
    dt = time.time() - t0
    ok = p.returncode == 0
    tail = (p.stdout + p.stderr).strip().splitlines()
    print(f"[selftest] {name}: {'OK' if ok else 'FAIL'} in {dt:.0f}s")
    for line in tail[-6:]:
        print(f"    {line}")
    (OUT / f"{name}.log").write_text(p.stdout + "\n--- STDERR ---\n" + p.stderr)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    steps = 60 if a.quick else None

    jobs = []
    # 1) module self-tests (D-002 patch geometry + loop-law properties)
    jobs.append(("donor_patch", [PY, "lab/donor_patch.py"], 900))
    # 2) lambda stand smoke (TW-0 instrument + D-002 kernel on CA rule 110)
    lam_cmd = [PY, "lab/exp_lambda.py", "--smoke",
               "--out", str(OUT / "lambda.json")] if a.quick \
        else [PY, "lab/exp_lambda.py", "--steps", "200",
              "--out", str(OUT / "lambda.json")]
    jobs.append(("lambda", lam_cmd, 3600))
    # 3) m-sweep smoke (two-timescale core + slow_ln regression)
    ms_cmd = [PY, "lab/exp_msweep.py", "--smoke", "--state_ln", "--slow_ln",
              "--out", str(OUT / "msweep.json")] if a.quick else \
        [PY, "lab/exp_msweep.py", "--steps", "300", "--state_ln", "--slow_ln",
         "--out", str(OUT / "msweep.json")]
    jobs.append(("msweep", ms_cmd, 3600))
    # 4) sudoku stand (board9/wave/inject code paths; full pools come from cache)
    su_cmd = [PY, "lab/exp_sudoku.py", "--smoke", "--variants", "ln_big_we",
              "--out", str(OUT / "sudoku.json")] if a.quick else \
        [PY, "lab/exp_sudoku.py", "--steps", "400", "--variants", "ln_big_we",
         "--out", str(OUT / "sudoku.json")]
    jobs.append(("sudoku", su_cmd, 5400))

    results = {}
    for name, cmd, to in jobs:
        try:
            results[name] = run(name, cmd, to)
        except subprocess.TimeoutExpired:
            print(f"[selftest] {name}: TIMEOUT")
            results[name] = False

    n_ok = sum(results.values())
    print(f"[selftest] ==== {n_ok}/{len(results)} green ====")
    (OUT / "summary.json").write_text(json.dumps(
        {"results": results, "quick": a.quick}, indent=2))
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
