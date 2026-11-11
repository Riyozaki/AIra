#!/bin/bash
# Pilot LR sweep, d=128 L=4, 600 steps (4.9M tokens), 1 seed, valid[:500k]. Equal tuning budget: 3 LR per branch.
cd "$(dirname "$0")/.."
for lr in 1e-3 3e-3 1e-2; do .venv/bin/python tools/lm.py --d 128 --quant none    --lr $lr --steps 600 --eval-bytes 500000 --tag pilot_lr --log results/runs/pilot_lr.jsonl; done
for lr in 3e-3 1e-2 3e-2; do .venv/bin/python tools/lm.py --d 128 --quant ternary --lr $lr --steps 600 --eval-bytes 500000 --tag pilot_lr --log results/runs/pilot_lr.jsonl; done
echo SWEEP_DONE
