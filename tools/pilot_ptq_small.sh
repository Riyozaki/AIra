#!/bin/bash
# Re-train fp d64/d96 (deterministic, identical config to pilot_size) only to save checkpoints for PTQ curve.
cd "$(dirname "$0")/.."
C="--steps 600 --eval-bytes 500000 --tag pilot_size_rerun --log results/runs/pilot_size_rerun.jsonl"
.venv/bin/python tools/lm.py --d 64 --quant none --lr 3e-3 $C --save results/ckpt/fp_d64_s0.npz
.venv/bin/python tools/lm.py --d 96 --quant none --lr 3e-3 $C --save results/ckpt/fp_d96_s0.npz
.venv/bin/python tools/ptq_eval.py results/ckpt/fp_d64_s0.npz --d 64 --layers 4 --bits 8 4 3 --eval-bytes 500000 --tag pilot_ptq
.venv/bin/python tools/ptq_eval.py results/ckpt/fp_d96_s0.npz --d 96 --layers 4 --bits 8 4 3 --eval-bytes 500000 --tag pilot_ptq
