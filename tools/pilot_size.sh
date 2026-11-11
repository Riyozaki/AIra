#!/bin/bash
# Pilot size sweep, 600 steps (4.9M tokens), 1 seed, valid[:500k]; LR from pilot_lr (fp 3e-3, ternary 1e-2), LR transfer across d is an assumption.
cd "$(dirname "$0")/.."
C="--steps 600 --eval-bytes 500000 --tag pilot_size --log results/runs/pilot_size.jsonl"
.venv/bin/python tools/lm.py --d 128 --quant none --lr 3e-3 $C --save results/ckpt/fp_d128_s0.npz
.venv/bin/python tools/lm.py --d 64  --quant none --lr 3e-3 $C
.venv/bin/python tools/lm.py --d 96  --quant none --lr 3e-3 $C
.venv/bin/python tools/lm.py --d 192 --quant ternary --lr 1e-2 $C
.venv/bin/python tools/lm.py --d 192 --quant none --lr 3e-3 $C --save results/ckpt/fp_d192_s0.npz
.venv/bin/python tools/ptq_eval.py results/ckpt/fp_d128_s0.npz --d 128 --layers 4 --bits 8 4 3 --eval-bytes 500000 --tag pilot_ptq
.venv/bin/python tools/ptq_eval.py results/ckpt/fp_d192_s0.npz --d 192 --layers 4 --bits 8 4 3 --eval-bytes 500000 --tag pilot_ptq
echo SIZE_DONE
