#!/bin/bash
# Grid extension (best LR 1e-3 was on the edge) + B-NN d128 checkpoint re-train (deterministic) for model M.
cd "$(dirname "$0")/.."
.venv/bin/python tools/lm.py --d 128 --ngram 7 --steps 600 --eval-bytes 500000 --tag pilot_ngR --log results/runs/pilot_ngram.jsonl --lr 3e-4 --save results/ckpt/ngR_d128_lr3e-4.npz
.venv/bin/python tools/lm.py --d 128 --quant none --lr 3e-3 --steps 600 --eval-bytes 500000 --tag pilot_size_rerun2 --log results/runs/pilot_size_rerun.jsonl --save results/ckpt/fp_d128_s0.npz
.venv/bin/python tools/mix_eval.py results/ckpt/fp_d128_s0.npz --d 128 --K 7
echo NG2_DONE
