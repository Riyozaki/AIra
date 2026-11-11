#!/bin/bash
# docs/04 §8: model R (residual over mKN K=7, table on train[:80M], NN on train[80M:90M]), d=128, 600 steps, 1 seed.
cd "$(dirname "$0")/.."
C="--d 128 --ngram 7 --steps 600 --eval-bytes 500000 --tag pilot_ngR --log results/runs/pilot_ngram.jsonl"
for lr in 1e-3 3e-3 1e-2; do .venv/bin/python tools/lm.py $C --lr $lr --save results/ckpt/ngR_d128_lr${lr}.npz; done
echo NG_DONE
