#!/bin/bash
set -e
export OPENBLAS_NUM_THREADS=2 MALLOC_TRIM_THRESHOLD_=1000000000 MALLOC_MMAP_THRESHOLD_=33554432
cd "$(dirname "$0")"
BASE="--kind ema --steps 500 --eval_every 50 --ssk 512 --ssfull 0.12"
python3 nano_lc.py $BASE --tag K_ctrl500 > results/K_ctrl500.out 2>&1
python3 nano_lc.py $BASE --kronfc 1 --tag K_kron500 > results/K_kron500.out 2>&1
echo KRON500_DONE
