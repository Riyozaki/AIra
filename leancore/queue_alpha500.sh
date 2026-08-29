#!/bin/bash
set -e
export PYTHONPATH=/home/user/pylibs
export OPENBLAS_NUM_THREADS=2 MALLOC_TRIM_THRESHOLD_=1000000000 MALLOC_MMAP_THRESHOLD_=33554432
cd "$(dirname "$0")"
BASE="--kind ema --opt muon --steps 500 --eval_every 50 --ssk 512 --ssfull 0.12 --seed 1"
python3 nano_lc.py $BASE --tag A_ctrl500 > results/A_ctrl500.out 2>&1
python3 nano_lc.py $BASE --ssalpha 0.75 --tag A_a75_500 > results/A_a75_500.out 2>&1
echo ALPHA500_DONE
