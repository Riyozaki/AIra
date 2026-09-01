#!/bin/bash
set -e
export OPENBLAS_NUM_THREADS=2 MALLOC_TRIM_THRESHOLD_=1000000000 MALLOC_MMAP_THRESHOLD_=33554432
cd "$(dirname "$0")"
python3 nano_lc.py --kind ema --steps 1500 --eval_every 100 --ssk 512 --ssfull 0.12 --seed 1 --tag L_ssa1500s1 > results/L_ssa1500s1.out 2>&1
python3 nano_lc.py --kind ema --opt muon --steps 1500 --eval_every 100 --ssk 512 --ssfull 0.12 --seed 1 --tag L_mussa1500s1 > results/L_mussa1500s1.out 2>&1
echo SEED1_DONE
