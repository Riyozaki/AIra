#!/bin/bash
# Воспроизведение @500-проб рычагов (TRICKS.md): ctrl / wdmask / wema / muon
set -e
export OPENBLAS_NUM_THREADS=2 MALLOC_TRIM_THRESHOLD_=1000000000 MALLOC_MMAP_THRESHOLD_=33554432
cd "$(dirname "$0")"
BASE="--kind ema --steps 500 --eval_every 50 --ssk 512 --ssfull 0.12"
python3 nano_lc.py $BASE --tag P_ctrl500   > results/P_ctrl500.out 2>&1
python3 nano_lc.py $BASE --wdmask 1 --tag P_wd500 > results/P_wd500.out 2>&1
python3 nano_lc.py $BASE --wema 0.99 --tag P_wema500 > results/P_wema500.out 2>&1
python3 nano_lc.py $BASE --opt muon --tag P_mu500 > results/P_mu500.out 2>&1
echo LEVERS500_DONE
