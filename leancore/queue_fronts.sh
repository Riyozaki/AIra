#!/bin/bash
# Три фронта: (a1) muon+wdmask, (a2) muon mulr 0.05, (b) muon на V=16000 (prep6k)
set -e
export PYTHONPATH=/home/user/pylibs
export OPENBLAS_NUM_THREADS=2 MALLOC_TRIM_THRESHOLD_=1000000000 MALLOC_MMAP_THRESHOLD_=33554432
cd "$(dirname "$0")"
BASE="--kind ema --opt muon --steps 1500 --eval_every 100 --ssk 512 --ssfull 0.12 --seed 1"
python3 nano_lc.py $BASE --wdmask 1 --tag L_muwd1500s1 > results/L_muwd1500s1.out 2>&1
python3 nano_lc.py $BASE --mulr 0.05 --tag L_mu05_1500s1 > results/L_mu05_1500s1.out 2>&1
python3 nano_lc.py $BASE --data data/prep6k --tag L_mu6k_1500s1 > results/L_mu6k_1500s1.out 2>&1
echo FRONTS_DONE
