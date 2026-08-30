#!/bin/bash
# WAVE2-остаток: только mADR1500s42 (снапшот убил на шаге 300). Коммит+пуш в конец и каждые 3 мин.
export PYTHONPATH=/home/user/pylibs
export OPENBLAS_NUM_THREADS=2 MALLOC_TRIM_THRESHOLD_=1000000000 MALLOC_MMAP_THRESHOLD_=33554432
cd "$(dirname "$0")"
COMMIT () {
  git add -A results 2>/dev/null || true
  git commit -qm "wave2 rest: $1" 2>/dev/null || true
  git push -q origin HEAD:arena/01a04a42-aira 2>/dev/null && return 0
  git fetch -q origin arena/01a04a42-aira 2>/dev/null || true
  git merge -q --no-edit FETCH_HEAD 2>/dev/null || git rebase --abort 2>/dev/null || true
  git add -A results 2>/dev/null || true
  git commit -qm "wave2 rest recover: $1" 2>/dev/null || true
  git push -q origin HEAD:arena/01a04a42-aira 2>/dev/null || true
}
( while :; do sleep 180; COMMIT periodic; done ) & PUSHPID=$!
trap 'kill $PUSHPID 2>/dev/null' EXIT
python3 nano_lc.py --kind ema --opt muon --adr 0.5 --steps 1500 --eval_every 100 --ssk 512 --ssfull 0.12 --seed 42 --tag L_mADR1500s42 > results/L_mADR1500s42.out 2>&1
COMMIT L_mADR1500s42_final
echo REST_DONE
