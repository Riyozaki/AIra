#!/bin/bash
# WAVE2: α-пара @1500 (перезапуск после снапшот-сброса) + два прогона советника.
# После КАЖДОГО прогона — авто-коммит и пуш (задел на снапшот-сбросы внутри хода).
export PYTHONPATH=/home/user/pylibs
export OPENBLAS_NUM_THREADS=2 MALLOC_TRIM_THRESHOLD_=1000000000 MALLOC_MMAP_THRESHOLD_=33554432
cd "$(dirname "$0")"

COMMIT () {
  git add -A results 2>/dev/null || true
  git commit -qm "wave2 progress: $1" 2>/dev/null || true
  git push -q origin HEAD:arena/01a04a42-aira 2>/dev/null && return 0
  git fetch -q origin arena/01a04a42-aira 2>/dev/null || true
  git reset -q --soft FETCH_HEAD 2>/dev/null || true
  git add -A results 2>/dev/null || true
  git commit -qm "wave2 recover: $1" 2>/dev/null || true
  git push -q origin HEAD:arena/01a04a42-aira 2>/dev/null || true
}

# фоновый толкатель частичных логов (jsonl пишется инкрементально) — потеря при сбросе ≤3 мин
( while :; do sleep 180; COMMIT periodic; done ) & PUSHPID=$!
trap 'kill $PUSHPID 2>/dev/null' EXIT

# 1-2. α-пара @1500, seed 1, muon, рецепт чемпиона (новый детерминированный инит)
python3 nano_lc.py --kind ema --opt muon --steps 1500 --eval_every 100 --ssk 512 --ssfull 0.12 --seed 1 --ssalpha 1.0  --tag L_a10_1500s1 > results/L_a10_1500s1.out 2>&1
COMMIT L_a10_1500s1
python3 nano_lc.py --kind ema --opt muon --steps 1500 --eval_every 100 --ssk 512 --ssfull 0.12 --seed 1 --ssalpha 0.75 --tag L_a75_1500s1 > results/L_a75_1500s1.out 2>&1
COMMIT L_a75_1500s1

# 3. muon + full-CE @1500, seed 1, ADR off — дыра «ss при muon на длинном горизонте» (прогноз советника 122.6, band 117.7–127.5)
python3 nano_lc.py --kind ema --opt muon --steps 1500 --eval_every 100 --seed 1 --tag L_muf1500s1 > results/L_muf1500s1.out 2>&1
COMMIT L_muf1500s1

# 4. muon + ss + ADR 0.5 @1500, seed 42 — дыра №1: чистая пара против L_ssa1500 (adam+ADR, 117.58) (мой прогноз 111–114)
python3 nano_lc.py --kind ema --opt muon --adr 0.5 --steps 1500 --eval_every 100 --ssk 512 --ssfull 0.12 --seed 42 --tag L_mADR1500s42 > results/L_mADR1500s42.out 2>&1
COMMIT L_mADR1500s42

echo WAVE2_DONE
