#!/bin/bash
# Воспроизводимый A/B волны-6 (sampled softmax + malloc-фиксы).
# ВАЖНО: env ниже лечит page-fault шторм glibc (см. WAVE6.md §2) — нужен ОБОИМ режимам.
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=2 \
       MALLOC_TRIM_THRESHOLD_=1000000000 \
       MALLOC_MMAP_THRESHOLD_=33554432
STEPS=${1:-500}
echo "== baseline full-CE =="
python3 nano_lc.py --kind ema --adr 0.5 --opt muon --tag ab6_full$STEPS --steps $STEPS --eval_every 100
echo "== sampled softmax K=512 =="
python3 nano_lc.py --kind ema --adr 0.5 --opt muon --ssk 512 --tag ab6_ss$STEPS --steps $STEPS --eval_every 100
echo "== sampled softmax K=512 + аннилинг хвоста 12% =="
python3 nano_lc.py --kind ema --adr 0.5 --opt muon --ssk 512 --ssfull 0.12 --tag ab6_ssa$STEPS --steps $STEPS --eval_every 100
echo "== erank96 + ss512 =="
python3 nano_lc.py --kind ema --adr 0.5 --opt muon --erank 96 --ssk 512 --tag ab6_er_ss$STEPS --steps $STEPS --eval_every 100
echo "AB6_DONE"
