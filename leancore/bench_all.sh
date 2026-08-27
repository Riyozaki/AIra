#!/bin/bash
# bench_all.sh — один прогон: сборка, целостность PPL, скорость (чистое ядро), артефакты.
# Использование: ./bench_all.sh [model.lcw2]   (по умолчанию results/champ2k_qat_qs.lcw2)
set -e
cd "$(dirname "$0")"
M=${1:-results/champ2k_qat_qs.lcw2}
echo "== BUILD: gcc -O3 -march=native lc_stream.c"
gcc -O3 -march=native -o lc_stream lc_stream.c -lm
echo "== PPL (4053-токенный вал-срез)"
VF=data/prep/val.npy; [ -f "$VF" ] || VF=results/data/prep/val.npy
./lc_stream "$M" ppl "$VF" 2>/dev/null
echo "== SPEED: 600 токенов x3 (1 ядро, taskset -c 1)"
for i in 1 2 3; do taskset -c 1 ./lc_stream "$M" bench 600 2>/dev/null; done
echo "== FILE: $(du -h "$M" | cut -f1)"
