#!/bin/bash
cd /home/user/AIra/leancore
export OPENBLAS_NUM_THREADS=1
python3 nano_lc.py --kind attn --tag r2_attn450 --steps 450 &
python3 nano_lc.py --kind ema --tag r2_ema450 --steps 450 &
python3 nano_lc.py --kind ema --adr 0.5 --tag r2_emaadr450 --steps 450 &
wait
echo R2_SUITE_DONE