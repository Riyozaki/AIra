# T0_RUNBOOK — первый день на железе (A1-путь, docs/A1_PLAN.md)

Цель документа: искать в TF/блогах ничего не придётся — все команды первого дня здесь.
Протокол и кил-критерии: **docs/A1_PLAN.md** (AX-1..AX-5 заморожены 2026-08-24).
Железо: T0 по docs/HARDWARE.md (32–64 ГБ RAM; VRAM 16–24 ГБ для A1a-скорости; NVMe ≥100 ГБ).

## 0. Развёртывание (20 мин)

```bash
git clone https://github.com/Riyozaki/AIra.git && cd AIra
git checkout arena/01a0333c-aira        # рабочая ветка сессии
python3 -m venv .venv && . .venv/bin/activate
pip install torch numpy safetensors huggingface_hub datasets
python3 lab/selftest.py --quick          # регрессия: должно быть 4/4 green
python3 lab/donor_patch.py               # на T0 HF достижим → сверит индекс весов
```

## 1. Донор (15 мин)

```bash
huggingface-cli download Qwen/Qwen2.5-0.5B --include "*.safetensors*" "config.json"
python3 - <<'EOF'
# склейка шардов -> единый state_dict для lab/donor_patch.load_donor_state(...)
from safetensors.torch import load_file
import glob, torch
shards = sorted(glob.glob("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/*/model-*.safetensors"))
state = {}
for s in shards: state.update(load_file(s))
torch.save(state, "donor_qwen25_05b.pt")     # ~1 ГБ
print(len(state), "tensors")
EOF
```

## 2. Данные A1a (параллельно фоном, 1–2 ч)

```bash
python3 - <<'EOF'
from datasets import load_dataset
fw = load_dataset("HuggingFaceTB/fineweb-edu", "sample-10BT", split="train", streaming=True)
# вал: 20k доков; трейн: поток на 1B токенов (токенизация — Qwen2.5 tokenizer)
fw.take(20000).to_parquet("data/fw_val.parquet") if hasattr(fw.take(20000),'to_parquet') else None
EOF
# плюс верифицируемая смесь: GSM8K train + MATH train + MetaMathQA-срез (seeds A1c)
```

## 3. Smoke AX-1 (час первый на GPU)

```bash
python3 lab/donor_a1.py --donor donor_qwen25_05b.pt --steps 200 --K 1 --batch 4 --seq 1024
# ВОРОТА AX-1: ppl(patched, K=1) / ppl(донор) ∈ [0.97, 1.03] после 200 шагов,
# и ≤1.05+ε на 0 шагов (init=копия слоя). Просадка >5% → KIL-клаусула AX-1.
```
(`lab/donor_a1.py` — тренировочный драйвер вокруг donor_patch.DonorLM:
fp32-master веса + bf16 autocast; диета K ~ LogNormal(ln4.5, 0.9)∩[2,64];
deep supervision на всех k; лосс только на patch-параметрах 14.9M.)

## 4. A1a полный прогон (10–30 ч)

```bash
python3 lab/donor_a1.py --steps 6000 --batch 16 --seq 2048 \
    --kdist lognorm --out results/a1/a1a.json --telemetry-every 100
# одновременно control: --disable-loop (compute-matched двойник, SCOREBOARD §5)
```

Телеметрия каждые 100 шагов (автоматом из lab/donor_patch.loop_telemetry):
λ̂_dir, d_norm(k), acc/ppl при K∈{1,4,16,64}, дрейф acc@64−acc@16 (AX-4), fp-конечность.

## 5. Вердикты (писать как-есть, fail-as-written легитимен)

| строка | по протоколу | куда пишем |
|---|---|---|
| AX-1 identity-proxy | A1_PLAN §6 | docs/A1_RESULTS.md (новый) |
| AX-2 ось качества (patched vs control) | сравнение на GSM8K-держателе | docs/A1_RESULTS.md + SCOREBOARD C1/C6 |
| AX-3 связанность | λ̂ ≤ 0.85, геометрический спад | docs/A1_RESULTS.md |
| AX-4 дрейф-диета | ≥ −5 п.п. | docs/A1_RESULTS.md |
| TW-H (бонус-час) | λ̂ на Huginn-3.5B весах из HF | docs/A1_RESULTS.md §H |

## 6. Если что-то пошло не так

- **OOM**: seq 1024→512, batch 4→1 + grad-accum; патч это 14.9M, донор fp16 = 1 ГБ — OOM значит утечку графа (проверить torch.no_grad на frozen-слоях, в donor_patch.py уже есть).
- **NaN/overflow**: это сигнал AX-3 в сторону KIL — не «чинить» уменьшением lr молча, а зафиксировать λ̂ в момент взрыва (телеметрия и так пишет).
- **ppl просел >5% на smoke**: кил AX-1 → ревизия патча РОВНО один раз (кандидаты: patch_at=8 или 16 вместо 12; l_copy=слой l вместо l+1).
- **HF недостижим с машины**: зеркало modelscope.cn (Qwen официально выкладывает), команды идентичны.
