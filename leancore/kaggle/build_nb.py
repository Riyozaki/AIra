#!/usr/bin/env python3
"""build_nb.py — собирает AIra_TOURNAMENT.ipynb из реальных файлов kaggle/.
Запуск: python3 build_nb.py → валидный ipynb рядом. Источник истины — файлы.
"""
import json, pathlib

HERE = pathlib.Path(__file__).resolve().parent

def rd(name):
    return (HERE / name).read_text(encoding="utf-8")

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}

LCXP = rd("lcxp.py")
FORK = rd("nano_lc_kg.py")
TT = rd("train_torch.py")
BR = rd("bracket.py")

cells = []

cells.append(md("""# ТУРНИР КОНФИГОВ LeanCore (ASHA-брекет) · Kaggle GPU T4×2 / P100

**Что делает**: отбор конфигов по непрерывным скалярам (`mulr, ssfull, ssk, ssalpha, lr`) с честной
методологией проекта AIra/LeanCore:

- **CTRL** = рецепт чемпиона — шагает все рунги как референс, не выбывает
- Парность: фиксированный инит у всех конфигов, `negrng`-разделение, сиды парные `[1,42]` → финал `[1,42,7,99]`
- Предохранители: авто-DQ «мёртвых» (`wn<3` после шага 100), NaN; **страховочный слот** лучшего наклона
  (ранний лидер ≠ поздний — наша измеренная ловушка ss/full)
- Рунги 60 → 200 → 500 → 1500 шагов, квоты 12 → 5 → 3 → финал (×4 сида)
- Всё сохраняется в `state.json` после каждой джобы → переживает таймаут сессии

## Как запустить
1. **Accelerator**: Settings → GPU T4 x2 (рекомендуется) или P100. Квота ~30 GPU-часов/нед (T4×2 тратит её быстрее — следите в интерфейсе).
2. **Данные**: три варианта (ячейка 3 разберёт сама): (а) приатачить Kaggle-датасет с `train.npy/val.npy/meta.json` корпуса `prep`; (б) положить их в `/kaggle/working/prep_data/`; (в) Internet ON → скачает с raw.githubusercontent нашей ветки.
3. **Run All**. Сначала пройдёт GPU-гейт честности (torch-GPU против numpy-CPU на 40 шагах CTRL), потом брекет.
4. По таймауту/завершении: **Save Version**. Для продолжения — новая сессия: Add Data → «Notebook Output Files» предыдущей версии → Run (state подхватится сам).

## Что вернуть в проект
Файлы из Output: `airaw/bracket_work/SUMMARY.md`, `PREDICTIONS.md`, `state.json`, и `results/*.jsonl` (+ `ckpt_*` финалистов). Сводку вставить в чат — я разберу против PREDICTIONS и запишу в TRICKS.
"""))

cells.append(md("## 0 · Окружение и пути"))

cells.append(code("""import os, sys, json, glob, time, shutil, subprocess, pathlib
ON_KAGGLE = os.path.exists("/kaggle/working")
BASE = "/kaggle/working/airaw" if ON_KAGGLE else os.path.abspath("./airaw_local")
os.makedirs(BASE, exist_ok=True)
print("kaggle:", ON_KAGGLE, "| BASE:", BASE, "| cpu:", os.cpu_count())
WORK = os.path.join(BASE, "bracket_work")
os.makedirs(WORK, exist_ok=True)
"""))

cells.append(md("## 1 · Файлы движка (lcxp / nano_lc_kg / train_torch / bracket)"))

cells.append(code(f"open(f'{{BASE}}/lcxp.py','w').write({json.dumps(LCXP)})\nprint('lcxp.py ok')"))

cells.append(code(f"open(f'{{BASE}}/nano_lc_kg.py','w').write({json.dumps(FORK)})\nprint('nano_lc_kg.py ok')"))

cells.append(code(f"open(f'{{BASE}}/train_torch.py','w').write({json.dumps(TT)})\nprint('train_torch.py ok')"))

cells.append(code(f"open(f'{{BASE}}/bracket.py','w').write({json.dumps(BR)})\nprint('bracket.py ok')"))

cells.append(md("""## 2 · Данные (prep: train.npy / val.npy / meta.json)

Порядок поиска: (а) любой приаттаченный датасет `/kaggle/input/*/` с тремя файлами;
(б) локальная папка `/kaggle/working/prep_data/`; (в) скачивание с raw.githubusercontent
(нужен Internet ON). + попытка подтянуть `lc_kernels.so` для ускорения numpy-пути (необязательно)."""))

cells.append(code("""import glob, os, shutil, urllib.request

dst = os.path.join(BASE, "data", "prep")
os.makedirs(dst, exist_ok=True)
need = ("train.npy", "val.npy", "meta.json")

def have(d): return all(os.path.exists(os.path.join(d, f)) for f in need)

src = None
for d in sorted(glob.glob("/kaggle/input/*")) + ["/kaggle/working/prep_data"]:
    if have(d): src = d; break

if src:
    for f in need: shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    print("данные из датасета:", src)
else:
    RAW = "https://raw.githubusercontent.com/Riyozaki/AIra/arena/01a04a42-aira/leancore/data/prep/"
    for f in need:
        print("скачиваю", f, "...")
        urllib.request.urlretrieve(RAW + f, os.path.join(dst, f))
    print("данные скачаны с GitHub raw")

# lc_kernels.so (ускоритель numpy-пути) — лучшая попытка, без падения
try:
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/Riyozaki/AIra/arena/01a04a42-aira/leancore/lc_kernels.so",
        os.path.join(BASE, "lc_kernels.so"))
    print("lc_kernels.so подтянут")
except Exception as e:
    print("lc_kernels.so не удалось (ок: numpy-фолбэк):", type(e).__name__)

import numpy as np, json as _j
tr = np.load(os.path.join(dst, "train.npy"))
print("train tokens:", len(tr), "| vocab:", _j.load(open(os.path.join(dst, 'meta.json')))['vocab'])
PREP = "data/prep"
"""))

cells.append(md("""## 3 · GPU-гейт честности и выбор движка

Если есть CUDA: прогон CTRL 40 шагов на **torch-GPU** и на **numpy-CPU** с одним сидом.
Критерий: |Δval_ppl| ≤ 2% → engine=torch (весь брекет на torch-GPU), иначе engine=numpy и
брекет едет на CPU-воркерах — но ни один бит плохих данных не попадёт в таблицу."""))

cells.append(code("""def run_once(engine, steps=40, seed=1):
    import subprocess, json
    env = dict(os.environ); env["LC_BACKEND"] = "numpy"
    if engine == "torch":
        cmd = [sys.executable, os.path.join(BASE, "train_torch.py"), "--steps", str(steps),
               "--eval_every", "20", "--ssk", "512", "--ssfull", "0.12", "--seed", str(seed),
               "--negrng", "1", "--trunknorm", "1", "--tag", f"gate_{engine}", "--data", PREP]
    else:
        cmd = [sys.executable, os.path.join(BASE, "nano_lc_kg.py"), "--kind", "ema", "--opt", "muon",
               "--steps", str(steps), "--eval_every", "20", "--ssk", "512", "--ssfull", "0.12",
               "--seed", str(seed), "--negrng", "1", "--trunknorm", "1", "--tag", f"gate_{engine}",
               "--data", PREP]
    r = subprocess.run(cmd, cwd=BASE, env=env, capture_output=True, text=True, timeout=3600)
    p = os.path.join(BASE, "results", f"run_gate_{engine}.jsonl")
    rows = [json.loads(l) for l in open(p) if l.strip().startswith("{")]
    return rows[-1]["val_ppl"], rows

NGPU, TORCH_OK = 0, False
try:
    import torch
    TORCH_OK = True
    NGPU = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print("torch", torch.__version__, "| cuda устройств:", NGPU,
          [torch.cuda.get_device_name(i) for i in range(NGPU)] if NGPU else "")
except Exception as e:
    print("torch недоступен:", type(e).__name__)

ENGINE = "numpy"
if NGPU > 0:
    ppl_t, rows_t = run_once("torch")
    ppl_n, rows_n = run_once("numpy")
    d = abs(ppl_t - ppl_n) / ppl_n
    print(f"GATE: torch={ppl_t} numpy={ppl_n} relΔ={d:.4%}")
    if d <= 0.02:
        ENGINE = "torch"; print("ГЕЙТ ПРОЙДЕН → engine=torch (GPU)")
    else:
        print("ГЕЙТ НЕ ПРОЙДЕН → engine=numpy (честный фолбэк)")
NWORKERS = NGPU if ENGINE == "torch" else max(1, min(3, (os.cpu_count() or 2) - 1))
print("ENGINE:", ENGINE, "| WORKERS:", NWORKERS)
"""))

cells.append(md("## 4 · Предсказания (зафиксированы ДО запуска — в PREDICTIONS.md)"))

cells.append(code("""print(open(os.path.join(WORK, "PREDICTIONS.md")).read() if os.path.exists(os.path.join(WORK, "PREDICTIONS.md")) else "(запишется init'ом ниже)")"""))

cells.append(md("""## 5 · Запуск брекета (init + воркеры + монитор)

`TARGET_HOURS` — мягкий стоп монитора (брекет сам сохраняет всё, что досчитал;
продолжение — следующая сессия). На T4×2: воркер на GPU. На P100: один. CPU-фолбэк: 3 воркера."""))

cells.append(code("""TARGET_HOURS = 10.0   # под сессию 12ч с запасом

def shout(*a): print(*a, flush=True)

subprocess.run([sys.executable, os.path.join(BASE, "bracket.py"), "init",
                "--workdir", WORK, "--data", PREP], cwd=BASE, check=True)

procs = []
for wid in range(NWORKERS):
    env = dict(os.environ)
    env["LC_BACKEND"] = "numpy"
    if ENGINE == "torch":
        cmd = [sys.executable, os.path.join(BASE, "bracket.py"), "worker", "--id", str(wid),
               "--engine", "torch", "--cuda", str(wid), "--workdir", WORK, "--data", PREP]
    else:
        env["OPENBLAS_NUM_THREADS"] = str(max(1, (os.cpu_count() or 2) // max(1, NWORKERS)))
        cmd = [sys.executable, os.path.join(BASE, "bracket.py"), "worker", "--id", str(wid),
               "--engine", "numpy", "--backend", "numpy", "--workdir", WORK, "--data", PREP]
    procs.append(subprocess.Popen(cmd, cwd=BASE, env=env))
    shout(f"воркер {wid} запущен (pid {procs[-1].pid})")

t0 = time.time()
while True:
    alive = sum(p.poll() is None for p in procs)
    st = subprocess.run([sys.executable, os.path.join(BASE, "bracket.py"), "status",
                         "--workdir", WORK], cwd=BASE, capture_output=True, text=True).stdout.strip()
    shout(f"--- t+{(time.time()-t0)/3600:.2f}ч | воркеров живо: {alive}\\n{st}")
    if alive == 0:
        shout("все воркеры завершились"); break
    if (time.time() - t0) / 3600 > TARGET_HOURS:
        shout("мягкий стоп монитора: процессы остаются, состояние сохранено"); break
    time.sleep(60)
"""))

cells.append(md("## 6 · Итоги и артефакты"))

cells.append(code("""subprocess.run([sys.executable, os.path.join(BASE, "bracket.py"), "summary",
                "--workdir", WORK], cwd=BASE, check=True)
print("=== файлы для скачивания/продолжения ===")
for f in sorted(glob.glob(os.path.join(WORK, "*")) + glob.glob(os.path.join(BASE, "results", "*.jsonl"))
                  + glob.glob(os.path.join(BASE, "results", "ckpt_bk_*.npz"))):
    print(f"{os.path.getsize(f)/1e6:8.2f} MB  {f}")
"""))

cells.append(md("""## 7 · Продолжение после таймаута

1. **Save Version** (все файлы Output сохранятся).
2. В новой сессии этого же ноутбука: **Add Data → Notebook Output Files** предыдущей версии.
   `init` сам найдёт `state.json` в `/kaggle/input/*/` и продолжит с места обрыва
   (зависшие джобы вернутся в pending, готовые не повторятся).
3. Не запускайте две сессии брекета одновременно на одном state — файловые локи не расчитаны на это.

### Что пасти в чат проекта
Содержимое `airaw/bracket_work/SUMMARY.md` целиком + финальную таблицу из ячейки 6.
Я сверю с PREDICTIONS, дам вердикты с метками и внесу в TRICKS."""))

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "kaggle": {"accelerator": "gpu", "dataSources": [], "dockerImageVersionId": 12,
                   "isInternetEnabled": True, "language": "python", "sourceType": "notebook"},
    },
    "cells": cells,
}

out = HERE / "AIra_TOURNAMENT.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[build_nb] {out} — {len(cells)} ячеек, {out.stat().st_size/1024:.0f} KB")
