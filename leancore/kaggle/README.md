# Kaggle-пакет: турнир конфигов LeanCore

## Файлы
| файл | роль | валидация |
|---|---|---|
| `AIra_TOURNAMENT.ipynb` | ноутбук турнира (генерируется `build_nb.py`) | JSON+компиляция ячеек [измерено] |
| `lcxp.py` | backend-шим numpy/cupy + scatter/ckpt хелперы | numpy-путь [измерено] |
| `make_fork.py` | генератор `nano_lc_kg.py` из `../nano_lc.py` (20 exact-патчей, падает при несовпадении) | [измерено] |
| `nano_lc_kg.py` | форк тренера: +`--negrng`, +`--trunknorm`, backend-шим | **битовый паритет с оригиналом** [измерено] |
| `train_torch.py` | torch-движок GPU (семантика nano_lc 1:1; sm75/60: NS5 в fp32) | [не валидировано локально; gate в ноутбуке] |
| `bracket.py` | ASHA-движок турнира: рунги, CTRL, страховка, DQ, resume | QUICK end-to-end [измерено] |
| `build_nb.py` | пересборка ноутбука из файлов | [измерено] |
| `dataset_prep/` | train.npy/val.npy/meta.json для загрузки датасетом на Kaggle | — |

## Как пользоваться (коротко)
1. На Kaggle: New Dataset → загрузить содержимое `dataset_prep/` (3 файла). Или ничего — ноутбук скачает с raw.githubusercontent (Internet ON).
2. Импортировать `AIra_TOURNAMENT.ipynb` (New Notebook → Import). Accelerator: **GPU T4 x2** (предпочтительно) или P100. Internet: ON (для фолбэка скачивания; torch предустановлен в образе Kaggle).
3. Save & Run All. Сначала GPU-гейт честности (40 шагов CTRL torch-GPU vs numpy-CPU, критерий |ΔPPL|≤2%), дальше брекет сам выберет движок.
4. По завершении/таймауту — Save Version. Продолжение: Add Data → Notebook Output Files прошлой версии → Run All (state.json подхватится автоматически, зависшие джобы вернутся в pending).
5. Из Output забрать `airaw/bracket_work/SUMMARY.md` (+ state.json, results/) и принести в проект.

## Методология турнира (зафиксирована в PREDICTIONS.md внутри брекета)
- Пространство: 24 конфига = CTRL + 23 латин-гиперкуба по {mulr∈[0.010,0.040] log, ssfull∈[0,0.30], ssk∈[256,768] log, ssalpha∈[0.5,1.0], lr∈[4e-4,9e-4] log}.
- Рунги 60/200/500/1500 шагов, квоты 12/5/3/финал(×4 сида), CTRL не выбывает, страховочный слот лучшего наклона.
- Авто-DQ: `wn<3` после шага 100 («мёртвый прогон»), NaN.
- Оценка: 2.5ч CPU-фолбэк / <1ч T4×2 [проекция]. Квота Kaggle ~30 GPU-ч/нед, T4×2 — следить за расходом.

## Пересборка после правок
```
cd leancore/kaggle
python3 make_fork.py    # после любых изменений ../nano_lc.py (упадёт, если патч не совпал)
python3 build_nb.py     # пересобрать ipynb
```

## Если гейт ушёл в numpy при включённом GPU
Симптом: CPU 300%, GPU 0% — это 3 CPU-воркера фолбэка, а GPU-квота при этом горит (считается по приаттаченному Accelerator, не по загрузке). Действия: (1) посмотри вывод ячейки гейта — v2 печатает КРАШ/stderr или relΔ; (2) пришли вывод в чат; (3) либо жди фикс, либо Settings → Accelerator → None и Run All — брекет честно поедет на CPU без траты GPU-квоты. Форс: `%env LC_FORCE_ENGINE=torch` перед ячейкой гейта (только по моему слову — после вердикта о природе расхождения).
