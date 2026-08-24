# lab/ — численный стенд верификации THEORY_V3 (T-неделя)

Протокол и замороженные предсказания: `docs/TWEEK.md`. Результаты: `docs/TWEEK_RESULTS.md`.

## Запуск (чистый CPU, HW-0)
```bash
pip install torch numpy          # единственные зависимости
python3 lab/exp_lambda.py --smoke           # проверка (~1 мин)
python3 lab/exp_lambda.py                   # ~20-30 мин: init-sweep + 2 обученных прогона
python3 lab/exp_msweep.py                   # ~30-40 мин: m в {1,2,4,8,16} + baseline
python3 lab/public_fit.py                   # секунды: фиты публичных кривых
```

## Файлы
- `models.py` — TinyLoopLM: общий слой, K итераций, depth-emb по флагу, двухмасштабное ядро (fast+m/m-slow, FiLM).
- `data_synth.py` — двухмасштабный марковский источник, известная правда τ=10, шум-пол 5%.
- `telemetry.py` — три независимых оценителя λ̂ (дисплейсмент, возмущение, спектральный радиус Якобиана через JVP) + фиты «геометрия+пол» и степенной закон, AICc.
- `train.py` — петля обучения с глубоким надзором на всех итерациях.
- `exp_lambda.py` → `results/tweek/lambda_results.json` (TW-0, TW-1, TW-1b, TW-1c, TW-2).
- `exp_msweep.py` → `results/tweek/msweep_results.json` (TW-3, TW-3b).
- `public_fit.py` → `results/tweek/public_fits.json` (TW-4).
- `public_curves/` — точки из публичных статей; у каждой точки происхождение (таблица/рисунок/arXiv-id).

## Ограничения по замыслу
Никаких своих фреймворков: голый torch. Никакого CUDA-кода: всё должно жить на 2 CPU / 3 ГБ.
Toy-шкала проверяет форму законов и согласованность приборов, а не абсолютные числа AS-1 —
абсолютные числа измеряются на T0/T1 позже (TW-H в `docs/TWEEK.md` §1).
