#!/usr/bin/env python3
"""make_fork.py — генерирует kaggle/nano_lc_kg.py из ../nano_lc.py точечными заменами.
Каждая замена ОБЯЗАНА матчиться ровно один раз — иначе выход громко падает.
Инвариант: при LC_BACKEND=numpy и --negrng 0 --trunknorm 0 форк битуально воспроизводит
оригинал (те же вызовы RNG в том же порядке, те же функции numpy).
Добавлено форком: --negrng (отдельный поток негативов), --trunknorm (лог ‖W‖/‖W₀‖ в jsonl),
backend-шим lcxp (cupy ГПУ-путь на Kaggle), хост-сохранение npz.
"""
import pathlib, re, sys

SRC = pathlib.Path(__file__).resolve().parent.parent / "nano_lc.py"
DST = pathlib.Path(__file__).resolve().parent / "nano_lc_kg.py"

code = SRC.read_text(encoding="utf-8")

PATCHES = [
    # 1. импорт numpy → шим
    (
        "import numpy as np",
        "import numpy as hnp\nfrom lcxp import xp, to_dev, asnumpy, on_gpu, save_npz, init_norms, trunk_ratio, scatter_rows\nnp = xp  # backend (numpy на CPU, cupy на GPU); host-операции — через hnp",
    ),
    # 2. ctypes-ядра — только CPU
    (
        "try:\n    _KS = _ct.CDLL(",
        "try:\n    if on_gpu: raise OSError('gpu backend: ctypes-ядра выключены')\n    _KS = _ct.CDLL(",
    ),
    # 3. scatter_add_rows → бэкенд-делегат (numpy-ветка битуально та же)
    (
        '''def scatter_add_rows(dst, ids, src):
    """dst[ids] += src с ДУБЛИКАТАМИ в ids (векторно: сорт + reduceat; замена медленному np.add.at).
    ids (N,) int, src (N,D) — суммируется по одинаковым id и пишется одной fancy-записью."""
    ids = ids.reshape(-1)
    src2 = src.reshape(ids.shape[0], -1)
    order = np.argsort(ids, kind="stable")
    sids = ids[order]
    bounds = np.flatnonzero(np.r_[True, sids[1:] != sids[:-1]])
    dst[sids[bounds]] += np.add.reduceat(src2[order], bounds)''',
        '''def scatter_add_rows(dst, ids, src):
    """Бэкенд-делегат (numpy: сорт+reduceat как в оригинале; cupy: атомарный scatter_add)."""
    scatter_rows(dst, ids.reshape(-1), src)''',
    ),
    # 4. детерминированный инит — host-RNG + to_dev
    (
        "        rng = np.random.default_rng(zlib.crc32(name.encode()) & 0xFFFFFFFF)  # детерминизм: hash() солится за процесс",
        "        rng = hnp.random.default_rng(zlib.crc32(name.encode()) & 0xFFFFFFFF)  # детерминизм: hash() солится за процесс",
    ),
    (
        "        self.d[name] = (rng.normal(0, std, shape).astype(f32) if init is None else init(shape))",
        "        self.d[name] = to_dev(rng.normal(0, std, shape).astype(f32) if init is None else init(shape))",
    ),
    (
        "                        return np.random.default_rng(1).normal(",
        "                        return hnp.random.default_rng(1).normal(",
    ),
    # 5. данные — на host
    (
        '    tr = np.load(f"{ROOT}/{args.data}/train.npy").astype(np.int64)',
        '    tr = hnp.load(f"{ROOT}/{args.data}/train.npy").astype(hnp.int64)',
    ),
    (
        '    va = np.load(f"{ROOT}/{args.data}/val.npy").astype(np.int64)',
        '    va = hnp.load(f"{ROOT}/{args.data}/val.npy").astype(hnp.int64)',
    ),
    # 6. основной RNG — host + negrng-поток
    (
        "    rng = np.random.default_rng(args.seed)",
        "    rng = hnp.random.default_rng(args.seed)\n    rng_neg = hnp.random.default_rng(args.seed * 1000003 + 17) if getattr(args, 'negrng', 0) else rng",
    ),
    # 7. initckpt / distill — host-load
    (
        "        td = np.load(args.distill)",
        "        td = hnp.load(args.distill)",
    ),
    (
        "        d0 = np.load(args.initckpt)",
        "        d0 = hnp.load(args.initckpt)",
    ),
    # 8. unigram-статистика — host
    (
        "        cnt = np.bincount(tr, minlength=V).astype(np.float64) + 1.0",
        "        cnt = hnp.bincount(tr, minlength=V).astype(hnp.float64) + 1.0",
    ),
    # 9. batch — host-сборка, device-выдача
    (
        "        x = np.stack([ids[s:s + args.ctx] for s in st]); y = np.stack([ids[s + 1:s + args.ctx + 1] for s in st])\n        return x, y",
        "        x = hnp.stack([ids[s:s + args.ctx] for s in st]); y = hnp.stack([ids[s + 1:s + args.ctx + 1] for s in st])\n        return to_dev(x), to_dev(y)",
    ),
    # 10. vloss-RNG — host
    (
        "        rr = np.random.default_rng(1234); tot = 0.0",
        "        rr = hnp.random.default_rng(1234); tot = 0.0",
    ),
    # 11. кандидаты/neg — host-сборка + device-копии; negrng-поток
    (
        """            neg = rng.choice(V, size=args.ssk, replace=True, p=ssq)
            cand = np.union1d(np.unique(y), neg).astype(np.int64)
            logQ = np.log1p(-np.power(1.0 - ssq[cand], args.ssk))      # log(1−(1−q)^K)
            logQ[np.isin(cand, y)] = 0.0                               # цели всегда в наборе
            Ec0 = (model.p.d["U"][cand] @ model.p.d["P"]) if model.erank > 0 else None
            loss, (dz, cand, Ec) = sampled_ce(H, y, model.p.d["E"] if model.erank == 0 else Ec0, cand, logQ, Ec=Ec0)
            head_cache = (cand, dz, Ec)""",
        """            neg = rng_neg.choice(V, size=args.ssk, replace=True, p=ssq)
            yh = asnumpy(y)
            cand = hnp.union1d(hnp.unique(yh), neg).astype(hnp.int64)
            logQ = hnp.log1p(-hnp.power(1.0 - ssq[cand], args.ssk))    # log(1−(1−q)^K)
            logQ[hnp.isin(cand, yh)] = 0.0                             # цели всегда в наборе
            cand_d, logQ_d = to_dev(cand), to_dev(logQ)
            Ec0 = (model.p.d["U"][cand_d] @ model.p.d["P"]) if model.erank > 0 else None
            loss, (dz, cand, Ec) = sampled_ce(H, y, model.p.d["E"] if model.erank == 0 else Ec0, cand_d, logQ_d, Ec=Ec0)
            head_cache = (cand, dz, Ec)""",
    ),
    # 12. trunk-norm: исходные нормы после построения модели
    (
        '    print(f"[{args.tag}] nano-{args.kind} params={nparams:,}", flush=True)',
        '    print(f"[{args.tag}] nano-{args.kind} params={nparams:,} backend={' + "'cupy-gpu'" + ' if on_gpu else ' + "'numpy'" + '}", flush=True)\n    N0_TRUNK = init_norms(model.p.d) if getattr(args, "trunknorm", 0) else None',
    ),
    # 13. trunk-norm поле в jsonl
    (
        "            print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + \"\\n\"); logf.flush()",
        "            if N0_TRUNK is not None: rec[\"wn\"] = round(trunk_ratio(model.p.d, N0_TRUNK), 3)\n            print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + \"\\n\"); logf.flush()",
    ),
    # 14. сохранение чекпоинтов — host
    (
        '    np.savez(f"{ROOT}/results/ckpt_{args.tag}.npz", **model.p.d)',
        '    save_npz(f"{ROOT}/results/ckpt_{args.tag}.npz", model.p.d)',
    ),
    (
        '        np.savez(f"{ROOT}/results/ckpt_{args.tag}_ema.npz", **{k: ew[k] / ew_corr for k in ew})',
        '        save_npz(f"{ROOT}/results/ckpt_{args.tag}_ema.npz", {k: ew[k] / ew_corr for k in ew})',
    ),
    # 15. новые флаги
    (
        '    ap.add_argument("--ssalpha", type=float, default=1.0, help="степень сглаживания unigram-предложения (word2vec 0.75)")',
        '''    ap.add_argument("--ssalpha", type=float, default=1.0, help="степень сглаживания unigram-предложения (word2vec 0.75)")
    ap.add_argument("--negrng", type=int, default=0, help="1 = отдельный rng-поток для негативов (парность порядка данных между конфигами)")
    ap.add_argument("--trunknorm", type=int, default=0, help="1 = лог ‖W_trunk‖/‖W₀‖ в jsonl (авто-DQ мёртвых прогонов)")''',
    ),
]

n_fail = 0
for i, (old, new) in enumerate(PATCHES, 1):
    cnt = code.count(old)
    if cnt != 1:
        print(f"[make_fork] ПАТЧ {i}: совпадений {cnt} (ожидал 1)\n---- искали ----\n{old[:200]}", file=sys.stderr)
        n_fail += 1
        continue
    code = code.replace(old, new, 1)

if n_fail:
    sys.exit(f"[make_fork] {n_fail} патчей не применились — форк НЕ собран")

DST.write_text(code, encoding="utf-8")
print(f"[make_fork] ok: {DST} ({len(code)} байт, {len(PATCHES)} патчей)")
