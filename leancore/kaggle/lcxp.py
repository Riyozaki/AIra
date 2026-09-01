"""lcxp — backend-шим для nano_lc: numpy (CPU) | cupy (GPU, Kaggle).
Управление: env LC_BACKEND = numpy | cupy  (по умолчанию numpy).
Принцип: тензорная математика → xp; данные/семплирование/индексы-кандидаты → hnp (host numpy).
Это гарантирует ОДИНАКОВЫЙ порядок данных и негативов на CPU и GPU (парность брекета).
"""
import os
import numpy as hnp

BACKEND = os.environ.get("LC_BACKEND", "numpy").strip().lower()
if BACKEND == "cupy":
    try:
        import cupy as xp
        import cupyx
        on_gpu = True
    except Exception as e:                                # noqa
        print(f"[lcxp] cupy недоступен ({e}); фолбэк на numpy", flush=True)
        BACKEND = "numpy"
        import numpy as xp
        cupyx = None
        on_gpu = False
else:
    import numpy as xp
    cupyx = None
    on_gpu = False
    BACKEND = "numpy"

f32 = xp.float32


def to_dev(a):
    return xp.asarray(a)


def asnumpy(a):
    if on_gpu:
        return xp.asnumpy(a)
    return hnp.asarray(a)


def scatter_rows(dst, ids, src):
    """dst[ids] += src с ДУБЛИКАТАМИ в ids.
    numpy-ветка — битуально та же reduceat-схема, что в оригинале (паритет форка).
    cupy-ветка — атомарный scatter_add (результат эквивалентен по сумме, порядок сложения иной)."""
    if on_gpu:
        src2 = src.reshape(ids.shape[0], -1)
        dst2 = dst.reshape(-1, src2.shape[-1])
        cupyx.scatter_add(dst2, ids, src2)
        return
    src2 = src.reshape(ids.shape[0], -1)
    order = hnp.argsort(ids, kind="stable")
    sids = ids[order]
    bounds = hnp.flatnonzero(hnp.r_[True, sids[1:] != sids[:-1]])
    dst[sids[bounds]] += hnp.add.reduceat(src2[order], bounds)


def save_npz(path, d):
    """Хост-сохранение чекпоинта (с device-конверсией)."""
    hnp.savez(path, **{k: asnumpy(v) for k, v in d.items()})


TRUNK_EXCLUDE = ("E", "U", "pos")


def trunk_keys(d):
    return [k for k, v in d.items() if v.ndim == 2 and k not in TRUNK_EXCLUDE and min(v.shape) >= 16]


def init_norms(d):
    """Нормы транка после инициализации (host-копии, дёшево)."""
    ks = trunk_keys(d)
    tot = 0.0
    for k in ks:
        v = asnumpy(d[k]).astype(hnp.float64)
        tot += float((v * v).sum())
    return tot ** 0.5


def trunk_ratio(d, n0):
    """‖W_trunk‖ / ‖W_trunk_init‖: авто-дисквалификация «мёртвых» прогонов (<3 к шагу 100)."""
    return init_norms(d) / max(n0, 1e-12)


def is_contig(a):
    """cupy-ndarray имеет .flags только с ключами c_contiguous/f_contiguous/owndata —
    обращения вида flags['C_CONTIGUOUS'] на GPU дают KeyError (нашли на Kaggle T4)."""
    if on_gpu:
        try:
            return bool(a.flags['c_contiguous'])
        except Exception:
            return True                      # всё, что приходит с matmul/elementwise — C-contiguous
    return bool(a.flags['C_CONTIGUOUS'])


def gpu_info():
    if not on_gpu:
        return "numpy"
    try:
        d = xp.cuda.runtime.getDeviceProperties(xp.cuda.Device().id)
        nm = d['name']
        return f"cupy:{nm.decode() if isinstance(nm, bytes) else nm}"
    except Exception:
        return "cupy"
