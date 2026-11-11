"""Измерение реального roofline текущей машины (фаза 1).
Измеряет: пиковую FP32/FP64 производительность GEMM (OpenBLAS через numpy),
пропускную способность памяти (copy / triad на массивах >> L3),
и GEMV (bandwidth-bound случай, аналог декодирования LLM).
Все числа — ИЗМЕРЕННЫЕ, медиана из N повторов.
"""
import time, json, os, sys
import numpy as np

def bench(fn, reps=7, warm=2):
    for _ in range(warm): fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts)//2], ts[0]

res = {"threads_env": os.environ.get("OPENBLAS_NUM_THREADS", "default"), "nproc": os.cpu_count()}

# 1) GEMM compute peak
for dt in (np.float32, np.float64):
    n = 2048
    a = np.random.rand(n, n).astype(dt); b = np.random.rand(n, n).astype(dt); c = np.empty((n, n), dt)
    med, best = bench(lambda: np.matmul(a, b, out=c), reps=5)
    res[f"gemm_{np.dtype(dt).name}_GFLOPs_median"] = 2*n**3/med/1e9
    res[f"gemm_{np.dtype(dt).name}_GFLOPs_best"] = 2*n**3/best/1e9

# 2) Memory bandwidth: arrays of 256 MB each (>> 54 MB L3)
N = 32 * 1024 * 1024  # 32M float64 = 256 MB
x = np.ones(N); y = np.empty(N); z = np.ones(N)
med, best = bench(lambda: np.copyto(y, x), reps=7)
res["copy_GBps_median"] = 2*8*N/med/1e9   # read + write
res["copy_GBps_best"] = 2*8*N/best/1e9
med, best = bench(lambda: np.add(x, z, out=y), reps=7)
res["add_GBps_median"] = 3*8*N/med/1e9    # 2 reads + 1 write
med, best = bench(lambda: x.sum(), reps=7)
res["read_sum_GBps_median"] = 8*N/med/1e9

# 3) GEMV (decode-like): W (8192x8192 fp32 = 256MB) @ v
W = np.random.rand(8192, 8192).astype(np.float32); v = np.random.rand(8192).astype(np.float32)
med, best = bench(lambda: W @ v, reps=7)
res["gemv_fp32_GBps_median"] = W.nbytes/med/1e9
res["gemv_fp32_GFLOPs_median"] = 2*W.size/med/1e9

# 4) Small GEMV that fits in L2/L3 (1024x1024 fp32 = 4 MB)
Ws = np.random.rand(1024, 1024).astype(np.float32); vs = np.random.rand(1024).astype(np.float32)
med, best = bench(lambda: Ws @ vs, reps=50)
res["gemv_small_cached_GBps_median"] = Ws.nbytes/med/1e9
res["gemv_small_cached_GFLOPs_median"] = 2*Ws.size/med/1e9

print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()}, indent=1))
