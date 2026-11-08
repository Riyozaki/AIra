#!/usr/bin/env python3
"""capacity.py -- ёмкость памяти фиксированного размера (задача 2.2 из docs/01).

Считается без градиентов и без обучения: только линейная алгебра и правила
записи. Это позволяет получить содержательные числа на 2 ядрах и 3 ГБ RAM.

Сравниваются четыре способа хранить k пар (ключ -> значение) в d-мерном
пространстве:

  hebbian   S = sum_i v_i k_i^T                 (линейное внимание без гейта)
  delta     W <- W + (v_i - W k_i) k_i^T        (delta rule, один проход)
  lstsq     W = V^T K (K^T K + eps I)^-1        (оптимум для любой линейной памяти)
  kv        точное хранение всех пар            (то, что делает KV-кэш)

Метрика: косинус между восстановленным и истинным значением.

Ключевые вопросы:
  1. при каком lambda = k/d память разрушается;
  2. растёт ли ёмкость линейно по d (а не по длине контекста);
  3. сколько бит состояния нужно, чтобы не терять ёмкость;
  4. при какой длине контекста фиксированное состояние дешевле KV по байтам.
"""

from __future__ import annotations

import csv
import numpy as np

EPS = 1e-6


# ---------------------------------------------------------------------------
# Памяти
# ---------------------------------------------------------------------------

def store_hebbian(K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Один проход без декорации: S = sum_i v_i k_i^T."""
    return V.T @ K


def store_delta(K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Последовательное правило delta за один проход (страдает от недавности)."""
    W = np.zeros((V.shape[1], K.shape[1]))
    for i in range(K.shape[0]):
        W += np.outer(V[i] - W @ K[i], K[i])
    return W


def store_lstsq(K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Оптимальная линейная память: минимум ||W K - V||. Верхняя граница
    для любой архитектуры, которая сжимает пары в матрицу d x d."""
    G = K.T @ K + EPS * np.eye(K.shape[1])
    return np.linalg.solve(G, K.T @ V).T


def retrieve(S: np.ndarray, K: np.ndarray) -> np.ndarray:
    """y_j = S k_j  <->  Y = K S^T."""
    return K @ S.T


def quantize(W: np.ndarray, bits: int) -> np.ndarray:
    """Симметричная пер-тензорная квантизация состояния."""
    if bits >= 32:
        return W
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(W).max() / qmax
    return W if scale == 0 else np.round(W / scale) * scale


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------

def mean_cosine(Y_hat: np.ndarray, V: np.ndarray) -> float:
    num = np.sum(Y_hat * V, axis=1)
    den = np.linalg.norm(Y_hat, axis=1) * np.linalg.norm(V, axis=1) + 1e-12
    return float(np.mean(num / den))


def make_pairs(k: int, d: int, rng: np.random.Generator, rank: int | None = None):
    """rank=None -- случайные ключи; rank=r -- ключи лежат в r-мерном подпространстве
    (имитация коррелированных ключей, как в естественном языке)."""
    if rank is None:
        K = rng.standard_normal((k, d))
    else:
        basis = rng.standard_normal((rank, d))
        K = rng.standard_normal((k, rank)) @ basis
    K /= np.linalg.norm(K, axis=1, keepdims=True)
    V = rng.standard_normal((k, d))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return K, V


STORES = {"hebbian": store_hebbian, "delta": store_delta, "lstsq": store_lstsq}


# ---------------------------------------------------------------------------
# Эксперименты
# ---------------------------------------------------------------------------

def load_sweep(d: int = 128, seeds: int = 3,
               lambdas=(0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)):
    """Качество против загрузки lambda = k/d. Среднее по seeds."""
    rows = []
    for lam in lambdas:
        k = max(1, int(round(lam * d)))
        acc = {name: [] for name in STORES}
        for s in range(seeds):
            rng = np.random.default_rng(1000 * s + d)
            K, V = make_pairs(k, d, rng)
            for name, fn in STORES.items():
                acc[name].append(mean_cosine(retrieve(fn(K, V), K), V))
        rows.append({"d": d, "lambda": lam, "k": k,
                     **{f"cos_{n}": float(np.mean(v)) for n, v in acc.items()}})
    return rows


def capacity_threshold(d: int, target: float = 0.9, seeds: int = 3) -> dict:
    """Максимальная загрузка, при которой cos >= target."""
    best = {name: 0.0 for name in STORES}
    data = load_sweep(d=d, seeds=seeds)
    for r in data:
        for name in STORES:
            if r[f"cos_{name}"] >= target:
                best[name] = max(best[name], r["lambda"])
    return best


def bit_sweep(d: int = 128, lam: float = 0.25, seeds: int = 3,
              bits=(2, 3, 4, 8, 16)):
    rows = []
    for b in bits:
        acc = []
        k = max(1, int(round(lam * d)))
        for s in range(seeds):
            rng = np.random.default_rng(2000 * s + d)
            K, V = make_pairs(k, d, rng)
            acc.append(mean_cosine(retrieve(quantize(store_lstsq(K, V), b), K), V))
        rows.append({"d": d, "lambda": lam, "bits": b, "cos": float(np.mean(acc))})
    return rows


def rank_sweep(d: int = 128, seeds: int = 3, target: float = 0.9):
    """Как ёмкость зависит от ранга распределения ключей."""
    out = []
    for rank in (4, 8, 16, 32, 64, None):
        best = 0.0
        for lam in (0.0625, 0.125, 0.25, 0.5, 1.0):
            k = max(1, int(round(lam * d)))
            acc = []
            for s in range(seeds):
                rng = np.random.default_rng(3000 * s + d + (rank or 0))
                K, V = make_pairs(k, d, rng, rank=rank)
                acc.append(mean_cosine(retrieve(store_lstsq(K, V), K), V))
            if float(np.mean(acc)) >= target:
                best = lam
        out.append({"d": d, "key_rank": rank if rank else d, "max_lambda": best,
                    "k": best * d})
    return out


def byte_analysis(d: int = 128, bits_state: int = 8, bits_kv: int = 8) -> dict:
    kv_per_pair = 2 * d * bits_kv / 8.0        # ключ + значение
    state_total = d * d * bits_state / 8.0     # вся матрица
    return {"kv_per_pair": kv_per_pair, "state_total": state_total,
            "break_even_pairs": state_total / kv_per_pair}


# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 78)
    print("1. Загрузка памяти, lambda = k/d  (d=128, среднее по 3 seeds)")
    print("=" * 78)
    rows = load_sweep()
    print(f"{'lambda':>7s} {'k':>5s} {'hebbian':>9s} {'delta':>8s} {'lstsq':>8s}")
    for r in rows:
        print(f"{r['lambda']:7.4f} {r['k']:5d} {r['cos_hebbian']:9.3f} "
              f"{r['cos_delta']:8.3f} {r['cos_lstsq']:8.3f}")

    print()
    print("=" * 78)
    print("2. Ёмкость при cos >= 0.9: линейна ли она по d?")
    print("=" * 78)
    print(f"{'d':>5s} {'max lam hebb':>13s} {'max lam delta':>14s} "
          f"{'max lam lstsq':>14s} {'k(lstsq)':>9s}")
    cap_rows = []
    for d in (32, 64, 128, 256):
        c = capacity_threshold(d)
        cap_rows.append({"d": d, **c, "k_lstsq": c["lstsq"] * d})
        print(f"{d:5d} {c['hebbian']:13.3f} {c['delta']:14.3f} "
              f"{c['lstsq']:14.3f} {c['lstsq']*d:9.0f}")
    ks = [r["k_lstsq"] for r in cap_rows]
    ds = [r["d"] for r in cap_rows]
    slope = np.polyfit(np.log(ds), np.log(ks), 1)[0]
    print(f"\n  наклон log(k) от log(d): {slope:.2f}  "
          f"(1.0 = ёмкость строго линейна по размеру состояния)")

    print()
    print("=" * 78)
    print("3. Битность состояния против качества (d=128, lstsq)")
    print("=" * 78)
    for lam in (0.25, 1.0):
        print(f"  lambda={lam:g}:")
        for r in bit_sweep(lam=lam):
            print(f"    {r['bits']:2d} бит -> cos {r['cos']:.3f}")

    print()
    print("=" * 78)
    print("3b. Ёмкость против ранга ключей (d=128, lstsq, cos>=0.9)")
    print("=" * 78)
    print(f"{'ранг ключей':>12s} {'max lambda':>11s} {'k':>6s}")
    for r in rank_sweep():
        print(f"{r['key_rank']:12d} {r['max_lambda']:11.4f} {r['k']:6.0f}")

    print()
    print("=" * 78)
    print("4. Байты: точный KV против сжатого состояния (d=128)")
    print("=" * 78)
    b = byte_analysis(128)
    print(f"  KV:    {b['kv_per_pair']:.0f} байт на пару, растёт с контекстом")
    print(f"  state: {b['state_total']:.0f} байт целиком, фиксировано")
    print(f"  точка окупаемости: {b['break_even_pairs']:.0f} пар")

    with open("runs/capacity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open("runs/capacity_scaling.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cap_rows[0].keys()))
        w.writeheader()
        w.writerows(cap_rows)
    rk = rank_sweep()
    with open("runs/capacity_rank.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rk[0].keys()))
        w.writeheader()
        w.writerows(rk)
    print("\nruns/capacity.csv, runs/capacity_scaling.csv, runs/capacity_rank.csv")


if __name__ == "__main__":
    main()
