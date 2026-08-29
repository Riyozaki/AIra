#!/usr/bin/env python3
"""lc_shelf — триггерная полка L0 (salvage из заброшенных AIra-brain/miniLLM) для word-level LeanCore.

Принцип T: цена ответа ∝ новизне. Полка: контекст (k последних токенов) → счётчики продолжений.
Отличия от провалившегося byte/BPE-моста miniLLM: наш токен = слово, полка и модель В ОДНОМ базисе,
а состояние EMA-потока продвигается входами (не выходами) — обход даёт точную алгебру состояния.

Протокол честности (по аудиту miniLLM): полка строится по train; θ калибруется на val[:half];
рынок и гибрид меряются на val[half:]. p_shelf — НОРМИРОВАННАЯ распределительная (add-α по всему V),
псевдо-PPL из «ценности уверенности» запрещена.

  python3 lc_shelf.py market            # сетка покрытие×точность на отложенной половине val
  python3 lc_shelf.py hybrid <ckpt>     # гибрид L0+EMA-поток: proper hybrid PPL + tok/s
"""
import os, sys, json, math, time
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__))


def wilson_lb(p, n, z=1.96):
    if n == 0: return 0.0
    return (p + z*z/(2*n) - z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)


class Shelf:
    """контекст-префикс (tuple, длина ≤ kmax) → Counter(продолжение). Хранится разреженно."""

    def __init__(self, kmax=3):
        self.kmax = kmax
        self.tab = {}      # tuple(len к) → np.array dict tok → (best, cnt_best, tot)

    def fit(self, ids):
        k = self.kmax
        # построение одним проходом по всем длинам сразу: dict[prefix] = {}
        from collections import defaultdict
        raw = defaultdict(lambda: defaultdict(int))
        ids = np.asarray(ids)
        for L in range(1, k + 1):
            nxt = ids[L:]
            if L == 1:
                for c, t in zip(ids[:-1], nxt):
                    raw[(int(c),)][int(t)] += 1
            else:
                key_gen = zip(*(ids[L - j - 1:len(ids) - j - 1] for j in range(L)))
                for tup, t in zip(key_gen, nxt):
                    raw[tuple(int(x) for x in tup)][int(t)] += 1
        for key, cnts in raw.items():
            tot = sum(cnts.values())
            best, cb = max(cnts.items(), key=lambda kv: kv[1])
            self.tab[key] = (best, cb, tot)
        return len(self.tab)

    def query(self, ctx_ids):
        """→ (best_tok, wilson_lb_acc, tot, counts_dict) по самому длинному совпавшему префиксу."""
        for L in range(min(self.kmax, len(ctx_ids)), 0, -1):
            got = self.tab.get(tuple(int(x) for x in ctx_ids[-L:]))
            if got:
                best, cb, tot = got
                return best, wilson_lb(cb / tot, tot), tot
        return None, 0.0, 0


def main_market():
    tr = np.load(f"{ROOT}/data/prep/train.npy").astype(np.int64)
    va = np.load(f"{ROOT}/data/prep/val.npy").astype(np.int64)
    sh = Shelf(kmax=3)
    nctx = sh.fit(tr)
    print(f"полка: {nctx:,} контекстов (k≤3) из {len(tr):,} токенов train", flush=True)
    half = len(va) // 2
    cal, test = va[:half], va[half:]
    # θ калибруем на cal, докладываем test
    grid = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    def scan(ids):
        cov = {g: 0 for g in grid}; hit = {g: 0 for g in grid}
        hist = [2]  # стартовый контекст
        for i in range(1, len(ids)):
            best, lb, tot = sh.query(ids[max(0, i - 3):i])
            if best is not None:
                for g in grid:
                    if lb >= g:
                        cov[g] += 1; hit[g] += int(best == ids[i])
        return [(g, cov[g] / max(1, len(ids) - 1), hit[g] / max(1, cov[g])) for g in grid]

    print("θ_Wilson  покрытие@cal  точность@cal | покрытие@test  точность@test")
    rc, rt = scan(cal), scan(test)
    for (g, cc, hc), (_, ct, ht) in zip(rc, rt):
        print(f"  {g:4.2f}    {cc:8.2%}    {hc:8.3%} |    {ct:8.2%}    {ht:8.3%}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "market":
        main_market()
    else:
        print(__doc__)
