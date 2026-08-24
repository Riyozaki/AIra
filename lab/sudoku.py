"""Sudoku (4x4 / 9x9) with ground-truth solution depth, for the G0S stand.

Design (TWEEK_RESULTS R4): fan-in per cell >> lookup capacity, so loops are
bought by the task itself. 9x9 puzzles are produced by DIGGING from a full
solution with uniqueness re-checked after every removal, and accepted only if
solvable by pure constraint propagation (naked + hidden singles, synchronous
waves). Ground-truth depth d* = number of synchronous relaxation waves:
one wave = one parallel propagation pass = one loop iteration of the model.
"""
import numpy as np


class SudokuConfig:
    def __init__(self, box: int = 3):
        self.box = box
        self.N = box * box
        self.S = self.N * self.N
        N, B, S = self.N, self.box, self.S
        units = []
        units += [[r * N + c for c in range(N)] for r in range(N)]
        units += [[r * N + c for r in range(N)] for c in range(N)]
        units += [[(br + i) * N + (bc + j) for i in range(B) for j in range(B)]
                  for br in range(0, N, B) for bc in range(0, N, B)]
        self.units = units
        self.peers = {i: {j for u in units if i in u for j in u if j != i}
                      for i in range(S)}
        # canonical pattern solution
        self.base = np.array([(B * (r % B) + r // B + c) % N + 1
                              for r in range(N) for c in range(N)], dtype=np.int8)

    def shuffled_solution(self, rng):
        N, B = self.N, self.box
        sol = self.base.reshape(N, N)
        lut = np.zeros(N + 1, dtype=np.int8)
        lut[1:] = rng.permutation(N) + 1
        sol = lut[sol]
        n_bands = N // B
        rows = np.concatenate([b * B + rng.permutation(B)
                               for b in rng.permutation(n_bands)])
        sol = sol[rows, :]
        cols = np.concatenate([s * B + rng.permutation(B)
                               for s in rng.permutation(n_bands)])
        sol = sol[:, cols]
        if rng.random() < 0.5:
            sol = sol.T
        return sol.reshape(-1).astype(np.int8)

    def count_solutions(self, grid, cap=2):
        cands = [set(range(1, self.N + 1)) if v == 0 else {int(v)} for v in grid]
        for i in range(self.S):
            if grid[i] != 0:
                for p in self.peers[i]:
                    cands[p].discard(int(grid[i]))
        if any(len(c) == 0 for c in cands):
            return 0
        count = 0

        def bt(cands):
            nonlocal count
            if count >= cap:
                return
            i = min((i for i in range(self.S) if len(cands[i]) > 1),
                    key=lambda i: len(cands[i]), default=None)
            if i is None:
                count += 1
                return
            for v in sorted(cands[i]):
                nc = [set(c) for c in cands]
                nc[i] = {v}
                bad = False
                for p in self.peers[i]:
                    nc[p] = nc[p] - {v}
                    if not nc[p]:
                        bad = True
                        break
                if not bad:
                    bt(nc)

        bt(cands)
        return count

    def propagation_depth(self, grid):
        """Synchronous waves to solve, None if guessing required."""
        solved = {}
        clues = {i: int(v) for i, v in enumerate(grid) if v != 0}
        waves = 0
        while True:
            cands = []
            for i in range(self.S):
                if i in solved or i in clues:
                    cands.append({clues.get(i, solved.get(i, 0))})
                    continue
                c = set(range(1, self.N + 1))
                for p in self.peers[i]:
                    if p in clues:
                        c.discard(clues[p])
                    elif p in solved:
                        c.discard(solved[p])
                if not c:
                    return None
                cands.append(c)
            if len(solved) + len(clues) == self.S:
                return waves
            new = {}
            for i in range(self.S):
                if i not in solved and i not in clues and len(cands[i]) == 1:
                    new[i] = next(iter(cands[i]))
            for u in self.units:
                for v in range(1, self.N + 1):
                    places = [i for i in u
                              if i not in solved and i not in clues and v in cands[i]]
                    if len(places) == 1:
                        new.setdefault(places[0], v)
            if not new:
                return None
            solved.update(new)
            waves += 1

    def dig_puzzle(self, rng, max_remove=None):
        """Dig cells from a full solution keeping uniqueness. Returns puzzle or None."""
        sol = self.shuffled_solution(rng)
        grid = sol.copy()
        max_remove = max_remove or (self.S * 4 // 5)
        remove_order = rng.permutation(self.S)
        removed = 0
        for c in remove_order:
            if removed >= max_remove:
                break
            v = grid[c]
            grid[c] = 0
            if self.count_solutions(grid) != 1:
                grid[c] = v
            else:
                removed += 1
        d = self.propagation_depth(grid)
        if d is None or d < 1:
            return None
        return {"grid": grid.astype(np.int8), "sol": sol, "depth": d}


def build_pool(cfg: SudokuConfig, n_total: int, seed: int,
               quotas: dict | None = None, attempts_cap: int = 40_000,
               verbose=True):
    rng = np.random.default_rng(seed)
    quotas = quotas or {}
    pool, seen, attempts = [], set(), 0
    counts: dict[int, int] = {}

    def quotas_met():
        return bool(quotas) and all(counts.get(d, 0) >= q for d, q in quotas.items())

    while len(pool) < n_total and attempts < attempts_cap and not quotas_met():
        attempts += 1
        if cfg.box == 3:
            it = cfg.dig_puzzle(rng)
            if it is None:
                continue
        else:
            sol = cfg.shuffled_solution(rng)
            mask = rng.random(cfg.S) < 0.6
            if mask.sum() < cfg.S // 3:
                continue
            grid = np.where(mask, 0, sol).astype(np.int8)
            if cfg.count_solutions(grid) != 1:
                continue
            d = cfg.propagation_depth(grid)
            if d is None or d < 1:
                continue
            it = {"grid": grid, "sol": sol, "depth": d}
        key = it["grid"].tobytes()
        if key in seen:
            continue
        seen.add(key)
        pool.append(it)
        counts[it["depth"]] = counts.get(it["depth"], 0) + 1
        if verbose and len(pool) % 200 == 0:
            print(f"  pool {len(pool)}/{n_total} attempts {attempts} "
                  f"depths {dict(sorted(counts.items()))}", flush=True)
    return pool, counts, attempts


class SudokuSource:
    """Batched LM source. vocab: 0 blank, 1..N digits, SEP = N+1."""

    def __init__(self, cfg: SudokuConfig, pool, seed: int, shuffle: bool = True):
        self.cfg = cfg
        self.pool = pool
        self.SEP = cfg.N + 1
        self.VOCAB = cfg.N + 2
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(pool))
        self.ptr = 0
        self.shuffle = shuffle
        if shuffle:
            self.rng.shuffle(self.order)

    def encode(self, item):
        grid, sol = item["grid"], item["sol"]
        seq = np.concatenate([grid, [self.SEP], sol]).astype(np.int64)
        return seq[:-1], seq[1:], (grid == 0)

    def next_items(self, B: int):
        items = []
        for _ in range(B):
            if self.ptr >= len(self.order):
                self.ptr = 0
                if self.shuffle:
                    self.rng.shuffle(self.order)
            items.append(self.pool[self.order[self.ptr]])
            self.ptr += 1
        return items

    def batch_tensors(self, B: int, device: str = "cpu"):
        import torch
        items = self.next_items(B)
        xs, ys, bm = zip(*[self.encode(it) for it in items])
        x = torch.tensor(np.array(xs), dtype=torch.long, device=device)
        y = torch.tensor(np.array(ys), dtype=torch.long, device=device)
        blank = torch.tensor(np.array(bm), dtype=torch.bool, device=device)
        depth = torch.tensor([it["depth"] for it in items], dtype=torch.long)
        return x, y, blank, depth, items
