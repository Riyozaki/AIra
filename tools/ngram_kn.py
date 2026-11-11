"""B-classic: interpolated *modified* Kneser-Ney (Chen & Goodman 1998) byte n-gram LM.

Exact keys (no hashing): an n-gram of up to 8 bytes is packed into uint64, newest byte lowest.
Top order uses raw counts, lower orders use continuation counts N1+(. h w).
Three discounts D1, D2, D3+ per order, estimated from count-of-counts on train
(standard closed form; nothing is tuned on valid except the max order K).

Usage: .venv/bin/python tools/ngram_kn.py --order 5 --split valid [--train-bytes N]
Reports bits-per-byte (bpb) on the chosen split; first `order` bytes of the split are
conditioned on the last bytes of the preceding split (contiguous file), so every byte is scored.
"""
import argparse, json, time, os
import numpy as np

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def pack(buf, k):
    """keys[t] = bytes buf[t-k..t] packed, newest (buf[t]) in lowest 8 bits. len = len(buf)-k"""
    n = len(buf) - k
    key = np.zeros(n, dtype=np.uint64)
    for j in range(k + 1):  # j = age of byte
        key |= buf[k - j: k - j + n].astype(np.uint64) << np.uint64(8 * j)
    return key


def uniq_counts(sorted_keys):
    """sorted keys -> (unique keys, counts)"""
    if len(sorted_keys) == 0:
        return sorted_keys, np.zeros(0, np.int64)
    b = np.flatnonzero(np.diff(sorted_keys)) + 1
    starts = np.concatenate(([0], b))
    cnt = np.diff(np.concatenate((starts, [len(sorted_keys)])))
    return sorted_keys[starts], cnt.astype(np.int64)


def discounts(cnt):
    n = [np.sum(cnt == i) for i in (1, 2, 3, 4)]
    if min(n) == 0:  # too few low counts (e.g. order 0 over <=256 symbols): fixed fallback
        return np.array([0.0, 0.5, 1.0, 1.5])
    Y = n[0] / (n[0] + 2 * n[1])
    D = np.array([0.0, 1 - 2 * Y * n[1] / n[0], 2 - 3 * Y * n[2] / n[1], 3 - 4 * Y * n[3] / n[2]])
    return np.clip(D, 0.0, [0, 1, 2, 3])  # D[i] in [0,i]; D[0]=0


class Order:
    """statistics of one order: n-gram keys (context len k + current byte)"""

    def __init__(self, keys, cnt, k):
        self.k = k
        self.keys, self.cnt = keys, cnt
        self.D = discounts(cnt)
        ctx = keys >> np.uint64(8)
        self.ctx, first = np.unique(ctx, return_index=True)  # keys sorted -> ctx sorted
        bounds = np.concatenate((first, [len(keys)]))
        cs = np.concatenate(([0], np.cumsum(cnt)))
        self.ctx_total = cs[bounds[1:]] - cs[bounds[:-1]]
        def nk(mask):
            m = np.concatenate(([0], np.cumsum(mask.astype(np.int64))))
            return m[bounds[1:]] - m[bounds[:-1]]
        self.n1, self.n2, self.n3 = nk(cnt == 1), nk(cnt == 2), nk(cnt >= 3)

    def lookup(self, qkeys):
        """returns c(h,w), c(h), gamma-numerator for each query key (0 where unseen)"""
        i = np.searchsorted(self.keys, qkeys)
        i = np.minimum(i, len(self.keys) - 1)
        hit = self.keys[i] == qkeys
        c = np.where(hit, self.cnt[i], 0)
        qctx = qkeys >> np.uint64(8)
        j = np.minimum(np.searchsorted(self.ctx, qctx), len(self.ctx) - 1)
        chit = self.ctx[j] == qctx
        tot = np.where(chit, self.ctx_total[j], 0)
        D = self.D
        gnum = np.where(chit, D[1] * self.n1[j] + D[2] * self.n2[j] + D[3] * self.n3[j], 0.0)
        disc = D[np.minimum(c, 3)]
        return c, tot, gnum, disc


def build(train, K):
    """orders 0..K (context lengths). returns list indexed by k."""
    orders = [None] * (K + 1)
    keys = pack(train, K); keys.sort()
    uk, uc = uniq_counts(keys); del keys
    orders[K] = Order(uk, uc, K)
    prev_unique = uk
    for k in range(K - 1, -1, -1):
        mask = np.uint64((1 << (8 * (k + 1))) - 1)
        ck = np.sort(prev_unique & mask)  # continuation: distinct left-extensions
        uk, uc = uniq_counts(ck); del ck
        orders[k] = Order(uk, uc, k)
        prev_unique = uk
    return orders


def score(orders, buf, start):
    """log2-probs for buf[start:], using contexts from buf[start-K:]"""
    K = len(orders) - 1
    n = len(buf) - start
    p = np.full(n, 1.0 / 256)  # order -1: uniform
    for k in range(0, K + 1):
        q = pack(buf[start - k:], k)[:n]
        c, tot, gnum, disc = orders[k].lookup(q)
        safe = np.maximum(tot, 1)
        pk = np.maximum(c - disc, 0) / safe + (gnum / safe) * p
        p = np.where(tot > 0, pk, p)
    return np.log2(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=5, help="max context length K (n = K+1)")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--train-bytes", type=int, default=90_000_000)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    full = np.fromfile(os.path.join(RAW, "enwik8"), dtype=np.uint8)
    train = full[: a.train_bytes]
    t0 = time.time()
    orders = build(train, a.order)
    tb = time.time() - t0
    off = {"valid": 90_000_000, "test": 95_000_000}[a.split]
    buf = full[off - a.order: off + 5_000_000]
    t1 = time.time()
    lp = score(orders, buf, a.order)
    bpb = float(-lp.mean())
    res = dict(model="mKN-byte", K=a.order, n=a.order + 1, split=a.split, train_bytes=a.train_bytes,
               bpb=bpb, build_s=round(tb, 1), score_s=round(time.time() - t1, 1),
               table_entries=int(sum(len(o.keys) for o in orders)))
    print(json.dumps(res))
    if a.log:
        with open(a.log, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
