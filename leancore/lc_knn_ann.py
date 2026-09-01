# lc_knn_ann.py — решающий зонд семейства «триггер-память»: приближённый индекс (IVF)
#
# Точный скан (lc_knn.py) = 6.07 мс/ток при −16% PPL — мёртв по скорости.
# IVF: k-means-lite центроиды (обычные ключи), запрос сканирует только nprobe ближайших списков.
# Цена теоретически: C·d + (N/C)·nprobe·d MAC vs N·d — ~100× дешевле.
# Честный протокол: (nprobe, λ) калибруются на val[:half], финальный отчёт на val[half:].
# Контроль качества индекса: recall@32 против точного скана (доля пересечения top-32).
import os, sys, json, math, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from nano_lc import layernorm, NanoGPT  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/results/ckpt_L_ssa1500.npz"
CACHE = f"{ROOT}/results/knn_ds_{os.path.basename(CKPT).replace('.npz','')}.npz"


def build_datastore(model, tr, stride=3, lim=903168, ctx=96, lane=64):
    chunks = [tr[i:i + ctx] for i in range(0, lim - ctx - 1, ctx)]
    Zs, Ys = [], []
    for i in range(0, len(chunks), lane):
        grp = chunks[i:i + lane]
        x = np.stack(grp)
        y = np.stack([tr[(i + j) * ctx + 1:(i + j) * ctx + ctx + 1] for j in range(len(grp))])
        H, _ = model.forward(x)
        z, _ = layernorm(H, model.p.d["lnfg"], model.p.d["lnfb"])
        Zs.append(z.reshape(-1, z.shape[-1]).astype(np.float32))
        Ys.append(y.reshape(-1))
    Z, Y = np.concatenate(Zs)[::stride], np.concatenate(Ys)[::stride]
    Dn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    return Dn.astype(np.float32), Y.astype(np.int64)


def kmeans_lite(Dn, C=512, iters=8, sample=40000, seed=7):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Dn), size=min(sample, len(Dn)), replace=False)
    S = Dn[idx]
    Cen = S[rng.choice(len(S), C, replace=False)].copy()
    for it in range(iters):
        lab = (S @ Cen.T).argmax(-1)
        for c in range(C):
            m = lab == c
            if m.any():
                Cen[c] = S[m].mean(0)
        Cen /= (np.linalg.norm(Cen, axis=1, keepdims=True) + 1e-8)
        print(f"  kmeans iter {it}: перемещено {int((lab != ((S @ Cen.T).argmax(-1))).sum())}", flush=True)
    return Cen


def main():
    C_CL, K_TOP = 512, 32
    tr = np.load(f"{ROOT}/data/prep/train.npy").astype(np.int64)
    va = np.load(f"{ROOT}/data/prep/val.npy").astype(np.int64)
    V = json.load(open(f"{ROOT}/data/prep/meta.json"))["vocab"]
    ck = np.load(CKPT)
    model = NanoGPT(V, D=192, L=4, h=6, ff=576, T=96, kind="ema")
    for k in model.p.d:
        model.p.d[k][...] = ck[k]

    if os.path.exists(CACHE):
        z = np.load(CACHE); Dn, Y = z["Dn"], z["Y"]
        print(f"datastore из кэша: {len(Dn)}", flush=True)
    else:
        print("строю datastore...", flush=True)
        Dn, Y = build_datastore(model, tr)
        np.savez(CACHE, Dn=Dn, Y=Y)

    print("kmeans-lite центроиды...", flush=True)
    Cen = kmeans_lite(Dn, C=C_CL)
    lab = (Dn @ Cen.T).argmax(-1)
    lists = [np.flatnonzero(lab == c) for c in range(C_CL)]
    sizes = np.array([len(l) for l in lists])
    print(f"index: {C_CL} списков, размер min/med/max = {sizes.min()}/{int(np.median(sizes))}/{sizes.max()}", flush=True)

    # --- val z (те же 128 чанков, тот же сплит, что lc_knn)
    chunks = [va[i:i + 96] for i in range(0, 96 * 128, 96)]
    Zv, Yv = [], []
    for i in range(0, len(chunks), 64):
        grp = chunks[i:i + 64]
        x = np.stack(grp)
        y = np.stack([va[(i + j) * 96 + 1:(i + j) * 96 + 97] for j in range(len(grp))])
        H, _ = model.forward(x)
        z, _ = layernorm(H, model.p.d["lnfg"], model.p.d["lnfb"])
        Zv.append(z.reshape(-1, 192).astype(np.float32)); Yv.append(y.reshape(-1))
    Zv, Yv = np.concatenate(Zv), np.concatenate(Yv)
    half = 96 * 64
    (Zc, Yc), (Zt, Yt) = (Zv[:half], Yv[:half]), (Zv[half:], Yv[half:])

    def model_probs(Z):
        E = model.p.d["E"]
        out = np.empty((len(Z), V), np.float32)
        for i in range(0, len(Z), 512):
            lg = Z[i:i + 512] @ E.T
            lg -= lg.max(-1, keepdims=True)
            p = np.exp(lg); p /= p.sum(-1, keepdims=True)
            out[i:i + 512] = p
        return out
    print("модельные вероятности...", flush=True)
    Pc, Pt = model_probs(Zc), model_probs(Zt)

    def knn_probs(Zq, nprobe, temp=0.05):
        """IVF top-K_TOP по nprobe ближайшим спискам; точность через recall ниже."""
        Qn = Zq / (np.linalg.norm(Zq, axis=1, keepdims=True) + 1e-8)
        csims = Qn @ Cen.T
        P = np.zeros((len(Zq), V), np.float16)
        recall = 0.0
        for qi in range(len(Zq)):
            cl = np.argpartition(csims[qi], -nprobe)[-nprobe:]
            cand = np.concatenate([lists[c] for c in cl])
            s = Dn[cand] @ Qn[qi]
            k_ = min(K_TOP, cand.size)
            top = np.argpartition(s, -k_)[-k_:]
            gi = cand[top]; w = s[top]
            # recall против точного скана (только метрика; считается дёшево каждые ~512 запросов)
            if qi % 512 == 0:
                s_full_top = np.argpartition(Dn @ Qn[qi], -K_TOP)[-K_TOP:]
                recall += len(set(gi.tolist()) & set(s_full_top.tolist())) / K_TOP
            w = np.exp((w - w.max()) / temp); w /= w.sum()
            np.add.at(P[qi], Y[gi], w.astype(np.float16))
        n_meas = (len(Zq) + 511) // 512
        return P.astype(np.float32), recall / n_meas

    def ppl(P, Km, lam, Yv_):
        p = (1 - lam) * P + lam * Km
        return math.exp(float(-np.log(np.maximum(p[np.arange(len(Yv_)), Yv_], 1e-12)).mean()))

    print("\n nprobe | recall@32 | цена мс/ток | PPL@cal λ* | PPL@test λ* (λ* — лучший по cal)", flush=True)
    best = None
    for nprobe in (1, 2, 4, 8):
        t0 = time.time()
        Kc, rc = knn_probs(Zc, nprobe)
        Kt, rt = knn_probs(Zt, nprobe)
        ms = (time.time() - t0) * 1e3 / (len(Zc) + len(Zt))
        row = (nprobe, ms, rc)
        grid = {}
        for lam in (0.1, 0.2, 0.3):
            grid[lam] = (ppl(Pc, Kc, lam, Yc), ppl(Pt, Kt, lam, Yt))
        lbest = min(grid, key=lambda l: grid[l][0])
        print(f"  {nprobe:4d}  |  {rc:.3f}   |   {ms:.3f}     | {grid[lbest][0]:7.2f} λ={lbest} | {grid[lbest][1]:7.2f}", flush=True)
        if best is None or grid[lbest][1] < best[1]:
            best = (grid[lbest][1], nprobe, lbest, rc, ms)
    base = ppl(Pt, np.zeros_like(Pt), 0.0, Yt)
    print(f"\nчистая модель PPL@test = {base:.2f}")
    print(f"лучший ANN: nprobe={best[1]}, λ={best[2]} → PPL@test={best[0]:.2f} "
          f"({(best[0]/base-1)*100:+.1f}%), recall@32={best[3]:.3f}, цена {best[4]:.3f} мс/ток")
    print("точный скан для сравнения: −16.0% при 6.07 мс/ток (lc_knn.py)")


if __name__ == "__main__":
    main()
