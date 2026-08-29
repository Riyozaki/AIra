# lc_knn.py — честный зонд kNN-памяти (самая сильная форма «триггер-всплытия»)
#
# Идея семейства: вспоминать похожие скрытые состояния и голосовать их продолжениями.
# Это строго ИНФОРМАТИВНЕЕ любой trigger-only схемы (ключ = полный z, а не урезанный код),
# поэтому её провал на малом объёме = провал всего семейства.
#
# Протокол (как lc_shelf.py): λ и температура калибруются на val[:half], отчёт на val[half:].
# Отдельно меряется цена retrieval (мс/токен) → вердикт «fit + скорость».
import os, sys, json, math, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from nano_lc import layernorm, NanoGPT  # noqa: E402


def stream_z(model, ids, ctx=96, lane_cap=64):
    """z (после финального LN) для каждой позиции ids, нарезанной на ctx-чанки, батч по lane'ам."""
    chunks = [ids[i:i + ctx] for i in range(0, len(ids) - ctx - 1, ctx)]
    Zs, Ys = [], []
    for i in range(0, len(chunks), lane_cap):
        grp = chunks[i:i + lane_cap]
        x = np.stack(grp)
        y = np.stack([ids[(i + j) * ctx + 1:(i + j) * ctx + ctx + 1] for j in range(len(grp))])
        H, _ = model.forward(x)
        z, _ = layernorm(H, model.p.d["lnfg"], model.p.d["lnfb"])
        Zs.append(z.reshape(-1, z.shape[-1]).astype(np.float32))
        Ys.append(y.reshape(-1))
    return np.concatenate(Zs), np.concatenate(Ys)


def main():
    tr = np.load(f"{ROOT}/data/prep/train.npy").astype(np.int64)
    va = np.load(f"{ROOT}/data/prep/val.npy").astype(np.int64)

    V = json.load(open(f"{ROOT}/data/prep/meta.json"))["vocab"]
    ck = np.load(f"{ROOT}/results/ckpt_L_ssa1500.npz")
    model = NanoGPT(V, D=192, L=4, h=6, ff=576, T=96, kind="ema")
    for k in model.p.d:
        model.p.d[k][...] = ck[k]
    Vsz = model.V

    # --- datastore: 300k позиций train (каждая 3-я), таргеты — следующие токены
    Z, Y = stream_z(model, tr[:903168])
    Z, Y = Z[::3], Y[::3]
    print(f"datastore: {len(Z)} позиций, d={Z.shape[1]}", flush=True)

    print("datastore готов, память: читаем val", flush=True)
    # --- val: 128 чанков
    half = 96 * 64
    Zv, Yv = stream_z(model, va[:96 * 128 + 1])
    (Zc, Yc), (Zt, Yt) = (Zv[:half], Yv[:half]), (Zv[half:], Yv[half:])
    print(f"val-z готов: {Zv.shape}", flush=True)

    # --- базовые логиты самой модели (polny softmax по tied-E)
    def model_logp(Z):
        E = model.p.d["E"]
        out = np.empty((len(Z), Vsz), np.float32)
        for i in range(0, len(Z), 512):
            lg = Z[i:i + 512] @ E.T
            lg -= lg.max(-1, keepdims=True)
            p = np.exp(lg); p /= p.sum(-1, keepdims=True)
            out[i:i + 512] = p
        return out
    t0 = time.time()
    Pc, Pt = model_logp(Zc), model_logp(Zt)
    base_ms = (time.time() - t0) * 1e3 / (len(Zc) + len(Zt))
    print("model probs готовы", flush=True)

    # --- kNN: нормируем ключи, батчевые запросы, top-k по косинусу
    Dn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    del Z  # datastore больше не нужен — только нормированные ключи
    t0 = time.time()

    def knn_logp(Zq, k=32, temp=0.05):
        """p_knn(v) ∝ Σ_{j∈topk} exp(cos/temp)·[Y_j = v]"""
        Qn = Zq / (np.linalg.norm(Zq, axis=1, keepdims=True) + 1e-8)
        P = np.zeros((len(Zq), Vsz), np.float32)
        for i in range(0, len(Zq), 128):
            Q = Qn[i:i + 256]
            S = Q @ Dn.T                                   # (b, N)
            idx = np.argpartition(S, -k, axis=1)[:, -k:]
            w = np.take_along_axis(S, idx, 1)
            w = np.exp((w - w.max(-1, keepdims=True)) / temp)
            w /= w.sum(-1, keepdims=True)
            tgt = Y[idx]                                   # (b, k)
            p = np.zeros((len(Q), Vsz), np.float32)
            np.add.at(p, (np.repeat(np.arange(len(Q)), k), tgt.ravel()), w.ravel())
            P[i:i + 256] = p
        return P
    Kc, Kt = knn_logp(Zc), knn_logp(Zt)
    knn_ms = (time.time() - t0) * 1e3 / (len(Zc) + len(Zt))
    print(f"цена: model {base_ms:.3f} мс/ток | kNN-retrieval {knn_ms:.3f} мс/ток", flush=True)

    # --- калибровка λ на половине cal, отчёт на test
    def ppl(P, K, lam, Y):
        p = (1 - lam) * P + lam * K
        return math.exp(float(-np.log(np.maximum(p[np.arange(len(Y)), Y], 1e-12)).mean()))
    print("λ      PPL@cal   PPL@test")
    best, bl, bt = None, None, ppl(Pt, Kt, 0.0, Yt)
    print(f"0.00  {ppl(Pc, Kc, 0.0, Yc):8.2f} {bt:8.2f}   (чистая модель)")
    for lam in (0.05, 0.1, 0.2, 0.3, 0.5):
        c, t = ppl(Pc, Kc, lam, Yc), ppl(Pt, Kt, lam, Yt)
        print(f"{lam:.2f}  {c:8.2f} {t:8.2f}")
    # температура (грубая сетка) на лучшей λ=0.2
    print("\n temp-сетка при λ=0.2 (пересчёт только kNN-части):")
    for temp in (0.01, 0.025, 0.05, 0.1, 0.2):
        Kall = knn_logp(np.concatenate([Zc, Zt]), temp=temp)
        Kc2, Kt2 = Kall[:len(Zc)], Kall[len(Zc):]
        print(f"temp={temp:<6} cal {ppl(Pc, Kc2, 0.2, Yc):8.2f}  test {ppl(Pt, Kt2, 0.2, Yt):8.2f}", flush=True)


if __name__ == "__main__":
    main()
