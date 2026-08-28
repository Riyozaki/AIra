#!/usr/bin/env python3
"""Инкрементальный (потоковый) инференс nano-моделей с EMA-миксерами.

Математика: EMA-плечо h_t = a⊙h_{t−1} + (1−a)⊙x_t переносится состоянием ТОЧНО
(сверено: max|ΔNLL| на позицию = 2.9e-6 против оконного forward). Граница чанка (t=T):
сброс состояний ≡ тренировочной семантике (h₀=0).
ADR в потоке: пороговый гейт σ(s)≥tau вместо top-k по окну (качество измерено ниже).

Режимы:
  ppl <ckpt> <kind> <adr|none> [--tau X]
  bench <ckpt> <kind> <adr|none> N
  gen <ckpt> <kind> <adr|none> "prompt" N
"""
import os, sys, json, math, time, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from nano_lc import NanoGPT, gelu, layernorm, linear, softmax_ce

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_model(ckpt, kind, adr, datadir="data/prep"):
    d = np.load(ckpt)
    V = json.load(open(f"{ROOT}/{datadir}/meta.json"))["vocab"]
    m = NanoGPT(V, kind=kind, adr_kf=adr)
    for k in m.p.d:
        if k in d.files:
            m.p.d[k][...] = np.asarray(d[k], dtype=m.p.d[k].dtype)
    return m


class Incr:
    """Потоковый движок: по каждому блоку — состояние EMA h (D,) и шаг t в чанке."""
    def __init__(self, model):
        self.m = model
        p = model.p.d
        self.blocks = [{"routed": f"b{i}.rw" in p, "h": np.zeros(p["E"].shape[1], np.float32)}
                       for i in range(model.L)]
        self.t = 0

    def reset(self):
        for b in self.blocks: b["h"][...] = 0
        self.t = 0

    def arm(self, x, i):
        m, p = self.m, self.m.p.d
        y, _ = layernorm(x[None, None], p[f"b{i}.ln1g"], p[f"b{i}.ln1b"])
        a = 1.0 / (1.0 + np.exp(-p[f"b{i}.th"]))
        h = self.blocks[i]["h"]
        h[...] = a * h + (1.0 - a) * y[0, 0]
        mix = (h * p[f"b{i}.sc"]) @ p[f"b{i}.Wm"]
        h2 = x + mix
        z, _ = layernorm(h2[None, None], p[f"b{i}.ln2g"], p[f"b{i}.ln2b"])
        pre, _ = linear(z, p[f"b{i}.fc1"])
        g1, _ = gelu(pre)
        o, _ = linear(g1, p[f"b{i}.fc2"])
        return mix + o[0, 0]

    def step(self, tok, tau=0.0):
        m, p = self.m, self.m.p.d
        if self.t >= m.T:
            self.reset()                               # chunk-exact режим
        x = (p["E"][tok] + p["pos"][self.t]).astype(np.float32).copy()
        for i in range(self.m.L):
            b = self.blocks[i]
            if b["routed"]:
                s = float(x @ p[f"b{i}.rw"] + p[f"b{i}.rb"])
                g = 1.0 / (1.0 + math.exp(-s))
                if g < tau:
                    continue
                x = x + g * self.arm(x, i)
            else:
                x = x + self.arm(x, i)
        self.t += 1
        return x


def eval_ppl(m, inc, X, tau):
    p = m.p.d; nll = 0.0; cnt = 0
    inc.reset()
    for t in range(len(X) - 1):
        h = inc.step(int(X[t]), tau)
        z, _ = layernorm(h[None, None], p["lnfg"], p["lnfb"])
        lg = (z @ p["E"].T)[0, 0]
        lg -= lg.max(); pe = np.exp(lg); pe /= pe.sum()
        if t % m.T == m.T - 1:          # граница чанка — пропускаем, как в оконной метрике
            continue
        nll -= math.log(pe[int(X[t + 1])]); cnt += 1
    return math.exp(nll / cnt)


def main():
    mode, ckpt, kind = sys.argv[1], sys.argv[2], sys.argv[3]
    adr = None if sys.argv[4] == "none" else float(sys.argv[4])
    DD = sys.argv[5] if len(sys.argv) > 5 and sys.argv[4+1].startswith("data/") else "data/prep"
    m = load_model(ckpt, kind, adr, DD.replace("../", ""))
    inc = Incr(m)

    if mode == "ppl":
        tau = 0.0
        if "--tau" in sys.argv: tau = float(sys.argv[sys.argv.index("--tau") + 1])
        va = np.load(f"{ROOT}/{DD}/val.npy").astype(np.int64)
        X = va[:96 * 4]
        tot = 0.0; nch = 0
        for off in range(0, len(X) - 96 + 1, 96):
            Hh, _ = m.forward(X[off:off + 96][None, :])
            lg = m.logits(Hh)[0]
            lg -= lg.max(-1, keepdims=True)
            pe = np.exp(lg); pe /= pe.sum(-1, keepdims=True)
            tot -= float(np.log(pe[np.arange(95), X[off + 1:off + 96]]).sum()); nch += 1
        pplw = math.exp(tot / (nch * 95))
        ppli = eval_ppl(m, inc, X, tau)
        print(json.dumps(dict(mode="ppl", ckpt=os.path.basename(ckpt), tau=tau,
                              ppl_windowed=round(pplw, 2), ppl_incremental=round(ppli, 2))))

    elif mode == "bench":
        n = int(sys.argv[5]); p = m.p.d
        ids = [2]
        inc.reset(); inc.step(2)
        t0 = time.time()
        for _ in range(n):
            x = inc.step(ids[-1])
            z, _ = layernorm(x[None, None], p["lnfg"], p["lnfb"])
            lg = (z @ p["E"].T)[0, 0]
            ids.append(int(np.argmax(lg)))
        dt = time.time() - t0
        print(json.dumps(dict(mode="bench_incr", tok_s=round(n / dt, 1))))
        ids2 = ids[-96:]
        t0 = time.time()
        for _ in range(n):
            H, _ = m.forward(np.array(ids2[-96:])[None, :])
            z, _ = layernorm(H[:, -1:], p["lnfg"], p["lnfb"])
            lg = (z @ p["E"].T)[0, 0]
            ids2.append(int(np.argmax(lg)))
        dt = time.time() - t0
        print(json.dumps(dict(mode="bench_windowed", tok_s=round(n / dt, 1))))

    elif mode == "gen":
        prompt, n = sys.argv[5], int(sys.argv[6])
        meta = json.load(open(f"{ROOT}/data/prep/meta.json"))
        stoi, itos = meta["stoi"], meta["itos"]
        enc = lambda s: [stoi.get(t, 2) for t in re.findall(r"[a-z']+|[0-9]+|[^\s\w]", s.lower())] or [1]
        ids = enc(prompt); rng = np.random.default_rng(3)
        inc.reset()
        p = m.p.d
        for tok in ids: inc.step(tok)
        for _ in range(n):
            x = inc.step(ids[-1], 0.4)
            z, _ = layernorm(x[None, None], p["lnfg"], p["lnfb"])
            lg = (z @ p["E"].T)[0, 0] / 0.9
            lg -= lg.max(); pe = np.exp(lg); pe /= pe.sum()
            ix = np.argsort(-pe)[:40]
            w = pe[ix]; w /= w.sum()
            ids.append(int(rng.choice(ix, p=w)))
        out = []
        for i in ids:
            w = itos.get(str(i), "<unk>")
            out.append((" " if out and w.isalpha() and out[-1][-1:].isalpha() else "") + w)
        print("".join(out))

if __name__ == "__main__":
    main()
