#!/usr/bin/env python3
"""cmp_word_ppl — честное сравнение моделей с РАЗНЫМИ токенизаторами: суммарный NLL
по вал-токенам делим на число слов вал-текста (слова = токены [a-z']+ или <unk> для v1;
для v2: слово-токены + маркеры <sp>, запускающие посимвольное кодирование).
Запуск: python3 cmp_word_ppl.py ckpt.npz <data_dir> <kind> <adr|none>"""
import os, sys, json, math, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nano_lc import NanoGPT, softmax_ce

ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    ckpt, datadir, kind = sys.argv[1], sys.argv[2], sys.argv[3]
    adr = None if sys.argv[4] == "none" else float(sys.argv[4])
    meta = json.load(open(f"{ROOT}/{datadir}/meta.json"))
    d = np.load(ckpt)
    m = NanoGPT(meta["vocab"], kind=kind, adr_kf=adr)
    for k in m.p.d:
        if k in d.files: m.p.d[k][...] = d[k]
    va = np.load(f"{ROOT}/{datadir}/val.npy").astype(np.int64)

    rng = np.random.default_rng(1234)
    tot, ntok = 0.0, 0
    for _ in range(10):
        st = rng.integers(0, len(va) - 97, size=16)
        for s in st:
            x = va[s:s+96]; y = va[s+1:s+97]
            H, _ = m.forward(x[None, :])
            l, _ = softmax_ce(m.logits(H), y[None, :])
            tot += float(l) * 96     # mean·N — восстановим сумму
            ntok += 96
    # words на ТОМ ЖЕ наборе окон
    itos = meta["itos"]; sp_id = meta["stoi"].get("<sp>", -1); unk_id = meta["stoi"].get("<unk>", -1)
    nwords = 0
    rng = np.random.default_rng(1234)
    for _ in range(10):
        st = rng.integers(0, len(va) - 97, size=16)
        for s in st:
            for t in va[s:s+96]:
                w = itos.get(str(int(t)), "")
                if re.fullmatch(r"[a-z']+", w): nwords += 1
                elif int(t) == unk_id or (sp_id >= 0 and int(t) == sp_id): nwords += 1
    print(json.dumps(dict(ckpt=os.path.basename(ckpt), data=datadir,
                          token_nll=round(tot / ntok, 4),
                          word_ppl=round(math.exp(tot / nwords), 2),
                          words_per_96tok=round(nwords / (10 * 16), 1))))

if __name__ == "__main__":
    main()
