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
            sp = sp_id if sp_id >= 0 else 1 << 30
            word0 = sp + 1 - 0 if False else None
            # v2: слова начинаются с id>=54 (после спец 4 + 50 чаров); v1 — все word-токены
            cid0 = meta["stoi"].get("<bos>")
            chars_end = 4 + 50 if "prep4k" in datadir else 0
            for t in va[s:s+96]:
                t = int(t)
                if "prep4k" in datadir:
                    if t >= 4 + 50 or t == sp: nwords += 1      # слово- или <sp>-токен
                else:
                    w = itos.get(str(t), "")
                    if re.fullmatch(r"[a-z']+", w) or t == unk_id: nwords += 1
    print(json.dumps(dict(ckpt=os.path.basename(ckpt), data=datadir,
                          token_nll=round(tot / ntok, 4),
                          word_ppl=round(math.exp(tot / nwords), 2),
                          words_per_96tok=round(nwords / (10 * 16), 1))))

if __name__ == "__main__":
    main()
