#!/usr/bin/env python3
"""prep_data4k — словарь 4096 с посимвольным fallback (OOV → chars), честная
сравнимость с 8k-версией через per-word NLL (делим сумму NLL на число слов корпуса,
а не на число токенов). Раскладка словаря:
  0..3   спецтокены <pad> <bos> <eos> <sp>
  4..259 байты/символы (printable + одиночные буквы как токены)
  260..N топ-слова."""
import os, re, json, glob, collections, numpy as np
from prep_data import clean, TXT  # переиспользую чистку

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data/prep4k")
VOCAB = 4096
NWORDS = VOCAB - 260

def read_corpus():
    raw_parts, seen = [], set()
    for f in TXT:
        base = os.path.basename(f)
        if base in seen: continue
        seen.add(base)
        t = clean(open(f, encoding="utf-8", errors="ignore").read())
        if len(t) > 3000: raw_parts.append("\n\n" + t)
    return "\n".join(raw_parts)

def main():
    corpus = read_corpus()
    words = re.findall(r"[a-z']+|[0-9]+|[^\s\w]", corpus.lower())
    # дедуп: как в v1 (удаление повторных строк)
    lines = corpus.split("\n")
    seen = set(); out = []
    for ln in lines:
        h = hash(ln.strip())
        if h in seen: continue
        seen.add(h); out.append(ln)
    corpus_d = "\n".join(out)
    words_d = re.findall(r"[a-z']+|[0-9]+|[^\s\w]", corpus_d.lower())

    freq = collections.Counter(words_d)
    top = [w for w, _ in freq.most_common(NWORDS)]
    stoi, itos = {}, {}
    for i, w in enumerate(["<pad>", "<bos>", "<eos>", "<sp>"]): stoi[w] = i; itos[str(i)] = w
    cid = 4
    charset = sorted(set("".join(top)) & set("abcdefghijklmnopqrstuvwxyz'0123456789"))
    chars = list("abcdefghijklmnopqrstuvwxyz'0123456789.,;:!?-\"()[] ")
    charset = [c for c in chars if c]                    # фиксированный набор
    for c in charset:
        if c not in stoi: stoi[c] = cid; itos[str(cid)] = c; cid += 1
    print("char tokens:", cid - 4)
    for w in top:
        if w not in stoi: stoi[w] = cid; itos[str(cid)] = w; cid += 1
    assert cid <= VOCAB

    # кодирование: известное слово → 1 токен; неизвестное → <sp> + chars (посимвольно)
    def encode(tokens):
        ids = [1]
        sp = stoi["<sp>"]
        for w in tokens:
            iw = stoi.get(w)
            if iw is not None: ids.append(iw)
            else:
                ids.append(sp)
                for ch in w:
                    ids.append(stoi.get(ch, sp))
        return ids

    ids = encode(words_d)
    n = len(ids); split = int(n * 0.94)
    os.makedirs(OUT, exist_ok=True)
    np.save(f"{OUT}/train.npy", np.array(ids[:split], np.uint16))
    np.save(f"{OUT}/val.npy", np.array(ids[split:], np.uint16))
    meta = dict(vocab=cid, stoi=stoi, itos=itos, tokens=n,
                words=len(words_d), oov_words=None)
    # метрики: сколько слов не в словаре и на сколько токенов они раздулись
    oov = sum(1 for w in words_d if w not in stoi)
    meta["oov_words"] = oov
    meta["oov_rate"] = round(oov / len(words_d), 5)
    json.dump(meta, open(f"{OUT}/meta.json", "w"))
    print(f"tokens={n:,} (words={len(words_d):,}, раздувание ×{n/len(words_d):.3f}); "
          f"oov_words={oov} ({100*oov/len(words_d):.2f}% слово-токенов пишутся посимвольно); "
          f"train={split:,} val={n-split:,} vocab={cid}")
    print("splits saved to", OUT)

if __name__ == "__main__":
    main()
