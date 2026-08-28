#!/usr/bin/env python3
"""LeanCore data prep v2 (corpus ×3): старые 43 файла (shakespeare+milton-regained)
+ неиспользованный локальный Milton (paradise lost, areopagitica, comus, lallegro, allegro)
+ GITenberg-книги (data/books_txt/*.txt; источник: codeload.github.com/GITenberg/* — список в stats.files_new): Faerie Queene I/II, Faustus-1604,
Bacon Essays, Donne Poems I, Defence of Poesie, Paradise Regained, KJV Bible (30.txt).
Хигиена: та же очистка/дедуп. V = 16000 (OOV ниже). VAL-SОVMESTIMOSTЬ: val = последние
6% token-потока СТАРОГО корпуса (те же строки, что у всех чемпионов), просто в новой нумерации.
Выход: data/prep6k/{train,val}.npy + meta.json (формат совместим). """
import os, re, json, glob, collections, numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
TXT_OLD = sorted(glob.glob(os.path.join(ROOT, "data/shakespeare-0.6/shksprdata/texts/*_gut.txt"))) \
    + [os.path.join(ROOT, "data/shakespeare-0.6/miltondata/texts/paradiseregained.txt")]
TXT_MILTON = [os.path.join(ROOT, "data/shakespeare-0.6/miltondata/texts", f) for f in
              ("paradise_lost_(no_introduction)_gut.txt", "areopagitica_gut.txt",
               "comus_gut.txt", "lallegro_il_penseroso_comus_and_lycidas_gut.txt", "allegro.txt")]
TXT_NEW = sorted(glob.glob(os.path.join(ROOT, "data/books_txt/*.txt")))

V = 16000

def clean(txt: str) -> str:
    m = re.search(r"\*\*\*\s*START OF[^*]*\*\*\*", txt)
    if m: txt = txt[m.end():]
    m = re.search(r"\*\*\*\s*END OF", txt)
    if m: txt = txt[:m.start()]
    txt = re.sub(r"\[.*?\]", " ", txt)
    txt = re.sub(r"\r", "", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def lines_of(files, minlen=3000):
    parts, names = [], set()
    for f in files:
        if not os.path.exists(f): continue
        base = os.path.basename(f)
        if base in names: continue
        names.add(base)
        t = clean(open(f, encoding="utf-8", errors="ignore").read())
        if len(t) > minlen: parts.append("\n\n" + t)
    return "\n".join(parts), sorted(names)

def dedup(corpus):
    out, seen, ndup, ntot = [], set(), 0, 0
    for ln in corpus.split("\n"):
        ln = ln.strip()
        if not ln: continue
        ntot += 1
        k = ln.lower()
        if k in seen: ndup += 1; continue
        seen.add(k); out.append(ln)
    return out, ndup, ntot

def tokenize(corpus_lines):
    return re.findall(r"[a-z']+|[0-9]+|[^\s\w]", "\n".join(corpus_lines).lower())

def main():
    # старый корпус (тот же набор и порядок!) → его хвост = наш val
    corpus_old, names_old = lines_of(TXT_OLD)
    lines_old, nd_o, nt_o = dedup(corpus_old)
    toks_old = tokenize(lines_old)
    n_val = int(len(toks_old) * 0.06)
    toks_train_old, toks_val = toks_old[:-n_val], toks_old[-n_val:]

    CAP = 700_000        # cap символов на файл: KJV не задавит микс (жанровый баланс)
    capped = []
    for f in TXT_MILTON + TXT_NEW:
        if not os.path.exists(f): continue
        t = open(f, encoding="utf-8", errors="ignore").read()[:CAP]
        capped.append((f, t))
    parts, names_new = [], []
    for f, t in capped:
        t2 = clean(t)
        if len(t2) > 3000: parts.append("\n\n" + t2); names_new.append(os.path.basename(f))
    corpus_new = "\n".join(parts)
    lines_new, nd_n, nt_n = dedup(corpus_new)
    toks_new = tokenize(lines_new)

    # словарь по полному train-потоку
    all_train = toks_train_old + toks_new
    cnt = collections.Counter(all_train)
    most = [w for w, _ in cnt.most_common(V - 3)]
    stoi = {"<pad>": 0, "<bos>": 1, "<unk>": 2}
    for w in most: stoi[w] = len(stoi)
    itos = {i: w for w, i in stoi.items()}

    tr = np.array([stoi.get(t, 2) for t in all_train], dtype=np.uint16)
    va = np.array([stoi.get(t, 2) for t in toks_val], dtype=np.uint16)
    oov_tr = float((tr == 2).mean()); oov_va = float((va == 2).mean())

    outp = os.path.join(ROOT, "data/prep6k")
    os.makedirs(outp, exist_ok=True)
    np.save(os.path.join(outp, "train.npy"), tr)
    np.save(os.path.join(outp, "val.npy"), va)
    json.dump({"stoi": stoi, "itos": itos, "vocab": len(stoi),
               "stats": {"chars": int(sum(len(t) for t in (toks_train_old + toks_new)) * 0 + len("\n".join(lines_old + lines_new))),
                         "tokens_train": int(len(tr)), "tokens_val": int(len(va)),
                         "dup_ratio_old": nd_o / max(nt_o, 1), "dup_ratio_new": nd_n / max(nt_n, 1),
                         "oov_train": oov_tr, "oov_val": oov_va,
                         "files_old": names_old, "files_new": names_new}},
              open(os.path.join(outp, "meta.json"), "w"))
    print(f"OLD: files={len(names_old)} toks={len(toks_old):,} | NEW: files={len(names_new)} toks={len(toks_new):,}")
    print(f"train={len(tr):,} val={len(va):,} (те же строки-источники, что раньше)  OOV tr={oov_tr:.2%} va={oov_va:.2%}  V={len(stoi)}")

if __name__ == "__main__":
    main()
