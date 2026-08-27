#!/usr/bin/env python3
"""LeanCore data prep: clean real corpus from the `shakespeare` PyPI package
(only reachable source in this sandbox), dedup, word-level tokenizer, uint16 tokens."""
import os, re, json, glob, collections, numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
TXT = sorted(glob.glob(os.path.join(ROOT, "data/shakespeare-0.6/shksprdata/texts/*_gut.txt")) \
             + [os.path.join(ROOT, "data/shakespeare-0.6/miltondata/texts/paradiseregained.txt")])

def clean(txt: str) -> str:
    # strip Project Gutenberg boilerplate if present
    m = re.search(r"\*\*\*\s*START OF[^*]*\*\*\*", txt)
    if m: txt = txt[m.end():]
    m = re.search(r"\*\*\*\s*END OF", txt)
    if m: txt = txt[:m.start()]
    txt = re.sub(r"\[.*?\]", " ", txt)          # stage/illustration notes
    txt = re.sub(r"\r", "", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def main():
    raw_parts, seen_files = [], set()
    for f in TXT:
        base = os.path.basename(f)
        if base in seen_files: continue
        seen_files.add(base)
        t = clean(open(f, encoding="utf-8", errors="ignore").read())
        if len(t) > 3000: raw_parts.append("\n\n" + t)
    corpus = "\n".join(raw_parts)

    # ---- component: exact line-level dedup (data hygiene, measured) ----
    lines, seen = [], set()
    n_tot, n_dup = 0, 0
    for ln in corpus.split("\n"):
        ln2 = ln.strip()
        if not ln2: continue
        n_tot += 1
        k = ln2.lower()
        if k in seen: n_dup += 1; continue
        seen.add(k); lines.append(ln2)
    dedup_ratio = n_dup / max(n_tot, 1)
    corpus = "\n".join(lines)
    print(f"files={len(seen_files)}  lines={n_tot}  dup_lines_removed={n_dup} ({dedup_ratio:.1%})  chars={len(corpus):,}")

    # difficulty per line (for curriculum study): mean chars per word
    diff = np.array([len(l) / max(1, len(l.split())) for l in lines], dtype=np.float32)

    # ---- tokenizer: lowercase word-level, top-V + specials ----
    V = 8000
    toks = re.findall(r"[a-z']+|[0-9]+|[^\s\w]", corpus.lower())
    cnt = collections.Counter(toks)
    most = [w for w, _ in cnt.most_common(V - 3)]
    stoi = {"<pad>": 0, "<bos>": 1, "<unk>": 2}
    for w in most: stoi[w] = len(stoi)
    itos = {i: w for w, i in stoi.items()}
    ids = np.array([stoi.get(t, 2) for t in toks], dtype=np.uint16)
    oov = float((ids == 2).mean())
    print(f"tokens={len(ids):,}  vocab={len(stoi)}  OOV_rate={oov:.2%}")

    # split: last 6% of tokens is val (fixed for all runs)
    n_val = int(len(ids) * 0.06)
    tr, va = ids[:-n_val], ids[-n_val:]
    os.makedirs(os.path.join(ROOT, "data/prep"), exist_ok=True)
    np.save(os.path.join(ROOT, "data/prep/train.npy"), tr)
    np.save(os.path.join(ROOT, "data/prep/val.npy"), va)
    np.save(os.path.join(ROOT, "data/prep/diff.npy"), diff)
    json.dump({"stoi": stoi, "itos": itos, "vocab": len(stoi),
               "stats": {"chars": len(corpus), "tokens": len(ids), "dup_ratio": dedup_ratio, "oov": oov,
                         "files": sorted(seen_files)}},
              open(os.path.join(ROOT, "data/prep/meta.json"), "w"))
    print(f"train_tokens={len(tr):,}  val_tokens={len(va):,}")

if __name__ == "__main__":
    main()
