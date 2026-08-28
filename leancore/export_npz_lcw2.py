#!/usr/bin/env python3
"""Экспорт nano-чекпойнта (npz) в контейнер LCW2 для потокового C-движка lc_stream.c.
Матрицы nano: (in,out). Тернаризация: q∈{-1,0,1}, s_out[j]=max_i|W[i,j]| — из QAT-значений
(уже точные кратные) восстанавливается БЕЗ потерь; из сырых fp32 — absmax-конвенция.
Голова Et: fp16 (D,V) — НЕ квантуем (потеря PPL +40% измерена, урок записан).
Формат: magic LCW2 | u32 L,D,0,V,T | f32 tau | u32 n | records: u64 namelen, name,
u8 dt(2=int8,3=fp16), u32 nd, dims, payload."""
import os, sys, struct, numpy as np

SRC, DST = sys.argv[1], sys.argv[2]
TAU = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4

d = dict(np.load(SRC))
D = int(d["E"].shape[1]); V = int(d["E"].shape[0]); T = int(d["pos"].shape[0])
L = max(int(k[1]) for k in d if k.startswith("b") and "." in k) + 1

def tern_outscale(W):                     # (in,out): q с per-out скейлом
    s = np.abs(W).max(0).clip(1e-5)
    q = np.clip(np.rint(W / s[None, :]), -1, 1).astype(np.int8)
    return np.ascontiguousarray(q), s.astype(np.float16)

recs = []
add = lambda name, arr, dt: recs.append((name.encode(), dt, np.ascontiguousarray(arr)))

for i in range(L):
    p = f"b{i}."
    add(p + "ln1", d[p + "ln1g"].astype(np.float16), 3); add(p + "ln1.bias", d[p + "ln1b"].astype(np.float16), 3)
    add(p + "th", d[p + "th"].astype(np.float16), 3);  add(p + "sc", d[p + "sc"].astype(np.float16), 3)
    q, s = tern_outscale(d[p + "Wm"].astype(np.float32))
    add(p + "Wm", np.ascontiguousarray(q.T), 2); add(p + "Wm.s", s, 3)
    add(p + "Wm.qs", q.astype(np.int32).sum(0).astype(np.int32), 0)
    add(p + "ln2", d[p + "ln2g"].astype(np.float16), 3); add(p + "ln2.bias", d[p + "ln2b"].astype(np.float16), 3)
    for nm in ("fc1", "fc2"):
        q, s = tern_outscale(d[p + nm].astype(np.float32))
        add(p + nm, np.ascontiguousarray(q.T), 2); add(p + nm + ".s", s, 3)
        add(p + nm + ".qs", q.astype(np.int32).sum(0).astype(np.int32), 0)   # Σ по входу для u8-офсета
    if p + "rw" in d:
        add(p + "rw", d[p + "rw"].astype(np.float16), 3)
        add(p + "rb", np.asarray(d[p + "rb"], np.float32).reshape(1).astype(np.float16), 3)

SHARE8 = "--share8" in sys.argv
if SHARE8:
    Ef = d["E"].astype(np.float32)
    es = (np.abs(Ef).max(1).clip(1e-8) / 127.0).astype(np.float16)          # per-row scale
    qE = np.clip(np.rint(Ef / es.astype(np.float32)[:, None]), -127, 127).astype(np.int8)
    add("E", np.ascontiguousarray(qE), 2)                                   # (V,D) int8
    add("E.s", es, 3)
    add("E.qs", qE.astype(np.int32).sum(1).astype(np.int32), 0)             # для u8-офсета головы
else:
    add("E", d["E"].astype(np.float16), 3)
add("pos", d["pos"].astype(np.float16), 3)
if os.environ.get("HEAD4"):
    Wt = d["E"].astype(np.float32).T
    sh = np.abs(Wt).max(0).clip(1e-6)
    q4 = np.clip(np.rint(Wt / sh[None, :] * 15), -7, 7).astype(np.int8)   # (D,V)
    qT = np.ascontiguousarray(q4.T.copy())                                # (V,D)
    nib = (qT.astype(np.int16) + 8).astype(np.uint8)
    pack = (nib[:, 0::2] | (nib[:, 1::2] << 4)).copy()                    # (V, D/2)
    add("Et4", pack, 4)
    add("Et4.s", (sh / 15).astype(np.float16), 3)
    add("Et4.qsum", qT.astype(np.int32).sum(1).astype(np.int32), 0)
elif SHARE8:                              # EtT не нужна: голова читает строки int8-E
    pass
elif True or "--i8head" in sys.argv:
    Wt = d["E"].astype(np.float32).T                      # (D,V)
    sh = np.abs(Wt).max(0).clip(1e-6)                     # per-out absmax
    q8 = np.clip(np.rint(Wt / sh[None, :] * 127), -127, 127).astype(np.int8)
    qT = np.ascontiguousarray(q8.T.copy())                # (V,D) row-major для VNNI-стиля
    add("EtT", qT, 2)                                     # транспонированная голова
    add("EtT.s", (sh / 127).astype(np.float16), 3)
    add("EtT.qsum", q8.T.astype(np.int32).sum(1).astype(np.int32), 0)   # Σ_d q[v,d] для офсета
else:
    add("Et", d["E"].astype(np.float16).T.copy(), 3)      # голова fp16 (D,V)
add("lnf", d["lnfg"].astype(np.float16), 3); add("lnf.bias", d["lnfb"].astype(np.float16), 3)

with open(DST, "wb") as f:
    f.write(b"LCW2")
    f.write(struct.pack("<5If", L, D, 0, V, T, TAU))
    f.write(struct.pack("<I", len(recs)))
    for name, dt, arr in recs:
        f.write(struct.pack("<Q", len(name))); f.write(name)
        f.write(struct.pack("<BI", dt, arr.ndim))
        f.write(struct.pack(f"<{arr.ndim}I", *arr.shape))
        f.write(arr.tobytes())  # payload бинарно по dtype-размеру dt
print(f"wrote {DST}: {os.path.getsize(DST):,} bytes, {len(recs)} recs, L={L} D={D} V={V} T={T} tau={TAU}")
