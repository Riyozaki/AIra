#!/usr/bin/env python3
"""Eval + generation demo for LeanCore checkpoints."""
import os, sys, json, math, time, argparse
import numpy as np, torch, torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from model import GPT
from quant import convert_to_ternary
from train import load, val_loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ternary", action="store_true")
    ap.add_argument("--lean", action="store_true")
    ap.add_argument("--prompt", default="the king")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--bench_tokens", type=int, default=200)
    args = ap.parse_args()

    tr, va, meta = load(ROOT)
    stoi, itos = meta["stoi"], meta["itos"]
    model = GPT(meta["vocab"], d=192, L=4, ctx=96, lean=args.lean)
    if args.ternary:
        convert_to_ternary(model)
        model.head.weight = model.tok.weight
    missing, unexpected = model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False), strict=False)
    model.eval()

    import re
    def enc(s):
        toks = re.findall(r"[a-z']+|[0-9]+|[^\s\w]", s.lower())
        return [stoi.get(t, 2) for t in toks] or [1]
    def dec(ids):
        out, prev = [], None
        for i in ids:
            w = itos.get(str(i), None) or itos.get(i, "<unk>")
            if prev is not None and w.isalpha() and (prev or "").isalnum(): out.append(" ")
            elif prev is not None and w not in ".,!?;:'\"-() " and w.isalpha(): out.append(" ")
            out.append(w); prev = w
        return "".join(out)

    ppl = math.exp(val_loss(model, va, 24, 96, iters=10))
    torch.manual_seed(3)
    ids = enc(args.prompt)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(args.bench_tokens):
            x = torch.tensor([ids[-96:]])
            logits = model(x)[0, -1]
            probs = F.softmax(logits / args.temp, dim=-1)
            top = torch.topk(probs, 40)
            nxt = top.indices[torch.multinomial(top.values, 1)].item()
            ids.append(nxt)
    dt = time.time() - t0
    gen = ids[-args.bench_tokens:]
    print(json.dumps(dict(ckpt=os.path.basename(args.ckpt), ternary=args.ternary,
                          val_ppl=round(ppl, 2),
                          gen_tok_s=round(args.bench_tokens / dt, 1))))
    print("--- SAMPLE (" + args.prompt + " ...) ---")
    print(dec(enc(args.prompt) + gen[:args.n]).capitalize())

if __name__ == "__main__":
    main()
