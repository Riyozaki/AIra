#!/usr/bin/env python3
"""LeanCore quantization stage.

1) take a trained checkpoint;
2) swap every big Linear for TernaryLinear (weights -> {-1,0,+1} * row-scale, BitNet-1.58 style);
3) measure val PPL zero-shot (damage visible);
4) brief QAT (quantization-aware finetune, STE) to recover;
5) pack the ternary model to real 1.58-bit file and compare bytes vs fp32.
"""
import os, sys, json, time, math, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from model import GPT, TernaryLinear
from train import load, get_batch, val_loss

def convert_to_ternary(model):
    count = 0
    for m in model.modules():
        for name, child in list(m.named_children()):
            if isinstance(child, nn.Linear) and not isinstance(child, TernaryLinear) and child.weight.numel() > 4096:
                tl = TernaryLinear(child.in_features, child.out_features, bias=child.bias is not None)
                tl.weight = child.weight
                if child.bias is not None: tl.bias = child.bias
                tl.quantize = True
                setattr(m, name, tl); count += 1
    return count

@torch.no_grad()
def pack_ternary(model, path):
    """true bit-packing: sign plane + zero plane, 2 bits/param ceiling (1.58 theoretical)."""
    total_bits_158, total_bytes, fp32_bytes = 0, 0, 0
    with open(path, "wb") as f:
        for name, p in model.state_dict().items():
            arr = p.detach().cpu().numpy()
            fp32_bytes += arr.nbytes
            if "weight" in name and arr.ndim == 2 and arr.size > 4096 and not any(k in name for k in ("tok", "pos", "router")):
                w = torch.from_numpy(arr)
                s = w.abs().mean(dim=-1, keepdim=True).clamp_min(1e-5)
                q = (w / s).round().clamp(-1, 1).to(torch.int8).numpy()
                zero = (q == 0).astype(np.uint8); neg = (q == -1).astype(np.uint8)
                zb, nb = np.packbits(zero).tobytes(), np.packbits(neg).tobytes()
                sb = s.squeeze(-1).numpy().astype(np.float16).tobytes()
                hdr = np.array([arr.shape[0], arr.shape[1] if arr.ndim > 1 else 1, len(zb), len(nb), len(sb)], dtype=np.int64)
                f.write(hdr.tobytes()); f.write(zb); f.write(nb); f.write(sb)
                total_bytes += hdr.nbytes + len(zb) + len(nb) + len(sb)
                total_bits_158 += 1.58 * arr.size
            else:
                a16 = arr.astype(np.float16)
                f.write(a16.tobytes()); total_bytes += a16.nbytes
    disk = os.path.getsize(path)
    return fp32_bytes, disk, total_bits_158 / 8

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--qat_steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--ctx", type=int, default=96)
    args = ap.parse_args()
    tr, va, meta = load(ROOT)
    torch.manual_seed(11)
    model = GPT(meta["vocab"], d=192, L=4, ctx=args.ctx, lean=True)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False))
    ppl_fp32 = math.exp(val_loss(model, va, args.batch, args.ctx, iters=10))

    nconv = convert_to_ternary(model)
    model.head.weight = model.tok.weight  # re-tie untouched
    ppl_tern0 = math.exp(val_loss(model, va, args.batch, args.ctx, iters=10))

    # QAT finetune (only non-quantized params are cheap anyway; train all)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    rng = np.random.default_rng(5)
    t0 = time.time()
    for step in range(args.qat_steps):
        x, y = get_batch(tr, args.batch, args.ctx, rng)
        loss = F.cross_entropy(model(x).view(-1, meta["vocab"]), y.view(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 50 == 0 or step == args.qat_steps - 1:
            vl = val_loss(model, va, args.batch, args.ctx)
            print(json.dumps(dict(step=step, qat_train=round(loss.item(), 4), qat_val_ppl=round(math.exp(vl), 2), wall=round(time.time() - t0, 1))), flush=True)
    vl = val_loss(model, va, args.batch, args.ctx, iters=10)
    ppl_qat = math.exp(vl)

    out = os.path.join(ROOT, "results/lean_ternary.lc15")
    fp32b, ternb, ideal158 = pack_ternary(model, out)
    torch.save(model.state_dict(), os.path.join(ROOT, "results/ckpt_lean_ternary.pt"))
    rep = dict(ppl_fp32_lean=round(ppl_fp32, 2), ppl_ternary_zeroshot=round(ppl_tern0, 2),
               ppl_ternary_qat=round(ppl_qat, 2), converted_linears=nconv,
               fp32_ckpt_bytes=fp32b, ternary_packed_bytes=ternb,
               ternary_ideal_158b_bytes=int(ideal158),
               compression_vs_fp32=round(fp32b / ternb, 1))
    print("QUANT " + json.dumps(rep, indent=2), flush=True)
    with open(os.path.join(ROOT, "results/quant_report.json"), "w") as f:
        json.dump(rep, f, indent=2)

if __name__ == "__main__":
    main()
