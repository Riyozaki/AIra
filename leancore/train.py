#!/usr/bin/env python3
import os, sys, json, time, math, argparse, resource
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from model import GPT, flops_per_token_fwd

def setup():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ["LD_LIBRARY_PATH"] = "/home/user/stublibs" + ":" + os.environ.get("LD_LIBRARY_PATH", "")
setup()

def load(tag_dir):
    tr = np.load(os.path.join(ROOT, "data/prep/train.npy")).astype(np.int64)
    va = np.load(os.path.join(ROOT, "data/prep/val.npy")).astype(np.int64)
    meta = json.load(open(os.path.join(ROOT, "data/prep/meta.json")))
    return tr, va, meta

def get_batch(ids, B, T, rng, easy_starts=None, p_easy=0.0):
    if easy_starts is not None and rng.random() < p_easy:
        st = easy_starts[rng.integers(0, len(easy_starts), size=B)]
    else:
        st = rng.integers(0, len(ids) - T - 1, size=B)
    x = np.stack([ids[s:s + T] for s in st])
    y = np.stack([ids[s + 1:s + T + 1] for s in st])
    return torch.from_numpy(x), torch.from_numpy(y)

@torch.no_grad()
def val_loss(model, va, B, T, iters=8):
    model.eval()
    rng = np.random.default_rng(1234)
    tot, n = 0.0, 0
    for _ in range(iters):
        x, y = get_batch(va, B, T, rng)
        with torch.autocast("cpu", enabled=False):
            logits = model(x)
            tot += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item(); n += 1
    model.train()
    return tot / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "lean"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--ctx", type=int, default=96)
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--kfrac", type=float, default=0.55)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--curriculum", type=float, default=0.0, help="prob of easy-bucket sampling during first 40%% steps")
    ap.add_argument("--init", default="", help="optional checkpoint to start from")
    ap.add_argument("--eval_every", type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(7); np.random.seed(7)
    tr, va, meta = load(ROOT)
    vocab = meta["vocab"]
    device = "cpu"
    model = GPT(vocab, d=args.d, L=args.L, ctx=args.ctx, lean=(args.variant == "lean"), k_frac=args.kfrac)
    if args.init:
        sd = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(sd, strict=False)
    nparams = model.n_params()

    # curriculum buckets: mean word length below median => "easy" lines' spans
    easy_starts = None
    if args.curriculum > 0:
        # cheap difficulty proxy available: use raw text line difficulty recomputed per position is overkill;
        # use token positions where local <unk> density is low (heuristic: token id != 2 fraction)
        win = 64
        unk = (tr[: len(tr) - args.ctx - 1] == 2).astype(np.float32)
        k = np.ones(win) / win
        dens = np.convolve(unk, k, mode="same")
        thr = np.quantile(dens, 0.5)
        easy_starts = np.where(dens <= thr)[0]
        easy_starts = easy_starts[easy_starts < len(tr) - args.ctx - 1].astype(np.int64)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    def lr_at(s):
        if s < args.warmup: return args.lr * (s + 1) / args.warmup
        p = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * p)))

    rng = np.random.default_rng(42)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    logf = open(os.path.join(ROOT, f"results/run_{args.tag}.jsonl"), "w")
    t0 = time.time(); tok_total = 0
    print(f"[{args.tag}] params={nparams:,} vocab={vocab} steps={args.steps} batch={args.batch}x{args.ctx}", flush=True)
    for step in range(args.steps):
        for g in opt.param_groups: g["lr"] = lr_at(step)
        p_easy = args.curriculum if step < int(0.4 * args.steps) else 0.0
        x, y = get_batch(tr, args.batch, args.ctx, rng, easy_starts, p_easy)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tok_total += x.numel()
        if step % args.eval_every == 0 or step == args.steps - 1:
            vl = val_loss(model, va, args.batch, args.ctx)
            el = time.time() - t0
            rec = dict(step=step, train_loss=round(loss.item(), 4), val_loss=round(vl, 4),
                       val_ppl=round(math.exp(vl), 2), wall=round(el, 1),
                       tok_s=round(tok_total / el, 1), tokens=tok_total, lr=lr_at(step))
            print(json.dumps(rec), flush=True)
            logf.write(json.dumps(rec) + "\n"); logf.flush()
    wall = time.time() - t0
    ckpt = os.path.join(ROOT, f"results/ckpt_{args.tag}.pt")
    torch.save(model.state_dict(), ckpt)
    fcfg = flops_per_token_fwd(nparams, args.d, args.L, 6, 3 * args.d, args.ctx,
                               lean=(args.variant == "lean"), k_frac=args.kfrac, vocab=vocab)
    summary = dict(tag=args.tag, variant=args.variant, params=nparams, steps=args.steps,
                   tokens=tok_total, wall_s=round(wall, 1), tok_s=round(tok_total / wall, 1),
                   peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                   flops_tok_fwd=round(fcfg), final_val_loss=rec["val_loss"], final_val_ppl=rec["val_ppl"])
    with open(os.path.join(ROOT, f"results/summary_{args.tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY " + json.dumps(summary), flush=True)

if __name__ == "__main__":
    main()
