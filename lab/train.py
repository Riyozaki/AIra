"""Shared training loop for T-week runs: deep supervision on all iterations."""
import time
import numpy as np
import torch
import torch.nn.functional as F


def eval_per_k(model, src, K: int, n_batches: int = 8, B: int = 32, T: int = 192,
               hard_window: int = 1, device: str = "cpu"):
    """Val CE per supervision step k, split hard/steady; plus hiddens for telemetry."""
    model.eval()
    losses, losses_hard, losses_steady = [], [], []
    all_hiddens, slow_states = [], []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y, hard = src.batch(B, T)
            out = model(x.to(device), K)
            logits = out["logits"]  # [K,B,T,V]
            lk = []
            for k in range(K):
                ce = F.cross_entropy(
                    logits[k].reshape(-1, logits.shape[-1]).float(),
                    y.reshape(-1).to(device), reduction="none").reshape(B, T)
                lc = ce.mean()
                lk.append(lc)
                losses_hard.append((k, ce[hard].mean().item() if hard.any() else float("nan")))
                losses_steady.append((k, ce[~hard].mean().item()))
            losses.append(torch.stack(lk))
            all_hiddens.append(out["hiddens"].cpu())
            if "slow_states" in out:
                slow_states.append(out["slow_states"].cpu())
    L = torch.stack(losses).mean(dim=0)  # [K]
    hb = torch.cat(all_hiddens, dim=1)
    res = {"L_per_k": L.tolist(), "hiddens": hb}
    hl = np.full(K, np.nan)
    sl = np.full(K, np.nan)
    for k, v in losses_hard:
        hl[k] = v if not np.isnan(v) else hl[k]
    # proper aggregation
    hard_agg = {}
    steady_agg = {}
    for k, v in losses_hard:
        hard_agg.setdefault(k, []).append(v)
    for k, v in losses_steady:
        steady_agg.setdefault(k, []).append(v)
    res["L_hard"] = [float(np.nanmean(hard_agg.get(k, [np.nan]))) for k in range(K)]
    res["L_steady"] = [float(np.mean(steady_agg[k])) for k in range(K)]
    if slow_states:
        res["slow_states"] = torch.cat(slow_states, dim=1)
    return res


def train_model(model, src, steps: int, B: int, T: int, K_min: int, K_max: int,
                lr: float = 3e-3, warmup: int = 100, device: str = "cpu",
                log_every: int = 500, seed: int = 0):
    torch.manual_seed(seed)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed + 1)
    t0 = time.time()
    hist = []
    for step in range(steps):
        x, y, _ = src.batch(B, T)
        x, y = x.to(device), y.to(device)
        K = int(torch.randint(K_min, K_max + 1, (1,), generator=g))
        out = model(x, K)
        logits = out["logits"]
        loss = torch.stack([
            F.cross_entropy(logits[k].reshape(-1, logits.shape[-1]).float(), y.reshape(-1))
            for k in range(K)]).mean()
        for pg in opt.param_groups:
            pg["lr"] = lr * min(1.0, (step + 1) / warmup) * \
                (0.5 * (1 + np.cos(np.pi * step / max(steps, 1))))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % log_every == 0 or step == 0:
            el = time.time() - t0
            hist.append({"step": step + 1, "loss": float(loss), "K": K,
                         "sec_per_step": el / (step + 1)})
            print(f"  step {step+1}/{steps} loss {float(loss):.4f} K={K} "
                  f"({el/(step+1):.3f}s/step)", flush=True)
    return hist
