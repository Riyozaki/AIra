"""Lambda telemetry: three independent estimators of the contraction factor.

1. displacement (trajectory): lambda from d_k = RMS(h_{k+1} - h_k)
2. perturbation (local Lipschitz): ||R(h+e) - R(h)|| / ||e||
3. Jacobian spectral radius via power iteration over JVP

Operationalizations frozen in docs/TWEEK.md before any run.
"""
import math
import numpy as np
import torch


def displacement_curve(hiddens: torch.Tensor) -> torch.Tensor:
    """hiddens [K+1, B, T, D] -> d_k [K], RMS over batch/time/dim."""
    return (hiddens[1:] - hiddens[:-1]).pow(2).mean(dim=(1, 2, 3)).sqrt()


def displacement_curve_per_batch(hiddens: torch.Tensor) -> torch.Tensor:
    """-> [K, B] RMS over time/dim per sequence."""
    return (hiddens[1:] - hiddens[:-1]).pow(2).mean(dim=(2, 3)).sqrt()


def fit_log_linear(ks: np.ndarray, ys: np.ndarray):
    """Fit log y = a + slope*k on points with y>0. Returns (slope, intercept, r2, n)."""
    mask = ys > 0
    k, ly = ks[mask].astype(float), np.log(ys[mask])
    if len(k) < 3:
        return None
    A = np.vstack([k, np.ones_like(k)]).T
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    pred = A @ coef
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(coef[0]), float(coef[1]), r2, len(k)


def lambda_from_displacement(d: torch.Tensor, floor_ratio: float = 1e-3):
    """d [K] -> lambda fit with automatic regime detection.

    Maps can be expansive (d_k rising, e.g. random init residual map ~|I+f'|)
    or contractive (d_k decaying after trained/test-time convergence).
    Fits log-linear on both segments around argmax and reports the regime:
    'contraction' (decay fit dominates), 'expansion' (rise dominates), 'flat'.
    """
    d = d.detach().cpu().numpy().astype(np.float64)
    d[~np.isfinite(d)] = np.nan
    idx = np.arange(len(d))
    valid = np.isfinite(d) & (d > 0)
    if valid.sum() < 6:
        return {"lambda": float("nan"), "regime": "flat", "r2": float("nan"),
                "window": None, "n_points": 0, "d": d.tolist()}
    dd = np.where(valid, d, np.nanmax(d[valid]))
    kmax = int(np.nanargmax(dd))
    floor = float(np.nanmax(dd)) * floor_ratio

    dec_mask = valid & (idx >= max(kmax, 1)) & (d > floor)
    ris_mask = valid & (idx >= 1) & (idx <= kmax)
    fits = {}
    if dec_mask.sum() >= 4:
        fits["contraction"] = fit_log_linear(idx[dec_mask], d[dec_mask])
    if ris_mask.sum() >= 4:
        fits["expansion"] = fit_log_linear(idx[ris_mask], d[ris_mask])
    regime, fit, win = None, None, None
    if "contraction" in fits and fits["contraction"] and kmax <= len(d) - 5:
        regime, fit, win = "contraction", fits["contraction"], dec_mask
    elif "expansion" in fits and fits["expansion"]:
        regime, fit, win = "expansion", fits["expansion"], ris_mask
    elif "contraction" in fits and fits["contraction"]:
        regime, fit, win = "contraction", fits["contraction"], dec_mask
    if fit is None:
        return {"lambda": float("nan"), "regime": "flat", "r2": float("nan"),
                "window": None, "n_points": 0, "d": d.tolist()}
    slope, intercept, r2, n = fit
    widx = idx[win]
    return {"lambda": math.exp(slope), "slope": slope, "regime": regime,
            "r2": r2, "window": [int(widx[0]), int(widx[-1])], "n_points": n,
            "d": d.tolist()}


def angular_curve_per_seq(hiddens: torch.Tensor) -> torch.Tensor:
    """Direction displacement on the unit sphere: ||u_{k+1} - u_k|| / sqrt(2).

    hiddens [K+1, B, T, D] -> [K, B] (mean over tokens). In [0, 1].
    """
    u = torch.nn.functional.normalize(hiddens, dim=-1)
    d = (u[1:] - u[:-1]).pow(2).sum(-1).sqrt() / math.sqrt(2.0)  # [K, B, T]
    return d.mean(-1)


def per_token_lambda(hiddens: torch.Tensor, k_lo: int = 2, k_hi: int = 20,
                     angular: bool = False):
    """Per-token slope of log d_k on fixed window, vectorized.

    hiddens [K+1, B, T, D]. angular=True: work on unit-normalized states.
    Returns distribution stats over tokens.
    """
    if angular:
        u = torch.nn.functional.normalize(hiddens, dim=-1)
        d = (u[1:] - u[:-1]).pow(2).sum(-1).sqrt() / math.sqrt(2.0)
    else:
        d = (hiddens[1:] - hiddens[:-1]).pow(2).mean(dim=-1).sqrt()  # [K, B, T]
    K = d.shape[0]
    k_hi = min(k_hi, K)
    ks = np.arange(k_lo, k_hi, dtype=np.float64)
    ld = np.log(d[k_lo:k_hi].detach().cpu().numpy().astype(np.float64) + 1e-12)  # [w, B, T]
    kc = ks - ks.mean()
    denom = float((kc ** 2).sum())
    slope = (ld * kc[:, None, None]).sum(axis=0) / denom  # [B, T]
    # per-token r2
    pred = ld.mean(axis=0, keepdims=True) + slope[None, :, :] * kc[:, None, None]
    ss_res = ((ld - pred) ** 2).sum(axis=0)
    ss_tot = ((ld - ld.mean(axis=0, keepdims=True)) ** 2).sum(axis=0) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    lam = np.exp(slope).reshape(-1)
    r2f = r2.reshape(-1)
    return {"median": float(np.median(lam)), "q25": float(np.percentile(lam, 25)),
            "q75": float(np.percentile(lam, 75)),
            "r2_median": float(np.median(r2f)),
            "q_ratio": float(np.percentile(lam, 75) / max(np.percentile(lam, 25), 1e-9))}


def perturbation_lambda(step_fn, h: torch.Tensor, r: float = 0.01,
                        n_dirs: int = 8, seed: int = 0) -> float:
    """Local Lipschitz of one core application at h. ||R(h+e)-R(h)||/||e||."""
    g = torch.Generator().manual_seed(seed)
    h = h.detach()
    with torch.no_grad():
        base = step_fn(h)
        qs = []
        for _ in range(n_dirs):
            e = torch.randn(h.shape, generator=g)
            e = e * (r * h.norm() / (e.norm() + 1e-12))
            q = (step_fn(h + e) - base).norm() / e.norm()
            qs.append(float(q))
    return float(np.mean(qs))


def jacobian_spectral_radius(step_fn, h: torch.Tensor, iters: int = 12,
                             seed: int = 0, tol: float = 1e-4) -> float:
    """Power iteration on the Jacobian of step_fn at h via JVP."""
    g = torch.Generator().manual_seed(seed)
    h = h.detach()
    v = torch.randn(h.shape, generator=g)
    v = v / (v.norm() + 1e-12)
    prev = 0.0
    lam = 0.0
    for _ in range(iters):
        _, w = torch.autograd.functional.jvp(step_fn, h, v)
        n = float(w.norm())
        if n < 1e-12:
            break
        lam = n
        v = w / n
        if abs(lam - prev) < tol * max(1.0, abs(lam)):
            break
        prev = lam
    return float(lam)


def geometric_floor_fit(ks: np.ndarray, ys: np.ndarray, n_grid: int = 60):
    """Fit L(k) = floor + C * lam^k by profiling over floor; least squares in log space.

    Returns dict(floor, C, lambda, r2) with R^2 computed in original space.
    """
    ks = ks.astype(np.float64)
    ys = ys.astype(np.float64)
    y_min, y_max = float(ys.min()), float(ys.max())
    span = max(y_max - y_min, 1e-9)
    best = None
    floors = np.linspace(y_min - 0.5 * span, y_min - 1e-6 * span, n_grid)
    for f in floors:
        r = ys - f
        if (r <= 0).any():
            continue
        A = np.vstack([ks, np.ones_like(ks)]).T
        coef, *_ = np.linalg.lstsq(A, np.log(r), rcond=None)
        pred = f + np.exp(A @ coef)
        sse = float(((ys - pred) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, f, float(np.exp(coef[1])), float(np.exp(coef[0])))
    if best is None:
        return None
    sse, f, C, lam = best
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    r2 = 1.0 - sse / ss_tot if ss_tot > 0 else 1.0
    return {"floor": f, "C": C, "lambda": lam, "r2": r2, "sse": sse}


def power_fit(ks: np.ndarray, ys: np.ndarray):
    """Pure power law y = a * k^-c (no floor). Parcae-world reference."""
    ks = ks.astype(np.float64)
    ys = ys.astype(np.float64)
    mask = ys > 0
    A = np.vstack([np.log(ks[mask]), np.ones(mask.sum())]).T
    coef, *_ = np.linalg.lstsq(A, np.log(ys[mask]), rcond=None)
    pred = np.exp(A @ coef)
    sse = float(((np.log(ys[mask]) - pred) ** 2).sum())
    pred_y = np.exp(np.log(ks[mask]) * coef[0] + coef[1])
    ss_tot = float(((ys[mask] - ys[mask].mean()) ** 2).sum())
    r2 = 1.0 - float(((ys[mask] - pred_y) ** 2).sum()) / ss_tot if ss_tot > 0 else 1.0
    return {"a": float(np.exp(coef[1])), "c": float(-coef[0]), "r2": r2,
            "sse_log": sse}


def aicc(sse: float, n: int, p: int) -> float:
    """Corrected Akaike criterion, gaussian errors."""
    if sse <= 0 or n <= p + 1:
        return float("inf")
    base = n * math.log(sse / n) + 2 * p
    return base + (2 * p * (p + 1)) / (n - p - 1)
