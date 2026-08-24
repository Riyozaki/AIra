"""Tiny recurrent-depth cores for THEORY verification (T-week stand).

HW-0 compliant: pure CPU torch. Shared-weight core iterated K times.
Variants: depth-embedding on/off, residual step beta, two-timescale core.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedCore(nn.Module):
    """One shared transformer layer: h <- h + beta * f(h).

    f = causal self-attention then MLP, both pre-norm residual.
    depth_emb=True adds a per-iteration embedding (non-autonomous map R_k).
    """

    def __init__(self, d: int, n_head: int, d_ff: int, max_loops: int,
                 beta: float = 1.0, depth_emb: bool = False):
        super().__init__()
        self.d = d
        self.beta = beta
        self.n_head = n_head
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        self.depth_emb = nn.Embedding(max_loops, d) if depth_emb else None

    def f(self, h: torch.Tensor, k: int = 0) -> torch.Tensor:
        """Non-residual part f(h). h: [B, T, D]."""
        x = h
        if self.depth_emb is not None:
            x = x + self.depth_emb.weight[k].view(1, 1, -1)
        B, T, D = x.shape
        u = self.ln1(x)
        q, k_, v = self.qkv(u).chunk(3, dim=-1)
        hd = D // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k_ = k_.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        # manual attention: keeps double-backward (JVP) working on CPU
        scores = q @ k_.transpose(-2, -1) / math.sqrt(hd)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        scores = scores.masked_fill(mask, float("-inf"))
        a = torch.softmax(scores, dim=-1) @ v
        a = a.transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(a)
        x = x + self.mlp(self.ln2(x))
        return x

    def forward(self, h: torch.Tensor, k: int = 0) -> torch.Tensor:
        return h + self.beta * self.f(h, k)


class TwoTimeCore(nn.Module):
    """Fast shared layer every step + slow block every m steps.

    Slow state s: [B, D] (per-sequence). Update at k % m == 0:
        s <- s + beta_s * SlowMLP([s, pool(LN_fast(h))])
    Slow state injected into h every step via FiLM: h <- h * (1 + g(s)) + b(s).
    """

    def __init__(self, d: int, n_head: int, d_ff: int, m: int = 8,
                 beta: float = 1.0, beta_s: float = 1.0, use_slow: bool = True,
                 slow_ln: bool = False):
        super().__init__()
        self.m = m
        self.beta = beta
        self.beta_s = beta_s
        self.use_slow = use_slow
        self.fast = SharedCore(d, n_head, d_ff, max_loops=1, beta=1.0, depth_emb=False)
        self.ln_pool = nn.LayerNorm(d)
        self.slow = nn.Sequential(nn.Linear(2 * d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        # R3 (TWEEK round 3): every recurrent loop must be bounded -- slow state too
        self.ln_s = nn.LayerNorm(d, elementwise_affine=False) if slow_ln else None
        self.film = nn.Linear(d, 2 * d)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, h: torch.Tensor, k: int, state: dict) -> torch.Tensor:
        if self.use_slow and state is not None:
            if k % self.m == 0:
                pooled = self.ln_pool(h).mean(dim=1)  # [B, D]
                s = state["s"]
                s = s + self.beta_s * self.slow(torch.cat([s, pooled], dim=-1))
                if self.ln_s is not None:
                    s = self.ln_s(s)
                state["s"] = s
                state["s_updates"] += 1
            g, b = self.film(state["s"]).unsqueeze(1).chunk(2, dim=-1)
            h = h * (1.0 + g) + b
        return h + self.beta * (self.fast.f(h) - h)


class TinyLoopLM(nn.Module):
    """Embedding + shared core + tied head. Returns per-iteration logits + hiddens."""

    def __init__(self, vocab: int, d: int = 128, n_head: int = 4, d_ff: int = 512,
                 max_len: int = 256, max_loops: int = 64, beta: float = 1.0,
                 depth_emb: bool = False, two_time: bool = False,
                 slow_m: int = 8, use_slow: bool = True, state_ln: bool = False):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.ln_out = nn.LayerNorm(d)
        self.ln_state = nn.LayerNorm(d, elementwise_affine=False) if state_ln else None
        if two_time:
            self.core = TwoTimeCore(d, n_head, d_ff, m=slow_m, beta=beta, use_slow=use_slow)
        else:
            self.core = SharedCore(d, n_head, d_ff, max_loops, beta=beta, depth_emb=depth_emb)
        self.two_time = two_time
        self.d = d

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        return self.tok(x) + self.pos.weight[:T].unsqueeze(0)

    def head(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(self.ln_out(h), self.tok.weight)

    def forward(self, x: torch.Tensor, K: int, want_hiddens: bool = False):
        h = self.embed(x)
        state = {"s": torch.zeros(x.shape[0], self.d), "s_updates": 0} if self.two_time else None
        logits, hiddens, states = [], [h], []
        for k in range(K):
            if self.two_time:
                h = self.core(h, k, state)
                states.append(state["s"].detach().clone())
            else:
                h = self.core(h, k)
            if self.ln_state is not None:
                h = self.ln_state(h)
            logits.append(self.head(h))
            hiddens.append(h)
        out = {"logits": torch.stack(logits), "hiddens": torch.stack(hiddens)}
        if self.two_time:
            out["slow_states"] = torch.stack(states)  # [K, B, D]
        return out


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
