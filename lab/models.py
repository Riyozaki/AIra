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

    def f(self, h: torch.Tensor, k: int = 0,
            block_mask: torch.Tensor = None) -> torch.Tensor:
        """Non-residual part f(h). h: [B, T, D].
        block_mask: optional extra bool [T,T] (True = block attention), ORed
        with the causal mask. Used by CDEMO R5 (slots must not read slots)."""
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
        if block_mask is not None:
            mask = mask | block_mask
        scores = scores.masked_fill(mask, float("-inf"))
        a = torch.softmax(scores, dim=-1) @ v
        a = a.transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(a)
        x = x + self.mlp(self.ln2(x))
        return x

    def forward(self, h: torch.Tensor, k: int = 0,
                block_mask: torch.Tensor = None) -> torch.Tensor:
        return h + self.beta * self.f(h, k, block_mask=block_mask)


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
                 slow_m: int = 8, use_slow: bool = True, state_ln: bool = False,
                 slow_ln: bool = False, inject: bool = False,
                 board9: bool = False):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        # G0S S-11: structural row/col/box embeddings for 9x9 board layout
        self.board9 = board9
        if board9:
            self.e_row = nn.Embedding(9, d)
            self.e_col = nn.Embedding(9, d)
            self.e_box = nn.Embedding(9, d)
        self.ln_out = nn.LayerNorm(d)
        self.ln_state = nn.LayerNorm(d, elementwise_affine=False) if state_ln else None
        # kernel v3 (G0S S-10): per-loop input injection with learnable per-channel gain
        self.e_proj = nn.Linear(d, d, bias=False) if inject else None
        self.e_gain = nn.Parameter(torch.full((d,), 0.1)) if inject else None
        if two_time:
            self.core = TwoTimeCore(d, n_head, d_ff, m=slow_m, beta=beta,
                                    use_slow=use_slow, slow_ln=slow_ln)
        else:
            self.core = SharedCore(d, n_head, d_ff, max_loops, beta=beta, depth_emb=depth_emb)
        self.two_time = two_time
        self.d = d

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        h = self.tok(x) + self.pos.weight[:T].unsqueeze(0)
        if self.board9 and T >= 2 * 81:
            # seq = [grid 0..80, SEP 81, sol 82..162]; x = seq[:-1] covers 0..161:
            # positions 0..80 -> grid cells; 81 -> SEP (no emb); 82..161 -> sol cells 0..79
            emb = torch.zeros_like(h)
            rows = torch.arange(81, device=x.device) // 9
            cols = torch.arange(81, device=x.device) % 9
            box = (rows // 3) * 3 + cols // 3
            cell = self.e_row(rows) + self.e_col(cols) + self.e_box(box)
            emb[:, 0:81] = cell
            n_ans = T - 82
            emb[:, 82:82 + n_ans] = cell[:n_ans]
            h = h + emb
        return h

    def head(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(self.ln_out(h), self.tok.weight)

    def forward(self, x: torch.Tensor, K: int, want_hiddens: bool = False,
                block_mask: torch.Tensor = None):
        h = self.embed(x)
        e_inj = (self.e_gain * self.e_proj(h)) if self.e_proj is not None else None
        state = {"s": torch.zeros(x.shape[0], self.d), "s_updates": 0} if self.two_time else None
        logits, hiddens, states = [], [h], []
        for k in range(K):
            if self.two_time:
                h = self.core(h, k, state)
                states.append(state["s"].detach().clone())
            else:
                h = self.core(h, k, block_mask=block_mask)
            if e_inj is not None:
                h = h + e_inj
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
