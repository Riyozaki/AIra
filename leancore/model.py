#!/usr/bin/env python3
"""LeanCore models.

Baseline: standard pre-norm causal transformer (GPT-style), tied embeddings.
Lean:    same block/dims, but blocks 1..L-1 use Adaptive Depth Routing (ADR):
         a per-token router sends only top-k tokens through attention+MLP
         (Mixture-of-Depths style), the rest pass through untouched.
         Router prob multiplies the block output (keeps gradients honest).
"""
import math, torch, torch.nn as nn, torch.nn.functional as F

def ternarize_(w: torch.Tensor):
    """BitNet-1.58-style absmean STE quant view of a weight matrix."""
    s = w.abs().mean(dim=-1, keepdim=True).clamp_min(1e-5)
    u = w / s
    q = u.round().clamp(-1, 1)
    return s * (u + (q - u).detach())

class TernaryLinear(nn.Linear):
    quantize = True
    def forward(self, x):
        w = ternarize_(self.weight) if self.quantize else self.weight
        return F.linear(x, w, self.bias)

class MHA(nn.Module):
    def __init__(self, d, h, causal=True):
        super().__init__()
        assert d % h == 0
        self.h, self.hd = h, d // h
        self.causal = causal
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
    def forward(self, x, pos_mask=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.hd).transpose(1, 2)
        k = k.view(B, T, self.h, self.hd).transpose(1, 2)
        v = v.view(B, T, self.h, self.hd).transpose(1, 2)
        if self.causal:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
            att = att.masked_fill(mask, float("-inf"))
            a = F.softmax(att, dim=-1)
            y = a @ v
        else:
            y = F.scaled_dot_product_attention(q, k, v)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, D))

class MLP(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.fc1 = nn.Linear(d, ff, bias=False)
        self.fc2 = nn.Linear(ff, d, bias=False)
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))

class Block(nn.Module):
    def __init__(self, d, h, ff, routed=False, k_frac=0.55):
        super().__init__()
        self.routed, self.k_frac = routed, k_frac
        self.ln1 = nn.LayerNorm(d); self.attn = MHA(d, h)
        self.ln2 = nn.LayerNorm(d); self.mlp = MLP(d, ff)
        self.router = nn.Linear(d, 1, bias=True) if routed else None
        if routed:
            nn.init.normal_(self.router.weight, std=0.02)
            nn.init.constant_(self.router.bias, 1.5)  # start near "everyone passes"
        self.last_k = None
    def deltas(self, h):
        """returns (attn_delta, mlp_delta) with inner residuals handled."""
        d1 = self.attn(self.ln1(h))
        d2 = self.mlp(self.ln2(h + d1))
        return d1 + d2
    def forward(self, x):
        if not self.routed:
            return x + self.deltas(x)
        B, T, D = x.shape
        k = max(1, int(round(self.k_frac * T)))
        self.last_k = k
        scores = self.router(x).squeeze(-1)                    # (B,T)
        idx = scores.topk(k, dim=1).indices.sort(dim=1).values # (B,k) ascending
        sel = idx.unsqueeze(-1).expand(B, k, D)
        xs = torch.gather(x, 1, sel)
        g = torch.sigmoid(scores.gather(1, idx)).unsqueeze(-1) # (B,k,1)
        # block delta computed ONLY on selected tokens; causal order preserved (sorted idx)
        contrib = self.deltas(xs) * g
        x = x + torch.zeros_like(x).scatter_add(1, sel, contrib)
        return x

class GPT(nn.Module):
    def __init__(self, vocab, d=192, L=4, h=6, ff_mult=3, ctx=96, lean=False, k_frac=0.55):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([
            Block(d, h, ff_mult * d, routed=(lean and i > 0), k_frac=k_frac)
            for i in range(L)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight  # weight tying: -1 parameter matrix
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
    def forward(self, ids):
        B, T = ids.shape
        x = self.tok(ids) + self.pos.weight[:T].unsqueeze(0)
        for b in self.blocks: x = b(x)
        return self.head(self.lnf(x))
    def n_params(self):
        return sum(p.numel() for p in self.parameters())
    def routed_stats(self):
        out = []
        for b in self.blocks:
            if b.routed and b.last_k: out.append(b.last_k)
        return out

def flops_per_token_fwd(cfg_params, d, L, h, ff, T, lean=False, k_frac=0.55, vocab=8000):
    """analytic forward FLOPs/token (matmul-dominated), for reporting ratios."""
    emb = d                             # embedding lookup ~free
    def block_flops(particip):          # particip in [0,1]
        att_proj = 4 * d * d * particip
        att_scores = 2 * 2 * T * d * particip * particip
        mlp = 2 * d * ff * particip
        router = (2 * d) if lean and particip < 1 else 0
        return att_proj + att_scores + mlp + router
    dense0 = block_flops(1.0)
    if lean:
        return emb + dense0 + (L - 1) * block_flops(k_frac) + 2 * d * vocab
    return emb + L * block_flops(1.0) + 2 * d * vocab
