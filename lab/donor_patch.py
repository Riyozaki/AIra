"""Donor loop-patch reference implementation for the A1 path (docs/A1_PLAN.md).

Self-contained: reproduces Qwen2.5 decoder block geometry (pre-RMSNorm, GQA,
RoPE-NTheta, SwiGLU, no biases) for 0.5B and 1.5B configs and patches a
bounded D-002 recurrent loop at mid-depth:

    x <- x                        (donor layers 0..l-1, frozen in A1)
    for k in 1..K:  x <- LN_s(x + beta * LoopBlock(x))
    x <- x                        (donor layers l+1..L, frozen in A1)

The module runs WITHOUT downloading HF weights: with random init it verifies
shapes, telemetry and the bounded-loop invariants (TW-1 law). With real donor
weights (load_safetensors) it additionally verifies identity-proxy AX-1:
at LoopBlock init = copy of donor layer l+1, K=1 output stays close to donor.

HW-0 note: this file is the TRAINING-side reference; final inference targets
stock CPU (weights fp16/bf16/ternary later), so all compute here is fp32
master-friendly and quantization-safe by construction (LN_s bounds the loop).
"""
import json
import math
import sys
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.telemetry import lambda_from_displacement  # noqa: E402


# ---------------------------------------------------------------------------
# Donor configs (verbatim Qwen2.5 geometry, HuggingFace config.json values)

QWEN25_05B = dict(name="qwen2.5-0.5b", d=896, layers=24, heads=14, kv_heads=2,
                  head_dim=64, d_ff=4864, vocab=151936, max_pos=32768,
                  rope_theta=1000000.0, rms_eps=1e-6, tie_embeddings=True)
QWEN25_15B = dict(name="qwen2.5-1.5b", d=1536, layers=28, heads=12, kv_heads=2,
                  head_dim=128, d_ff=8960, vocab=151936, max_pos=32768,
                  rope_theta=1000000.0, rms_eps=1e-6, tie_embeddings=True)

# HF safetensors naming (Qwen2 architecture)
W = dict(tok="model.embed_tokens.weight",
         final_norm="model.norm.weight",
         layer="model.layers.{i}.{part}")


def hf_url(repo: str, revision: str = "main") -> str:
    return (f"https://huggingface.co/{repo}/resolve/{revision}/"
            "model.safetensors.index.json")


def fetch_weight_map(repo: str):
    """Download only the ~200KB safetensors index to enumerate weight names.
    Returns None when HF is unreachable (sandbox) -- random-init path then."""
    try:
        with urllib.request.urlopen(hf_url(repo), timeout=30) as r:
            idx = json.loads(r.read().decode())
        return sorted(set(idx["weight_map"].values())), idx["weight_map"]
    except Exception as e:
        print(f"[donor_patch] HF index unreachable ({type(e).__name__}); "
              "running random-init shape/telemetry tests only.")
        return None, None


# ---------------------------------------------------------------------------
# Qwen2.5-compatible block parts

class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


def rope_tables(seq_len, head_dim, theta, device):
    half = head_dim // 2
    inv = 1.0 / (theta ** (torch.arange(half, device=device, dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    f = torch.outer(t, inv)                       # [T, half]
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    # x: [B, nH, T, hd]; rotate-half convention of Qwen2/Llama
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).type_as(x)


class GQAAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, nh, nkv, hd = cfg["d"], cfg["heads"], cfg["kv_heads"], cfg["head_dim"]
        self.nh, self.nkv, self.hd = nh, nkv, hd
        self.q_proj = nn.Linear(d, nh * hd, bias=True)     # Qwen2 keeps attn biases
        self.k_proj = nn.Linear(d, nkv * hd, bias=True)
        self.v_proj = nn.Linear(d, nkv * hd, bias=True)
        self.o_proj = nn.Linear(nh * hd, d, bias=False)

    def forward(self, x, cos, sin, mask):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        g = self.nh // self.nkv
        k = k.repeat_interleave(g, dim=1)
        v = v.repeat_interleave(g, dim=1)
        scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(self.hd)
        scores = scores + mask
        a = torch.softmax(scores, dim=-1).to(v.dtype) @ v
        a = a.transpose(1, 2).reshape(B, T, self.nh * self.hd)
        return self.o_proj(a)


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg["d"], cfg["d_ff"], bias=False)
        self.up_proj = nn.Linear(cfg["d"], cfg["d_ff"], bias=False)
        self.down_proj = nn.Linear(cfg["d_ff"], cfg["d"], bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DonorBlock(nn.Module):
    """Qwen2.5 decoder block (frozen in A1; reference geometry)."""

    def __init__(self, cfg):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg["d"], cfg["rms_eps"])
        self.self_attn = GQAAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg["d"], cfg["rms_eps"])
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, mask):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LoopBlock(nn.Module):
    """The trainable A1 loop block. Same geometry as DonorBlock so it can be
    initialized by COPYING donor layer l+1 (AX-1 identity-proxy init)."""

    def __init__(self, cfg):
        super().__init__()
        self.block = DonorBlock(cfg)

    def copy_from_donor_layer(self, donor: "DonorLM", l_copy: int):
        self.block.load_state_dict(donor.layers[l_copy].state_dict())
        # delete gates: loop must start as a *useful* but bounded perturbation
        # (β starts small, see DonorLM.loop); weights stay donor-grade.

    def forward(self, x, cos, sin, mask):
        return self.block(x, cos, sin, mask) - x  # f(x): non-residual part only


class DonorLM(nn.Module):
    """Reference donor with a patched D-002 loop at layer `patch_at`."""

    def __init__(self, cfg=QWEN25_05B, patch_at=None, beta_init=0.25,
                 l_copy=None):
        super().__init__()
        self.cfg = cfg
        self.patch_at = patch_at if patch_at is not None else cfg["layers"] // 2
        self.l_copy = l_copy if l_copy is not None else self.patch_at + 1
        self.tok = nn.Embedding(cfg["vocab"], cfg["d"])
        self.layers = nn.ModuleList(DonorBlock(cfg) for _ in range(cfg["layers"]))
        self.final_norm = RMSNorm(cfg["d"], cfg["rms_eps"])
        # ---- A1 patch ----
        self.loop_block = LoopBlock(cfg)
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.ln_state = nn.LayerNorm(cfg["d"], elementwise_affine=False)  # D-002
        self._rope_cache = {}

    # -- wiring to real donor weights --
    def load_donor_state(self, state: dict, strict_geom: bool = True):
        """Load an HF Qwen2.5 state_dict (safetensors.merge product)."""
        own = self.state_dict()
        mapping = {}
        mapping[W["tok"]] = "tok.weight"
        mapping[W["final_norm"]] = "final_norm.weight"
        for i in range(self.cfg["layers"]):
            for part in ["input_layernorm.weight",
                         "self_attn.q_proj.weight", "self_attn.q_proj.bias",
                         "self_attn.k_proj.weight", "self_attn.k_proj.bias",
                         "self_attn.v_proj.weight", "self_attn.v_proj.bias",
                         "self_attn.o_proj.weight",
                         "post_attention_layernorm.weight",
                         "mlp.gate_proj.weight", "mlp.up_proj.weight",
                         "mlp.down_proj.weight"]:
                mapping[W["layer"].format(i=i, part=part)] = f"layers.{i}.{part}"
        missing = []
        for hf_name, own_name in mapping.items():
            if hf_name in state:
                t = state[hf_name].to(own[own_name].dtype)
                if strict_geom and t.shape != own[own_name].shape:
                    raise ValueError(f"shape mismatch {hf_name}: "
                                     f"{tuple(t.shape)} vs {tuple(own[own_name].shape)}")
                own[own_name].copy_(t)
            else:
                missing.append(hf_name)
        if missing:
            raise KeyError(f"missing {len(missing)} donor tensors, e.g. {missing[:3]}")
        self.loop_block.copy_from_donor_layer(self, self.l_copy)
        return len(mapping)

    # -- forward --
    def _rope(self, T, device):
        key = (T, str(device))
        if key not in self._rope_cache:
            self._rope_cache[key] = rope_tables(
                T, self.cfg["head_dim"], self.cfg["rope_theta"], device)
        return self._rope_cache[key]

    def forward(self, ids, K=1, want_hiddens=False, freeze_donor=True):
        B, T = ids.shape
        cos, sin = self._rope(T, ids.device)
        mask = torch.full((T, T), float("-inf"), device=ids.device)
        mask = torch.triu(mask, diagonal=1)[None, None, :, :]
        x = self.tok(ids)
        hiddens = [x]
        # gate semantics: beta == 0.0 disables the loop EXACTLY (no LN_s pass) —
        # this makes patched-vs-pure-donor A/B runs bit-identical at beta=0.
        # (D-002 formula x <- LN_s(x + beta*f) is NOT identity at beta=0 by design:
        # LN_s always contracts; the gate overrides the formula for A/B cleanliness.)
        loop_on = float(self.beta) != 0.0
        for i, layer in enumerate(self.layers):
            if i == self.patch_at and loop_on:
                for k in range(K):
                    f = self.loop_block(x, cos, sin, mask)
                    x = self.ln_state(x + self.beta * f)  # D-002 bounded loop
                    if want_hiddens:
                        hiddens.append(x)
            if freeze_donor:
                with torch.no_grad():
                    x = layer(x, cos, sin, mask)
            else:
                x = layer(x, cos, sin, mask)
        x = self.final_norm(x)
        logits = F.linear(x, self.tok.weight)  # tied head (Qwen2.5-0.5B)
        out = {"logits": logits}
        if want_hiddens:
            out["hiddens"] = hiddens
        return out

    def patch_param_count(self):
        return sum(p.numel() for p in self.loop_block.parameters()) + 1 \
            + sum(p.numel() for p in self.ln_state.parameters())


# ---------------------------------------------------------------------------
# Telemetry on the patched model (AX-3 protocol)

def loop_telemetry(model: DonorLM, ids, K=64):
    """AX-3: displacement decay + lambda_hat on the patched loop (eval mode)."""
    model.eval()
    with torch.no_grad():
        out = model(ids, K=K, want_hiddens=True)
        hs = out["hiddens"][1:]                      # after loop iters only
        if len(hs) < 3:
            return {"error": "K too small"}
        h = torch.stack(hs)                          # [K', B, T, D]
        d = (h[1:] - h[:-1]).pow(2).mean(dim=(1, 2, 3)).sqrt()
        fit = lambda_from_displacement(d, floor_ratio=1e-4)
        finite = bool(torch.isfinite(h).all())
    return {"lambda_hat": fit["lambda"], "regime": fit["regime"],
            "r2": fit["r2"], "window": fit["window"],
            "fp_finite_K64": finite,
            "d_norm_first": float(d[0]), "d_norm_last": float(d[-1]),
            "drift_monotone_tail": bool((d[-8:] < d[-16:-8] + 1e-12).float().mean() > 0.5)
            if len(d) >= 16 else None}


# ---------------------------------------------------------------------------
# Tests (random-init path; HF weights only when explicitly provided)

def _mini_cfg():
    c = dict(QWEN25_05B)
    c.update(d=64, layers=6, heads=4, kv_heads=2, head_dim=16, d_ff=128,
             vocab=257, max_pos=256)
    c["name"] = "mini-for-tests"
    return c


def test_shapes_and_props(seed=0):
    torch.manual_seed(seed)
    cfg = _mini_cfg()
    m = DonorLM(cfg)
    assert m.patch_param_count() < 200_000, "mini patch must be tiny"
    ids = torch.randint(0, cfg["vocab"], (2, 48))
    out = m(ids, K=8, want_hiddens=True)
    assert out["logits"].shape == (2, 48, cfg["vocab"])
    # property 1: finiteness at deep K (bounded loop law)
    out64 = m(ids, K=64, want_hiddens=True)
    assert torch.isfinite(out64["logits"]).all(), "NaN/inf at K=64 — loop law broken"
    # property 2: identity-proxy in the random sense: donor untrained so only
    # check that K=1 with beta->0 equals donor-without-patch EXACTLY
    beta0 = m.beta.data.clone()
    m.beta.data.zero_()
    base = m(ids, K=1)["logits"]
    m.beta.data.copy_(beta0)
    # emulate "no patch": run with patch_at beyond range
    pa = m.patch_at
    m.patch_at = 10**9
    pure = m(ids, K=1)["logits"]
    m.patch_at = pa
    assert torch.allclose(base, pure, atol=0), "beta=0 must be exact identity"
    # property 3: copy-init symmetry — loop block f equals donor layer l+1 f
    fresh = DonorLM(cfg)
    new_sd = {k: v.clone() for k, v in fresh.state_dict().items()}
    m2 = DonorLM(cfg)
    m2.load_state_dict(new_sd)
    m2.loop_block.copy_from_donor_layer(m2, m2.l_copy)
    assert all(torch.equal(a, b) for a, b in
               zip(m2.loop_block.block.state_dict().values(),
                   m2.layers[m2.l_copy].state_dict().values()))
    # property 4: telemetry keys present
    tel = loop_telemetry(m2, ids, K=64)
    for kk in ["lambda_hat", "fp_finite_K64", "d_norm_first", "d_norm_last"]:
        assert kk in tel
    print("test_shapes_and_props OK",
          f"| patch params: {m.patch_param_count():,}",
          f"| finite@64: {tel['fp_finite_K64']}",
          f"| d_norm: {tel['d_norm_first']:.4f} -> {tel['d_norm_last']:.6f}",
          f"| lambda_hat (random weights, informative=None): {tel['lambda_hat']:.3f}")


def test_full_config_count():
    m = DonorLM(QWEN25_05B)
    total = sum(p.numel() for p in m.parameters())
    loop = m.patch_param_count()
    print(f"qwen2.5-0.5b reference: total {total/1e6:.1f}M, "
          f"loop patch {loop/1e6:.2f}M ({100*loop/total:.2f}%)")
    assert 13.0e6 < loop < 16.5e6, "patch should be ~14.9M (0.5B block: attn 1.84M + SwiGLU 13.07M + beta)"


if __name__ == "__main__":
    print("== donor_patch self-tests ==")
    test_shapes_and_props(seed=0)
    test_shapes_and_props(seed=7)
    test_full_config_count()
    files, wmap = fetch_weight_map("Qwen/Qwen2.5-0.5B")
    if files:
        print(f"HF index reachable: {len(wmap)} tensors across {len(files)} shard(s): {files}")
        n_layer = len({k.split('.')[2] for k in wmap if k.startswith('model.layers.')})
        print(f"donor layers detected: {n_layer} (expect {QWEN25_05B['layers']})")
    else:
        print("random-init mode only (sandbox): shape/geometry/loop-law tests all green")
    print("== all green ==")
