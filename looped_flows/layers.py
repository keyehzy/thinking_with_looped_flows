"""Building blocks shared by the denoiser: RMSNorm, SwiGLU, attention with RoPE,
and the token-mixing MLP used for Sudoku. Follows the TRM architecture."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def find_multiple(a: int, b: int) -> int:
    return (-(a // -b)) * b


def rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    var = x.pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(var + eps)).to(dtype)


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0) -> torch.Tensor:
    """Truncated normal initialisation with the variance correction used by JAX/TRM."""
    if std == 0:
        with torch.no_grad():
            tensor.zero_()
        return tensor
    sqrt2 = math.sqrt(2)
    a = math.erf(lower / sqrt2)
    b = math.erf(upper / sqrt2)
    z = (b - a) / 2
    c = (2 * math.pi) ** -0.5
    pdf_u = c * math.exp(-0.5 * lower**2)
    pdf_l = c * math.exp(-0.5 * upper**2)
    comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)
    with torch.no_grad():
        tensor.uniform_(a, b)
        tensor.erfinv_()
        tensor.mul_(sqrt2 * comp_std)
        tensor.clip_(lower * comp_std, upper * comp_std)
    return tensor


class CastedLinear(nn.Module):
    """Linear layer whose parameters live in float32 but compute in the activation dtype."""

    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(trunc_normal_init_(torch.empty(out_features, in_features), std=1.0 / (in_features**0.5)))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class CastedEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, dim: int, init_std: float):
        super().__init__()
        self.weight = nn.Parameter(trunc_normal_init_(torch.empty(num_embeddings, dim), std=init_std))

    def forward(self, idx: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return F.embedding(idx, self.weight.to(dtype))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_up = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos, self.sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [B, L, H, D]; cos/sin: [L, D]
    cos = cos.unsqueeze(-2).to(q.dtype)
    sin = sin.unsqueeze(-2).to(q.dtype)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv_proj = CastedLinear(hidden_size, 3 * hidden_size, bias=False)
        self.o_proj = CastedLinear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos_sin: tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor:
        B, L, _ = x.shape
        qkv = self.qkv_proj(x).view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary(q, k, cos[:L], sin[:L])
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False)
        return self.o_proj(out.transpose(1, 2).reshape(B, L, self.hidden_size))


class Block(nn.Module):
    """Post-norm block: token mixing (attention or transposed MLP) followed by a SwiGLU channel MLP."""

    def __init__(self, hidden_size: int, num_heads: int, expansion: float, seq_len: int, mixer: str, rms_norm_eps: float = 1e-5):
        super().__init__()
        self.mixer_type = mixer
        self.eps = rms_norm_eps
        if mixer == "attention":
            self.mixer = Attention(hidden_size, num_heads)
        elif mixer == "mlp":
            self.mixer = SwiGLU(hidden_size=seq_len, expansion=expansion)
        else:
            raise ValueError(f"unknown mixer {mixer}")
        self.mlp = SwiGLU(hidden_size=hidden_size, expansion=expansion)

    def forward(self, x: torch.Tensor, cos_sin) -> torch.Tensor:
        if self.mixer_type == "attention":
            x = rms_norm(x + self.mixer(x, cos_sin), self.eps)
        else:
            x = rms_norm(x + self.mixer(x.transpose(1, 2)).transpose(1, 2), self.eps)
        return rms_norm(x + self.mlp(x), self.eps)


class ReasoningModule(nn.Module):
    """The shared network F_theta: a stack of blocks applied to (state + injection)."""

    def __init__(self, layers: list[Block]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden: torch.Tensor, injection: torch.Tensor, cos_sin) -> torch.Tensor:
        hidden = hidden + injection
        for layer in self.layers:
            hidden = layer(hidden, cos_sin)
        return hidden
