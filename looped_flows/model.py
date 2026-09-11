"""Stateful denoiser D_t(x_t, z; c) -> (x_hat, z') built on the TRM architecture.

One denoiser call runs the TRM recurrence for a single supervision step:

    l <- F(l + h + e_t)   (m times)
    h <- F(h + l)

repeated ``H_cycles`` times, with gradients only through the last cycle. The
recurrent state is z = (h, l); the prediction is decoded from h and the ACT
halting score q from the first (puzzle-embedding) position of h.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .layers import Block, CastedEmbedding, CastedLinear, ReasoningModule, RotaryEmbedding, trunc_normal_init_


@dataclass
class DenoiserConfig:
    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int = 1
    problem_tokens: bool = False  # embed the problem c separately (ARC, graph colouring)
    hidden_size: int = 512
    num_heads: int = 8
    expansion: float = 4.0
    num_layers: int = 2
    H_cycles: int = 3
    L_cycles: int = 4
    mixer: str = "attention"  # "attention" or "mlp"
    puzzle_emb_ndim: int = 512
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    time_conditioning: bool = True
    forward_dtype: str = "bfloat16"

    @property
    def puzzle_emb_len(self) -> int:
        return -(self.puzzle_emb_ndim // -self.hidden_size) if self.puzzle_emb_ndim > 0 else 0

    @property
    def total_len(self) -> int:
        return self.seq_len + self.puzzle_emb_len


@dataclass
class DenoiserOutput:
    logits: torch.Tensor  # [B, L, V] float32
    q_logit: torch.Tensor  # [B] float32
    h: torch.Tensor  # detached recurrent states
    l: torch.Tensor
    extra: dict = field(default_factory=dict)


class LoopedFlowDenoiser(nn.Module):
    def __init__(self, cfg: DenoiserConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_size
        self.forward_dtype = getattr(torch, cfg.forward_dtype)
        self.embed_scale = math.sqrt(D)

        self.noise_proj = CastedLinear(cfg.vocab_size, D, bias=False)
        if cfg.problem_tokens:
            self.problem_embed = CastedEmbedding(cfg.vocab_size, D, init_std=1.0 / math.sqrt(D))
            self.problem_proj = CastedLinear(D, D, bias=False)
        if cfg.time_conditioning:
            self.time_mlp = nn.Sequential(nn.Linear(1, D), nn.SiLU(), nn.Linear(D, D))

        if cfg.puzzle_emb_ndim > 0:
            self.puzzle_emb = nn.Embedding(cfg.num_puzzle_identifiers, cfg.puzzle_emb_ndim)
            nn.init.zeros_(self.puzzle_emb.weight)

        self.lm_head = CastedLinear(D, cfg.vocab_size, bias=False)
        self.q_head = CastedLinear(D, 1, bias=True)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)

        self.rotary = RotaryEmbedding(D // cfg.num_heads, cfg.total_len, cfg.rope_theta) if cfg.mixer == "attention" else None
        self.core = ReasoningModule(
            [Block(D, cfg.num_heads, cfg.expansion, cfg.total_len, cfg.mixer, cfg.rms_norm_eps) for _ in range(cfg.num_layers)]
        )

        self.register_buffer("h_init", trunc_normal_init_(torch.empty(D), std=1.0), persistent=True)
        self.register_buffer("l_init", trunc_normal_init_(torch.empty(D), std=1.0), persistent=True)

    # ------------------------------------------------------------------ state
    def initial_state(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (batch_size, self.cfg.total_len, self.cfg.hidden_size)
        h = self.h_init.to(self.forward_dtype).expand(shape).clone()
        l = self.l_init.to(self.forward_dtype).expand(shape).clone()
        return h, l

    def puzzle_parameters(self):
        return [self.puzzle_emb.weight] if self.cfg.puzzle_emb_ndim > 0 else []

    def main_parameters(self):
        skip = {id(p) for p in self.puzzle_parameters()}
        return [p for p in self.parameters() if id(p) not in skip]

    # --------------------------------------------------------------- encoding
    def encode(self, x_t: torch.Tensor, t: torch.Tensor, problem: torch.Tensor | None, puzzle_ids: torch.Tensor | None) -> torch.Tensor:
        cfg = self.cfg
        e = self.noise_proj(x_t.to(self.forward_dtype))
        if cfg.problem_tokens:
            assert problem is not None
            e = e + self.problem_proj(self.problem_embed(problem, self.forward_dtype))
        if cfg.puzzle_emb_len > 0:
            pe = self.puzzle_emb(puzzle_ids).to(self.forward_dtype)
            pad = cfg.puzzle_emb_len * cfg.hidden_size - cfg.puzzle_emb_ndim
            if pad > 0:
                pe = F.pad(pe, (0, pad))
            pe = pe.view(-1, cfg.puzzle_emb_len, cfg.hidden_size)
            e = torch.cat((pe, e), dim=1)
        e = self.embed_scale * e
        if cfg.time_conditioning:
            te = self.time_mlp(t.float().view(-1, 1)).to(self.forward_dtype)
            e = e + te.unsqueeze(1)
        return e

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
        l: torch.Tensor,
        problem: torch.Tensor | None = None,
        puzzle_ids: torch.Tensor | None = None,
    ) -> DenoiserOutput:
        cfg = self.cfg
        cos_sin = self.rotary() if self.rotary is not None else None
        e = self.encode(x_t, t, problem, puzzle_ids)
        h = h.detach()
        l = l.detach()

        with torch.no_grad():
            for _ in range(cfg.H_cycles - 1):
                for _ in range(cfg.L_cycles):
                    l = self.core(l, h + e, cos_sin)
                h = self.core(h, l, cos_sin)
        for _ in range(cfg.L_cycles):
            l = self.core(l, h + e, cos_sin)
        h = self.core(h, l, cos_sin)

        logits = self.lm_head(h[:, cfg.puzzle_emb_len :]).float()
        q_logit = self.q_head(h[:, 0]).float().squeeze(-1)
        return DenoiserOutput(logits=logits, q_logit=q_logit, h=h.detach(), l=l.detach())

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
