"""Flow-side utilities: time samplers, interpolants, inference (Algorithm 2) and
best-Q ensembling."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .losses import IGNORE_LABEL_ID, probs
from .model import LoopedFlowDenoiser


# ------------------------------------------------------------------ time samplers
def sample_times(kind: str, batch_size: int, k: int, device, sort: bool = True) -> torch.Tensor:
    """Returns [B, k+1] timesteps in [0, 1]. ``sorted``: order k+1 uniform draws.
    ``random_start``: draw t0 ~ U[0,1] then k draws from U[t0, 1]. With ``sort=False``
    (ablation "w/o decreasing noise") the draws are left unordered."""
    if kind == "sorted":
        t = torch.rand(batch_size, k + 1, device=device)
    elif kind == "random_start":
        t0 = torch.rand(batch_size, 1, device=device)
        rest = t0 + (1 - t0) * torch.rand(batch_size, k, device=device)
        t = torch.cat((t0, rest), dim=1)
    else:
        raise ValueError(kind)
    if sort:
        t = t.sort(dim=1).values
    return t


# -------------------------------------------------------------------- interpolant
def clean_target(labels: torch.Tensor, inputs: torch.Tensor, given_mask: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """One-hot x_1 [B, L, V]: labels where known, inputs at fixed (given) positions, pad elsewhere."""
    tokens = torch.where(labels == IGNORE_LABEL_ID, torch.zeros_like(labels), labels)
    tokens = torch.where(given_mask, inputs, tokens)
    return F.one_hot(tokens, vocab_size).float()


def given_values(inputs: torch.Tensor, vocab_size: int) -> torch.Tensor:
    return F.one_hot(inputs, vocab_size).float()


def fix_given(x: torch.Tensor, given_mask: torch.Tensor, given_onehot: torch.Tensor) -> torch.Tensor:
    return torch.where(given_mask.unsqueeze(-1), given_onehot, x)


def interpolant(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, given_mask: torch.Tensor, given_onehot: torch.Tensor) -> torch.Tensor:
    """I_t = (1 - t) x0 + t x1 on free positions, the fixed problem entries elsewhere. t: [B]."""
    t = t.view(-1, 1, 1).to(x0.dtype)
    return fix_given((1 - t) * x0 + t * x1, given_mask, given_onehot)


# ---------------------------------------------------------------------- sampling
@dataclass
class SampleConfig:
    n_steps: int = 32
    gamma: float = 1.0
    sigma: float = 1.0
    prob_kind: str = "stablemax"
    record_states: bool = False
    H_cycles: int | None = None  # override the denoiser's recurrence cycles at inference


@dataclass
class SampleResult:
    tokens: torch.Tensor  # [B, L]
    x: torch.Tensor  # [B, L, V] final flow state
    q_logit: torch.Tensor  # [B] halting score of the final state
    residuals: torch.Tensor | None = None  # [B, n] stepwise relative residuals of h
    step_tokens: torch.Tensor | None = None  # [B, n, L] argmax of x_hat at every step


@torch.no_grad()
def sample(model: LoopedFlowDenoiser, batch: dict, cfg: SampleConfig, generator: torch.Generator | None = None) -> SampleResult:
    """Algorithm 2: integrate the probability flow coupled with the recurrent state."""
    inputs, given_mask = batch["inputs"], batch["given_mask"]
    device = inputs.device
    B, L = inputs.shape
    V = model.cfg.vocab_size
    problem = inputs if model.cfg.problem_tokens else None
    puzzle_ids = batch["puzzle_ids"]
    g_onehot = given_values(inputs, V)

    def noise():
        return cfg.sigma * torch.randn(B, L, V, device=device, generator=generator)

    x = fix_given(noise(), given_mask, g_onehot)
    h, l = model.initial_state(B)
    n = cfg.n_steps
    residuals = [] if cfg.record_states else None
    step_tokens = [] if cfg.record_states else None
    q_logit = None
    for i in range(n):
        t_i, t_next = i / n, (i + 1) / n
        a = min(1.0, max(0.0, 1 - cfg.gamma * (t_next - t_i)))
        s = a * t_i
        std = math.sqrt(max(0.0, (1 - s) ** 2 - (a - s) ** 2))
        x_bar = a * x + std * noise()
        x_bar = fix_given(x_bar, given_mask, g_onehot)
        out = model(x_bar, torch.full((B,), s, device=device), h, l, problem, puzzle_ids, H_cycles=cfg.H_cycles)
        x_hat = probs(out.logits, cfg.prob_kind)
        x = x_bar + (t_next - s) * (x_hat - x_bar) / (1 - s)
        x = fix_given(x, given_mask, g_onehot)
        if cfg.record_states:
            prev = h.float()
            residuals.append((out.h.float() - prev).flatten(1).norm(dim=1) / prev.flatten(1).norm(dim=1).clamp_min(1e-12))
            step_tokens.append(x_hat.argmax(-1))
        h, l = out.h, out.l
        q_logit = out.q_logit
    return SampleResult(
        tokens=x.argmax(-1),
        x=x,
        q_logit=q_logit,
        residuals=torch.stack(residuals, 1) if residuals else None,
        step_tokens=torch.stack(step_tokens, 1) if step_tokens else None,
    )


def repeat_batch(batch: dict, r: int) -> dict:
    """Repeat every problem r times along the batch axis (problem i occupies rows i*r .. i*r+r-1)."""
    return {k: (v.repeat_interleave(r, dim=0) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.no_grad()
def sample_repeated(model: LoopedFlowDenoiser, batch: dict, cfg: SampleConfig, r: int, max_batch: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw r independent samples per problem in as few sampler calls as possible.
    Returns tokens [B, r, L] and halting scores [B, r]. ``max_batch`` caps the number of
    sequences per call (defaults to B * r, i.e. a single call)."""
    B = batch["inputs"].shape[0]
    rep = repeat_batch(batch, r)
    total = B * r
    max_batch = total if max_batch is None else max(1, max_batch)
    tokens, q = [], []
    for s in range(0, total, max_batch):
        chunk = {k: (v[s : s + max_batch] if isinstance(v, torch.Tensor) else v) for k, v in rep.items()}
        res = sample(model, chunk, cfg)
        tokens.append(res.tokens)
        q.append(res.q_logit)
    return torch.cat(tokens).view(B, r, -1), torch.cat(q).view(B, r)


@torch.no_grad()
def sample_best_q(model: LoopedFlowDenoiser, batch: dict, cfg: SampleConfig, num_trajectories: int, max_batch: int | None = None) -> SampleResult:
    """Run several independent trajectories (batched together) and keep, per problem, the one
    with the highest halting score."""
    tokens, q = sample_repeated(model, batch, cfg, num_trajectories, max_batch)
    best = q.argmax(dim=1)
    idx = torch.arange(tokens.shape[0], device=tokens.device)
    return SampleResult(tokens=tokens[idx, best], x=F.one_hot(tokens[idx, best], model.cfg.vocab_size).float(), q_logit=q[idx, best])
