"""StableMax cross-entropy (Prieto et al., 2025) and helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_LABEL_ID = -100


def s(x: torch.Tensor) -> torch.Tensor:
    """s(x) = x + 1 for x >= 0 and 1 / (1 - x) otherwise. Both branches are evaluated on
    clamped inputs so the unselected branch cannot produce inf/nan gradients."""
    return torch.where(x < 0, 1 / (1 - x.clamp(max=0)), x.clamp(min=0) + 1)


def stablemax_log_probs(logits: torch.Tensor) -> torch.Tensor:
    sx = s(logits)
    return torch.log(sx / sx.sum(dim=-1, keepdim=True))


def stablemax_probs(logits: torch.Tensor) -> torch.Tensor:
    sx = s(logits)
    return sx / sx.sum(dim=-1, keepdim=True)


def probs(logits: torch.Tensor, kind: str) -> torch.Tensor:
    logits = logits.float()
    if kind == "stablemax":
        return stablemax_probs(logits)
    if kind == "softmax":
        return torch.softmax(logits, dim=-1)
    raise ValueError(kind)


def token_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, kind: str) -> torch.Tensor:
    """Per-token cross-entropy, zero at ignored positions. logits [.., V], labels [..]."""
    logits = logits.float()
    valid = labels != IGNORE_LABEL_ID
    safe_labels = torch.where(valid, labels, torch.zeros_like(labels))
    if kind == "stablemax":
        logp = stablemax_log_probs(logits)
    elif kind == "softmax":
        logp = torch.log_softmax(logits, dim=-1)
    else:
        raise ValueError(kind)
    nll = -logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return torch.where(valid, nll, torch.zeros_like(nll))


def sequence_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, kind: str) -> torch.Tensor:
    """Mean cross-entropy over valid tokens for each sequence. Returns [B]."""
    nll = token_cross_entropy(logits, labels, kind)
    valid = (labels != IGNORE_LABEL_ID).sum(-1).clamp_min(1)
    return nll.sum(-1) / valid


def sequence_correct(pred_tokens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    valid = labels != IGNORE_LABEL_ID
    return ((pred_tokens == labels) | ~valid).all(-1)


def bce_with_logits(q_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(q_logit.float(), target.float(), reduction="none")
