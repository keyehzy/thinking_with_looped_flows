"""Adam-atan2 (Everett et al., 2024), signSGD for sparse puzzle embeddings,
learning-rate schedules and a parameter EMA."""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


class AdamAtan2(Optimizer):
    """Adam whose update is ``a * atan2(m_hat, b * sqrt(v_hat))``; scale invariant and
    epsilon free. Weight decay is decoupled (AdamW style)."""

    def __init__(self, params, lr: float = 1e-4, betas=(0.9, 0.95), weight_decay: float = 0.0, a: float = 1.27, b: float = 1.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay, a=a, b=b))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, (beta1, beta2), wd, a, b = group["lr"], group["betas"], group["weight_decay"], group["a"], group["b"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                step = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]
                if wd > 0:
                    p.mul_(1 - lr * wd)
                m.lerp_(grad, 1 - beta1)
                v.lerp_(grad.square(), 1 - beta2)
                bc1 = 1 - beta1**step
                bc2 = 1 - beta2**step
                update = torch.atan2(m / bc1, b * torch.sqrt(v / bc2)) * a
                p.add_(update, alpha=-lr)
        return loss


class SparseSignSGD(Optimizer):
    """signSGD with decoupled weight decay, applied only to embedding rows that received gradient."""

    def __init__(self, params, lr: float = 1e-4, weight_decay: float = 1.0):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                touched = (grad != 0).any(dim=-1, keepdim=True).to(p.dtype)
                p.add_(-lr * touched * (torch.sign(grad) + wd * p))
        return loss


def lr_at(step: int, base_lr: float, warmup_steps: int, total_steps: int, schedule: str, min_ratio: float = 0.1) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if schedule == "constant":
        return base_lr
    if schedule == "cosine":
        progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)))
        return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))
    raise ValueError(schedule)


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].lerp_(v, 1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}

    def copy_to(self, model: torch.nn.Module):
        model.load_state_dict(self.shadow)
