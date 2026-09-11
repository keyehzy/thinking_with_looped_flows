"""Training loop for looped flows (Algorithm 1).

Each optimisation step performs one denoiser call for every slot of a persistent
batch ("carry"). A slot holds a problem until it halts (ACT head fires, or the
maximum of k steps is reached), after which it is refilled with a fresh problem,
noise sample and time grid. This is the same scheme TRM uses and makes the cost
per optimisation step constant regardless of how many steps each problem takes.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from .data.common import Task, batch_to
from .flow import clean_target, fix_given, given_values, interpolant, sample_times
from .losses import IGNORE_LABEL_ID, bce_with_logits, probs, sequence_correct, sequence_cross_entropy
from .model import DenoiserConfig, LoopedFlowDenoiser
from .optim import EMA, AdamAtan2, SparseSignSGD, lr_at


@dataclass
class TrainConfig:
    total_steps: int = 100_000
    batch_size: int = 768
    k: int = 16  # maximum denoising steps per problem during training
    act_weight: float = 0.5
    exploration_prob: float = 0.1
    time_sampler: str = "sorted"  # "sorted" | "random_start"
    sigma: float | None = None  # None -> task default (1/sqrt|V|)
    loss_kind: str = "stablemax"
    pseudotargets: bool = False
    pseudotarget_ramp_steps: int = 20_000
    lr: float = 1e-4
    warmup_steps: int = 2_000
    lr_schedule: str = "constant"  # "constant" | "cosine"
    lr_min_ratio: float = 0.1
    weight_decay: float = 1.0
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    puzzle_emb_lr: float = 1e-4
    puzzle_emb_weight_decay: float = 1.0
    # ablations
    share_noise: bool = True  # reuse x0 across the steps of one problem
    decreasing_noise: bool = True  # sort the timesteps
    use_interpolant: bool = True  # feed I_t (else only fixed problem entries + noise)
    seed: int = 0
    log_every: int = 100


@dataclass
class Carry:
    inputs: torch.Tensor
    labels: torch.Tensor
    puzzle_ids: torch.Tensor
    given_mask: torch.Tensor
    x0: torch.Tensor
    x1: torch.Tensor
    given_onehot: torch.Tensor
    times: torch.Tensor
    step: torch.Tensor
    h: torch.Tensor
    l: torch.Tensor
    prev_probs: torch.Tensor
    halted: torch.Tensor


class RunningStats:
    def __init__(self):
        self.sums: dict[str, float] = {}
        self.count = 0

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v)
        self.count += 1

    def flush(self) -> dict[str, float]:
        out = {k: v / max(1, self.count) for k, v in self.sums.items()}
        self.sums, self.count = {}, 0
        return out


class Trainer:
    def __init__(self, task: Task, model: LoopedFlowDenoiser, cfg: TrainConfig, device):
        self.task = task
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.V = model.cfg.vocab_size
        self.sigma = cfg.sigma if cfg.sigma is not None else task.spec.default_sigma
        self.rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)

        self.opt = AdamAtan2(model.main_parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
        self.puzzle_opt = SparseSignSGD(model.puzzle_parameters(), lr=cfg.puzzle_emb_lr, weight_decay=cfg.puzzle_emb_weight_decay) if model.puzzle_parameters() else None
        self.ema = EMA(model, cfg.ema_decay)
        self.step = 0
        self.carry: Carry | None = None
        self.stats = RunningStats()

    # ------------------------------------------------------------------ carry
    def _empty_carry(self) -> Carry:
        B, L, V, cfg = self.cfg.batch_size, self.task.spec.seq_len, self.V, self.cfg
        dev = self.device
        h, l = self.model.initial_state(B)
        long = lambda *s: torch.zeros(*s, dtype=torch.long, device=dev)
        return Carry(
            inputs=long(B, L),
            labels=torch.full((B, L), IGNORE_LABEL_ID, dtype=torch.long, device=dev),
            puzzle_ids=long(B),
            given_mask=torch.zeros(B, L, dtype=torch.bool, device=dev),
            x0=torch.zeros(B, L, V, device=dev),
            x1=torch.zeros(B, L, V, device=dev),
            given_onehot=torch.zeros(B, L, V, device=dev),
            times=torch.zeros(B, cfg.k + 1, device=dev),
            step=long(B),
            h=h,
            l=l,
            prev_probs=torch.zeros(B, L, V, device=dev),
            halted=torch.ones(B, dtype=torch.bool, device=dev),
        )

    @torch.no_grad()
    def _refill(self, c: Carry):
        idx = c.halted.nonzero(as_tuple=True)[0]
        n = idx.numel()
        if n == 0:
            return
        batch = batch_to(self.task.sample_train(n, self.rng), self.device)
        c.inputs[idx] = batch["inputs"]
        c.labels[idx] = batch["labels"]
        c.puzzle_ids[idx] = batch["puzzle_ids"]
        c.given_mask[idx] = batch["given_mask"]
        c.given_onehot[idx] = given_values(batch["inputs"], self.V)
        c.x1[idx] = clean_target(batch["labels"], batch["inputs"], batch["given_mask"], self.V)
        c.x0[idx] = self.sigma * torch.randn(n, *c.x0.shape[1:], device=self.device)
        c.times[idx] = sample_times(self.cfg.time_sampler, n, self.cfg.k, self.device, sort=self.cfg.decreasing_noise)
        c.step[idx] = 0
        h0, l0 = self.model.initial_state(n)
        c.h[idx] = h0
        c.l[idx] = l0
        c.prev_probs[idx] = 0
        c.halted[idx] = False

    def pseudotarget_prob(self) -> float:
        if not self.cfg.pseudotargets:
            return 0.0
        return min(1.0, self.step / max(1, self.cfg.pseudotarget_ramp_steps))

    # ------------------------------------------------------------- one step
    def train_step(self) -> dict:
        cfg, c = self.cfg, self.carry
        if c is None:
            c = self.carry = self._empty_carry()
        self._refill(c)
        B = cfg.batch_size
        dev = self.device

        with torch.no_grad():
            t = c.times.gather(1, c.step.view(-1, 1)).squeeze(1)
            x0 = c.x0 if cfg.share_noise else self.sigma * torch.randn_like(c.x0)
            target = c.x1
            p_pt = self.pseudotarget_prob()
            if p_pt > 0:
                use_pt = (c.step >= 1) & (torch.rand(B, device=dev) < p_pt)
                target = torch.where(use_pt.view(-1, 1, 1), c.prev_probs, c.x1)
            if cfg.use_interpolant:
                x_t = interpolant(x0, target, t, c.given_mask, c.given_onehot)
            else:
                x_t = fix_given(x0, c.given_mask, c.given_onehot)

        problem = c.inputs if self.model.cfg.problem_tokens else None
        out = self.model(x_t, t, c.h, c.l, problem, c.puzzle_ids)

        with torch.no_grad():
            pred = out.logits.argmax(-1)
            correct = sequence_correct(pred, c.labels)
        ce = sequence_cross_entropy(out.logits, c.labels, cfg.loss_kind)
        act = bce_with_logits(out.q_logit, correct)
        loss = ce.mean() + cfg.act_weight * act.mean()

        lr = lr_at(self.step, cfg.lr, cfg.warmup_steps, cfg.total_steps, cfg.lr_schedule, cfg.lr_min_ratio)
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.opt.zero_grad(set_to_none=True)
        if self.puzzle_opt is not None:
            self.puzzle_opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.main_parameters(), cfg.grad_clip)
        self.opt.step()
        if self.puzzle_opt is not None:
            self.puzzle_opt.step()
        self.ema.update(self.model)
        self.step += 1

        with torch.no_grad():
            c.h, c.l = out.h, out.l
            c.prev_probs = probs(out.logits, cfg.loss_kind)
            c.step += 1
            is_last = c.step >= cfg.k
            halt = is_last | (out.q_logit > 0)
            explore = torch.rand(B, device=dev) < cfg.exploration_prob
            min_steps = explore * torch.randint(2, cfg.k + 1, (B,), device=dev)
            c.halted = halt & (c.step >= min_steps)

            valid = c.labels != IGNORE_LABEL_ID
            tok_acc = ((pred == c.labels) & valid).sum() / valid.sum().clamp_min(1)
            q_acc = ((out.q_logit > 0) == correct).float().mean()
            self.stats.add(
                loss=loss.item(), ce=ce.mean().item(), act=act.mean().item(), tok_acc=tok_acc.item(),
                seq_acc=correct.float().mean().item(), q_acc=q_acc.item(), halt_rate=c.halted.float().mean().item(),
                mean_t=t.mean().item(), gnorm=float(gnorm), lr=lr, pt_prob=p_pt,
            )
        return {"loss": loss.item()}

    # ------------------------------------------------------------ checkpoint
    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.opt.state_dict(),
            "puzzle_opt": self.puzzle_opt.state_dict() if self.puzzle_opt is not None else None,
            "step": self.step,
            "train_config": asdict(self.cfg),
            "model_config": asdict(self.model.cfg),
            "task": self.task.spec.name,
        }

    def load_state_dict(self, sd: dict):
        self.model.load_state_dict(sd["model"])
        self.ema.load_state_dict({k: v.to(self.device) for k, v in sd["ema"].items()})
        self.opt.load_state_dict(sd["opt"])
        if self.puzzle_opt is not None and sd.get("puzzle_opt") is not None:
            self.puzzle_opt.load_state_dict(sd["puzzle_opt"])
        for opt in (self.opt, self.puzzle_opt):
            if opt is None:
                continue
            for state in opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        self.step = sd["step"]

    def ema_model(self) -> LoopedFlowDenoiser:
        m = LoopedFlowDenoiser(DenoiserConfig(**asdict(self.model.cfg))).to(self.device)
        m.load_state_dict(self.ema.state_dict())
        m.eval()
        return m
