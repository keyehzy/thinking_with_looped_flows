import math

import numpy as np
import pytest
import torch

from looped_flows.flow import SampleConfig, clean_target, given_values, interpolant, sample, sample_times
from looped_flows.losses import IGNORE_LABEL_ID, sequence_cross_entropy, stablemax_probs, token_cross_entropy
from looped_flows.model import DenoiserConfig, LoopedFlowDenoiser
from looped_flows.optim import AdamAtan2, lr_at


def small_model(**kw):
    cfg = DenoiserConfig(vocab_size=5, seq_len=6, hidden_size=32, num_heads=4, num_layers=1, H_cycles=2, L_cycles=2, puzzle_emb_ndim=32, forward_dtype="float32", **kw)
    return LoopedFlowDenoiser(cfg)


def test_time_samplers_sorted_and_random_start():
    t = sample_times("sorted", 64, 16, "cpu")
    assert t.shape == (64, 17)
    assert (t[:, 1:] >= t[:, :-1]).all()
    t = sample_times("random_start", 64, 16, "cpu")
    assert (t[:, 1:] >= t[:, :-1]).all()
    assert (t >= t[:, :1]).all()
    t = sample_times("sorted", 64, 16, "cpu", sort=False)
    assert not (t[:, 1:] >= t[:, :-1]).all()


def test_interpolant_keeps_given_entries_clean():
    V = 5
    inputs = torch.tensor([[1, 0, 2, 0]])
    labels = torch.tensor([[1, 3, 2, 4]])
    given = inputs != 0
    x1 = clean_target(labels, inputs, given, V)
    g = given_values(inputs, V)
    x0 = torch.randn(1, 4, V)
    xt = interpolant(x0, x1, torch.tensor([0.3]), given, g)
    assert torch.allclose(xt[0, 0], torch.nn.functional.one_hot(torch.tensor(1), V).float())
    assert torch.allclose(xt[0, 1], 0.7 * x0[0, 1] + 0.3 * x1[0, 1])


def test_stablemax_matches_definition_and_has_finite_grads():
    logits = torch.tensor([[1.0, -1.0, 0.0, 1.0]], requires_grad=True)
    p = stablemax_probs(logits)
    s = torch.tensor([2.0, 0.5, 1.0, 2.0])
    assert torch.allclose(p[0], s / s.sum())
    loss = token_cross_entropy(logits, torch.tensor([0]), "stablemax").sum()
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_cross_entropy_ignores_padding():
    logits = torch.randn(2, 3, 4)
    labels = torch.tensor([[1, IGNORE_LABEL_ID, 2], [IGNORE_LABEL_ID, IGNORE_LABEL_ID, 3]])
    ce = sequence_cross_entropy(logits, labels, "softmax")
    ref0 = torch.nn.functional.cross_entropy(logits[0, [0, 2]], labels[0, [0, 2]])
    ref1 = torch.nn.functional.cross_entropy(logits[1, [2]], labels[1, [2]])
    assert torch.allclose(ce, torch.stack([ref0, ref1]), atol=1e-5)


def test_model_forward_shapes_and_state_detached():
    m = small_model()
    B = 3
    h, l = m.initial_state(B)
    x = torch.randn(B, 6, 5)
    out = m(x, torch.rand(B), h, l, None, torch.zeros(B, dtype=torch.long))
    assert out.logits.shape == (B, 6, 5)
    assert out.q_logit.shape == (B,)
    assert not out.h.requires_grad and not out.l.requires_grad
    out.logits.sum().backward()
    assert m.noise_proj.weight.grad is not None


def test_model_with_problem_tokens():
    m = small_model(problem_tokens=True)
    B = 2
    h, l = m.initial_state(B)
    out = m(torch.randn(B, 6, 5), torch.rand(B), h, l, torch.randint(0, 5, (B, 6)), torch.zeros(B, dtype=torch.long))
    assert out.logits.shape == (B, 6, 5)


def test_gamma_zero_sampler_is_euler_and_ends_at_prediction():
    """With gamma = 0 the last update returns exactly the final denoiser prediction."""
    torch.manual_seed(0)
    m = small_model().eval()
    inputs = torch.tensor([[1, 0, 2, 0, 0, 0]])
    batch = {"inputs": inputs, "labels": inputs, "puzzle_ids": torch.zeros(1, dtype=torch.long), "given_mask": inputs != 0}
    res = sample(m, batch, SampleConfig(n_steps=4, gamma=0.0, sigma=1.0, prob_kind="softmax"))
    assert res.tokens.shape == (1, 6)
    assert torch.allclose(res.x.sum(-1), torch.ones(1, 6), atol=1e-4)
    assert (res.tokens[0, [0, 2]] == torch.tensor([1, 2])).all()


def test_stochastic_sampler_noise_level_identity():
    """The backtracking step must map p_t samples to p_s samples (variance identity)."""
    gamma, t_i, t_next = 5.0, 0.5, 0.5 + 1 / 32
    a = min(1.0, max(0.0, 1 - gamma * (t_next - t_i)))
    s = a * t_i
    var = a**2 * (1 - t_i) ** 2 + (1 - s) ** 2 - (a - s) ** 2
    assert math.isclose(var, (1 - s) ** 2)


def test_adam_atan2_step_is_bounded():
    p = torch.nn.Parameter(torch.zeros(10))
    opt = AdamAtan2([p], lr=0.1, weight_decay=0.0)
    p.grad = torch.randn(10) * 1e6
    opt.step()
    assert p.abs().max() <= 0.1 * 1.27 * math.pi / 2 + 1e-6


def test_lr_schedule():
    assert lr_at(0, 1.0, 10, 100, "constant") == pytest.approx(0.1)
    assert lr_at(10, 1.0, 10, 100, "constant") == 1.0
    assert lr_at(100, 1.0, 10, 100, "cosine", 0.1) == pytest.approx(0.1)


def test_sample_repeated_and_best_q_batched():
    torch.manual_seed(0)
    m = small_model().eval()
    inputs = torch.tensor([[1, 0, 2, 0, 0, 0], [0, 0, 0, 0, 3, 0]])
    batch = {"inputs": inputs, "labels": inputs, "puzzle_ids": torch.zeros(2, dtype=torch.long), "given_mask": inputs != 0}
    from looped_flows.flow import sample_best_q, sample_repeated

    cfg = SampleConfig(n_steps=3, gamma=1.0, sigma=1.0, prob_kind="softmax")
    tokens, q = sample_repeated(m, batch, cfg, r=5, max_batch=4)
    assert tokens.shape == (2, 5, 6) and q.shape == (2, 5)
    assert (tokens[0, :, 0] == 1).all() and (tokens[1, :, 4] == 3).all()
    best = sample_best_q(m, batch, cfg, num_trajectories=5, max_batch=4)
    assert best.tokens.shape == (2, 6)


def test_h_cycles_override_changes_output():
    torch.manual_seed(0)
    m = small_model().eval()
    B = 2
    h, l = m.initial_state(B)
    x = torch.randn(B, 6, 5)
    t = torch.rand(B)
    ids = torch.zeros(B, dtype=torch.long)
    full = m(x, t, h, l, None, ids).logits
    one = m(x, t, h, l, None, ids, H_cycles=1).logits
    assert not torch.allclose(full, one)
    assert torch.allclose(full, m(x, t, h, l, None, ids, H_cycles=2).logits)


def test_compile_core_keeps_state_dict_keys():
    m = small_model()
    keys = set(m.state_dict().keys())
    m.compile_core()
    assert set(m.state_dict().keys()) == keys
