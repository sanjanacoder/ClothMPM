"""Tests for src.neural_solver — MLP shapes, gradient flow, and a tiny
overfitting check."""

import torch
from torch import nn

from src.neural_solver import MLPConfig, MLPSolver, build_solver


def test_mlp_forward_shape():
    cfg = MLPConfig(feature_dim=30)
    m = MLPSolver(cfg)
    x = torch.randn(2, 64, 9, 30)
    y = m(x)
    assert y.shape == (2, 64, 3)


def test_mlp_param_count_matches_config():
    """A small sanity check that we can size models predictably."""
    m = MLPSolver(MLPConfig(feature_dim=30, hidden_dims=(64, 64)))
    n = sum(p.numel() for p in m.parameters())
    # 30*9 -> 64 -> 64 -> 3 with LayerNorms; should be around 22-25k.
    assert 15_000 < n < 30_000, f"unexpected param count {n}"


def test_mlp_gradient_flow():
    m = MLPSolver(MLPConfig(feature_dim=30))
    x = torch.randn(4, 16, 9, 30, requires_grad=False)
    target = torch.randn(4, 16, 3)
    y = m(x)
    loss = nn.functional.mse_loss(y, target)
    loss.backward()
    # Every parameter must have a finite gradient
    for p in m.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_mlp_can_overfit_one_batch():
    """An MLP with enough params must drive loss < 0.05 on a single batch
    in a few hundred steps. This guards against silent silent broken plumbing
    (wrong reshape, frozen params, etc.)."""
    torch.manual_seed(0)
    m = MLPSolver(MLPConfig(feature_dim=30))
    x = torch.randn(2, 16, 9, 30)
    target = torch.randn(2, 16, 3)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        loss = nn.functional.mse_loss(m(x), target)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05, f"failed to overfit one batch: final loss={loss.item():.4f}"


def test_build_solver_factory():
    cfg = {"type": "mlp", "feature_dim": 30, "hidden_dims": [64, 64]}
    m = build_solver(cfg)
    assert isinstance(m, MLPSolver)
    # Unknown type raises ValueError
    import pytest
    with pytest.raises(ValueError):
        build_solver({"type": "lstm", "feature_dim": 30})
