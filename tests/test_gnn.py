"""Tests for the GNN backbone (Model B) and the mesh-edge data path."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.data import (TrajectoryDataset, assemble_edge_features,
                      assemble_node_features_for_gnn, build_mesh_edge_index,
                      fit_normalization_stats)
from src.neural_solver import GNNConfig, GNNSolver, build_solver
from src.train import rollout_horizon_for_epoch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "cloth_trajectories"


# -----------------------------------------------------------------------------
# Mesh-edge graph
# -----------------------------------------------------------------------------

def test_mesh_edge_count_3x3():
    """3x3 grid:
       - horizontal (i+1 connects to i): (gx-1)*gy = 2*3 = 6
       - vertical (j+1 connects to j):   gx*(gy-1) = 3*2 = 6
       - NE-SW diag (i+1, j+1):          (gx-1)*(gy-1) = 2*2 = 4
       - NW-SE diag (i+1, j-1):          (gx-1)*(gy-1) = 2*2 = 4
       Total undirected = 20; directed = 40.
    """
    e = build_mesh_edge_index(3, 3)
    assert e.shape == (2, 40)
    # No self-loops
    assert (e[0] != e[1]).all()


def test_mesh_edges_are_symmetric():
    """For every (i, j) edge, (j, i) must also exist."""
    e = build_mesh_edge_index(4, 4)
    pairs = set((int(a), int(b)) for a, b in zip(e[0].tolist(), e[1].tolist()))
    for a, b in pairs:
        assert (b, a) in pairs


def test_assemble_edge_features_shape():
    n = 16
    x = torch.randn(n, 3)
    e = build_mesh_edge_index(4, 4)
    feat = assemble_edge_features(x, e)
    assert feat.shape == (e.shape[1], 4)
    # Last column is non-negative (it's a norm)
    assert (feat[:, 3] >= 0).all()


def test_assemble_node_features_for_gnn_translation_invariant():
    n, C = 16, 5
    x = torch.randn(n, 3)
    v = torch.randn(n, C, 3)
    F = torch.eye(3).expand(n, 3, 3).contiguous()
    f_ext = torch.zeros(n, 3)
    a = assemble_node_features_for_gnn(x, v, F, f_ext, mean_center=True)
    b = assemble_node_features_for_gnn(x + 5.0, v, F, f_ext, mean_center=True)
    assert torch.allclose(a, b)


# -----------------------------------------------------------------------------
# GNN model
# -----------------------------------------------------------------------------

def test_gnn_forward_shape():
    m = GNNSolver(GNNConfig(node_feature_dim=30, edge_feature_dim=4,
                            message_passing_steps=3, hidden_dim=32))
    n_nodes = 12
    n_edges = 30
    nf = torch.randn(n_nodes, 30)
    ef = torch.randn(n_edges, 4)
    ei = torch.randint(0, n_nodes, (2, n_edges))
    y = m(nf, ef, ei)
    assert y.shape == (n_nodes, 3)
    assert torch.isfinite(y).all()


def test_gnn_gradient_flow():
    m = GNNSolver(GNNConfig(node_feature_dim=30, edge_feature_dim=4,
                            message_passing_steps=2, hidden_dim=32))
    nf = torch.randn(8, 30)
    ef = torch.randn(16, 4)
    ei = torch.randint(0, 8, (2, 16))
    target = torch.randn(8, 3)
    loss = nn.functional.mse_loss(m(nf, ef, ei), target)
    loss.backward()
    for p in m.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_gnn_can_overfit_one_batch():
    """Tiny GNN must overfit a fixed graph + target in <500 steps."""
    torch.manual_seed(0)
    m = GNNSolver(GNNConfig(node_feature_dim=30, edge_feature_dim=4,
                            message_passing_steps=2, hidden_dim=64))
    nf = torch.randn(8, 30)
    ef = torch.randn(16, 4)
    ei = torch.randint(0, 8, (2, 16))
    target = torch.randn(8, 3)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(500):
        opt.zero_grad()
        loss = nn.functional.mse_loss(m(nf, ef, ei), target)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05, f"GNN overfit failed: final loss={loss.item():.4f}"


def test_build_solver_gnn_factory():
    cfg = {"type": "gnn", "node_feature_dim": 30, "edge_feature_dim": 4,
           "message_passing_steps": 3, "hidden_dim": 64}
    m = build_solver(cfg)
    assert isinstance(m, GNNSolver)
    assert m.cfg.message_passing_steps == 3


# -----------------------------------------------------------------------------
# Rollout curriculum
# -----------------------------------------------------------------------------

def test_rollout_horizon_schedule():
    """At default boundaries (0.5, 0.7, 0.85), horizon must ramp 1->2->4->8."""
    schedule = (1, 2, 4, 8)
    # 30 epochs
    horizons = [rollout_horizon_for_epoch(e, 30, schedule) for e in range(30)]
    # First 50% (epochs 0..14) -> h=1
    assert all(h == 1 for h in horizons[:14])
    # Middle 20% (epochs 15..20 ish) -> h=2
    assert horizons[15] == 2
    # Next 15% -> h=4
    assert horizons[22] == 4
    # Final 15% -> h=8
    assert horizons[28] == 8


# -----------------------------------------------------------------------------
# GNN data path against the smoke dataset
# -----------------------------------------------------------------------------

def test_gnn_dataset_returns_dict():
    if not (DATA_DIR / "index.csv").exists():
        pytest.skip("smoke dataset not present")
    ds = TrajectoryDataset(DATA_DIR / "index.csv", mode="gnn")
    sample = ds[0]
    assert set(sample.keys()) == {"node_feat", "edge_feat", "edge_index", "target"}
    assert sample["node_feat"].shape == (ds.N, 30)
    assert sample["edge_feat"].shape == (ds.edge_index.shape[1], 4)
    assert sample["edge_index"].shape[0] == 2
    assert sample["target"].shape == (ds.N, 3)


def test_gnn_normalization_with_edges():
    if not (DATA_DIR / "index.csv").exists():
        pytest.skip("smoke dataset not present")
    ds_raw = TrajectoryDataset(DATA_DIR / "index.csv", mode="gnn")
    stats = fit_normalization_stats(ds_raw, n_samples=20, fit_edges=True)
    assert stats.edge_mean is not None
    assert stats.edge_std is not None
    assert stats.edge_mean.shape == (4,)
    assert stats.edge_std.shape == (4,)
    # Edge stats round-trip via dict
    s2 = stats.from_dict(stats.to_dict())
    assert torch.allclose(s2.edge_mean, stats.edge_mean)
