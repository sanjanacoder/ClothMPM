"""Neural grid-update models (M3).

Two backbones, both predicting per-particle acceleration `a_p`:

- `MLPSolver` (Model A): per-particle MLP that consumes the concatenated
  self + 1-ring features assembled by `src.data.assemble_features`.
- `GNNSolver` (Model B): MeshGraphNets-style Encode-Process-Decode GNN with
  L=5 message-passing rounds on the mesh-edge graph. Hand-rolled in plain
  PyTorch using scatter_add for aggregation; no torch_geometric dependency.

Both share the integration step downstream:
    a_t = solver(state_t)
    v_{t+1} = v_t + dt * a_t       (semi-implicit Euler matches GNS)
    x_{t+1} = x_t + dt * v_{t+1}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import nn


@dataclass
class MLPConfig:
    feature_dim: int        # F = 30 by default with C=5 velocity history
    n_nodes: int = 9        # 1 self + 8 neighbors
    hidden_dims: Sequence[int] = (128, 128, 128)
    activation: str = "silu"
    norm: str = "layer"     # 'layer' | 'none'
    output_dim: int = 3     # acceleration


def _activation(name: str) -> nn.Module:
    return {
        "silu": nn.SiLU(),
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
    }[name]


class MLPSolver(nn.Module):
    """Per-particle MLP: concat (self + 8 neighbors) features -> predict accel.

    Input shape : (B*N, K, F)   K=9 (self + 8 neighbors), F = feature_dim
    Output shape: (B*N, 3)      acceleration (or in normalized units if a
                                 stats object was applied at the data layer)
    """

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.cfg = config
        in_dim = config.feature_dim * config.n_nodes
        layers: list[nn.Module] = []
        prev = in_dim
        for h in config.hidden_dims:
            layers.append(nn.Linear(prev, h))
            if config.norm == "layer":
                layers.append(nn.LayerNorm(h))
            layers.append(_activation(config.activation))
            prev = h
        layers.append(nn.Linear(prev, config.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (..., K, F)
        flat = features.reshape(*features.shape[:-2], -1)
        return self.net(flat)


# -----------------------------------------------------------------------------
# GNN (MeshGraphNets-style, hand-rolled message passing)
# -----------------------------------------------------------------------------

def _make_mlp(in_dim: int, hidden: Sequence[int], out_dim: int,
              activation: str = "relu", norm: str = "layer",
              norm_output: bool = True) -> nn.Module:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(_activation(activation))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    if norm_output and norm == "layer":
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


@dataclass
class GNNConfig:
    node_feature_dim: int       # raw per-node feature dim (e.g. 30 for x_centered+v_history+F+f_ext)
    edge_feature_dim: int       # raw per-edge feature dim (relative pos + |rel_pos| -> 4)
    hidden_dim: int = 128
    message_passing_steps: int = 5
    encoder_hidden: Sequence[int] = (128, 128)
    processor_hidden: Sequence[int] = (128, 128)
    decoder_hidden: Sequence[int] = (128, 128)
    activation: str = "relu"
    norm: str = "layer"
    output_dim: int = 3


class GraphNetBlock(nn.Module):
    """One message-passing round.

    edge_mlp:  takes [edge_feat_l, src_node_l, dst_node_l] -> new edge_feat_l
    aggregator: scatter_add over destination nodes
    node_mlp:  takes [node_feat_l, agg_msg]                -> new node_feat_l
    Residual connection on both edge and node features (MeshGraphNets, p. 4).
    """

    def __init__(self, hidden_dim: int, mlp_hidden: Sequence[int],
                 activation: str, norm: str):
        super().__init__()
        self.edge_mlp = _make_mlp(
            in_dim=3 * hidden_dim,
            hidden=mlp_hidden,
            out_dim=hidden_dim,
            activation=activation, norm=norm, norm_output=True,
        )
        self.node_mlp = _make_mlp(
            in_dim=2 * hidden_dim,
            hidden=mlp_hidden,
            out_dim=hidden_dim,
            activation=activation, norm=norm, norm_output=True,
        )

    def forward(self,
                node_feat: torch.Tensor,    # (N, H)
                edge_feat: torch.Tensor,    # (E, H)
                edge_index: torch.Tensor,   # (2, E)  -- [src, dst]
                ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index[0], edge_index[1]
        e_input = torch.cat([edge_feat, node_feat[src], node_feat[dst]], dim=-1)
        new_edge = self.edge_mlp(e_input) + edge_feat
        # Aggregate by destination (sum over incoming messages)
        agg = torch.zeros_like(node_feat)
        agg.index_add_(0, dst, new_edge)
        v_input = torch.cat([node_feat, agg], dim=-1)
        new_node = self.node_mlp(v_input) + node_feat
        return new_node, new_edge


class GNNSolver(nn.Module):
    """MeshGraphNets-style Encode-Process-Decode for cloth.

    Shapes (one frame):
      raw_node_feat : (N, F_node)    e.g., (N, 30)
      raw_edge_feat : (E, F_edge)    e.g., (E, 4)
      edge_index    : (2, E)         long tensor
    Output:
      acceleration  : (N, output_dim)
    """

    def __init__(self, config: GNNConfig):
        super().__init__()
        self.cfg = config
        self.node_encoder = _make_mlp(
            config.node_feature_dim, config.encoder_hidden, config.hidden_dim,
            activation=config.activation, norm=config.norm, norm_output=True,
        )
        self.edge_encoder = _make_mlp(
            config.edge_feature_dim, config.encoder_hidden, config.hidden_dim,
            activation=config.activation, norm=config.norm, norm_output=True,
        )
        self.processor = nn.ModuleList([
            GraphNetBlock(config.hidden_dim,
                          config.processor_hidden,
                          config.activation, config.norm)
            for _ in range(config.message_passing_steps)
        ])
        self.decoder = _make_mlp(
            config.hidden_dim, config.decoder_hidden, config.output_dim,
            activation=config.activation, norm=config.norm, norm_output=False,
        )

    def forward(self,
                node_feat: torch.Tensor,
                edge_feat: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        v = self.node_encoder(node_feat)
        e = self.edge_encoder(edge_feat)
        for block in self.processor:
            v, e = block(v, e, edge_index)
        return self.decoder(v)


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

def build_solver(model_cfg: dict) -> nn.Module:
    """Construct a solver from a config dict matching configs/mlp.yaml or
    configs/gnn.yaml's `model:` block.
    """
    kind = model_cfg.get("type", "mlp").lower()
    if kind == "mlp":
        return MLPSolver(MLPConfig(
            feature_dim=int(model_cfg["feature_dim"]),
            n_nodes=int(model_cfg.get("n_nodes", 9)),
            hidden_dims=tuple(model_cfg.get("hidden_dims", (128, 128, 128))),
            activation=str(model_cfg.get("activation", "silu")),
            norm=str(model_cfg.get("norm", "layer")),
            output_dim=int(model_cfg.get("output_dim", 3)),
        ))
    if kind == "gnn":
        return GNNSolver(GNNConfig(
            node_feature_dim=int(model_cfg["node_feature_dim"]),
            edge_feature_dim=int(model_cfg.get("edge_feature_dim", 4)),
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            message_passing_steps=int(model_cfg.get("message_passing_steps", 5)),
            encoder_hidden=tuple(model_cfg.get("encoder_mlp_hidden", (128, 128))),
            processor_hidden=tuple(model_cfg.get("processor_mlp_hidden", (128, 128))),
            decoder_hidden=tuple(model_cfg.get("decoder_hidden", (128, 128))),
            activation=str(model_cfg.get("activation", "relu")),
            norm=str(model_cfg.get("norm", "layer")),
            output_dim=int(model_cfg.get("output_dim", 3)),
        ))
    raise ValueError(f"unknown solver type: {kind}")
