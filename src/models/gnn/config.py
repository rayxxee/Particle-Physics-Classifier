"""
GNN model configuration for particle physics event classification.

The GNN treats each event as a graph:
  - Nodes  = final-state particles (HIGGS: 4 particles × 7 features = 28 features)
  - Edges  = k-nearest neighbors in (eta, phi) space
  - Message passing = EdgeConv (DGCNN-style)

HIGGS dataset input reshaping:
    Raw input: (n_events, 28) — flat feature vector
    GNN input: (n_events, 4, 7) — 4 particles × 7 features each

Usage:
    from src.models.gnn.config import GNNConfig

    config = GNNConfig(k_neighbors=8, n_edge_conv_layers=3, hidden_dim=64)
    # For HIGGS: n_particles=4, n_node_features=7 → input (N, 4, 7)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GNNConfig:
    """
    Full hyperparameter specification for the EdgeConv GNN.

    Architecture:
        Input(n_events, n_particles, n_node_features)
        → [kNN graph construction + EdgeConv] × n_edge_conv_layers
        → Global Mean Pool
        → MLP head
        → Sigmoid
    """

    # ── Graph construction ────────────────────────────────────────────────────
    k_neighbors: int = 8
    """k-nearest neighbors for kNN graph construction in (eta, phi) space."""

    n_particles: int = 4
    """Number of nodes (particles) per event graph."""

    n_node_features: int = 7
    """Feature dimension per node. For HIGGS: 28 features / 4 particles = 7."""

    # ── Architecture ──────────────────────────────────────────────────────────
    n_edge_conv_layers: int = 3
    """Number of EdgeConv message-passing layers."""

    hidden_dim: int = 64
    """Hidden dimension in each EdgeConv MLP."""

    mlp_head_dims: list[int] = field(default_factory=lambda: [128, 64])
    """Hidden dims of the classification MLP head after global pooling."""

    dropout: float = 0.3
    """Dropout probability in the MLP head."""

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = 50
    """Maximum training epochs."""

    batch_size: int = 1024
    """Training batch size (number of graphs per batch)."""

    learning_rate: float = 1e-3
    """Initial AdamW learning rate."""

    weight_decay: float = 1e-4
    """AdamW L2 regularization."""

    gradient_clip_val: float = 1.0
    """Gradient norm clipping. 0.0 = disabled."""

    mixed_precision: bool = True
    """Enable AMP mixed precision on CUDA."""

    # ── Early stopping ────────────────────────────────────────────────────────
    early_stopping: bool = True
    """Enable early stopping on val AUC."""

    early_stopping_patience: int = 10
    """Epochs with no improvement before stopping."""

    early_stopping_min_delta: float = 1e-4
    """Minimum improvement to count as improvement."""

    # ── Data ──────────────────────────────────────────────────────────────────
    num_workers: int = 0
    """DataLoader workers. 0 = main process (safe on Windows)."""

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed: int = 42
    """Random seed."""

    # ── Logging ───────────────────────────────────────────────────────────────
    log_every_n_epochs: int = 1

    # ── Persistence ───────────────────────────────────────────────────────────
    checkpoint_dir: str = "models/gnn"

    def to_dict(self) -> dict:
        """Return flat dict for MLflow param logging."""
        return {
            "k_neighbors": self.k_neighbors,
            "n_particles": self.n_particles,
            "n_node_features": self.n_node_features,
            "n_edge_conv_layers": self.n_edge_conv_layers,
            "hidden_dim": self.hidden_dim,
            "mlp_head_dims": self.mlp_head_dims,  # list, not str — required for load()
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip_val": self.gradient_clip_val,
            "mixed_precision": self.mixed_precision,
            "early_stopping_patience": self.early_stopping_patience,
            "seed": self.seed,
        }

    def config_hash(self) -> str:
        """SHA-256 hash for cache keying."""
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode()
        ).hexdigest()[:12]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GNNConfig":
        """Load GNNConfig from a YAML file."""
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f)
        # Filter to known fields only
        known = {k for k in cls.__dataclass_fields__}
        filtered = {k: v for k, v in raw.items() if k in known}
        return cls(**filtered)
