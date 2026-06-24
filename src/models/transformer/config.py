"""
Particle Transformer (ParT) model configuration.

The Particle Transformer treats each event as a sequence of particles:
  - Input: (batch, n_particles, n_features) — same reshaping as GNN
  - Architecture: CLS token prepended → TransformerEncoder × n_layers → CLS output → MLP
  - No external dependencies beyond standard PyTorch (nn.TransformerEncoder)

HIGGS dataset input:
    Raw: (n_events, 28) → reshaped to (n_events, 4, 7)
    n_particles = 4, n_features = 7

Reference:
    Qu & Gouskos (2022) "Particle Transformer for Jet Tagging"
    https://arxiv.org/abs/2202.03772

Usage:
    from src.models.transformer.config import TransformerConfig

    config = TransformerConfig(n_particles=4, n_features=7, d_model=64)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TransformerConfig:
    """
    Full hyperparameter specification for the Particle Transformer.

    Architecture:
        Input(n_events, n_particles, n_features)
        → Linear embedding to d_model
        → Prepend CLS token
        → TransformerEncoder × n_encoder_layers
        → Extract CLS token output
        → MLP head
        → Sigmoid
    """

    # ── Input format ──────────────────────────────────────────────────────────
    n_particles: int = 4
    """Number of particles per event (sequence length). HIGGS: 4."""

    n_features: int = 7
    """Feature dimension per particle. HIGGS: 28 / 4 = 7."""

    # ── Transformer architecture ──────────────────────────────────────────────
    d_model: int = 64
    """Model dimension (embedding size). Must be divisible by n_heads."""

    n_heads: int = 4
    """Number of attention heads. d_model must be divisible by n_heads."""

    n_encoder_layers: int = 4
    """Number of TransformerEncoder layers."""

    dim_feedforward: int = 256
    """Inner dimension of the FFN sub-layer in each encoder block."""

    dropout: float = 0.1
    """Dropout probability in transformer layers."""

    # ── Classification head ───────────────────────────────────────────────────
    head_dims: list[int] = field(default_factory=lambda: [128, 64])
    """Hidden dims of the MLP head after CLS token extraction."""

    head_dropout: float = 0.2
    """Dropout in MLP head."""

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = 60
    """Maximum training epochs."""

    batch_size: int = 2048
    """Training batch size."""

    learning_rate: float = 5e-4
    """AdamW learning rate."""

    weight_decay: float = 1e-4
    """AdamW L2 regularization."""

    gradient_clip_val: float = 1.0
    """Gradient norm clipping. 0.0 = disabled."""

    warmup_epochs: int = 5
    """Linear LR warmup epochs before cosine decay."""

    mixed_precision: bool = True
    """Enable AMP mixed precision on CUDA."""

    # ── Early stopping ────────────────────────────────────────────────────────
    early_stopping: bool = True
    """Enable early stopping on val AUC."""

    early_stopping_patience: int = 12
    """Epochs with no improvement before stopping."""

    early_stopping_min_delta: float = 1e-4
    """Minimum improvement threshold."""

    # ── Data ──────────────────────────────────────────────────────────────────
    num_workers: int = 0
    """DataLoader workers. 0 = main process (safe on Windows)."""

    pin_memory: bool = True
    """Pin DataLoader memory to GPU for faster transfer."""

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed: int = 42

    # ── Logging ───────────────────────────────────────────────────────────────
    log_every_n_epochs: int = 1

    # ── Persistence ───────────────────────────────────────────────────────────
    checkpoint_dir: str = "models/transformer"

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})."
            )

    def to_dict(self) -> dict:
        """Return flat dict for MLflow param logging."""
        return {
            "n_particles": self.n_particles,
            "n_features": self.n_features,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_encoder_layers": self.n_encoder_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "head_dims": self.head_dims,  # list, not str — required for load()
            "head_dropout": self.head_dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip_val": self.gradient_clip_val,
            "warmup_epochs": self.warmup_epochs,
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
    def from_yaml(cls, path: str | Path) -> "TransformerConfig":
        """Load TransformerConfig from YAML file."""
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f)
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in known})
