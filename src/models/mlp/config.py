"""
MLP model configuration using dataclasses + OmegaConf.

Defines all hyperparameters for the deep MLP architecture.
Configs can be loaded from YAML, overridden via CLI, or constructed in code.

Usage:
    from src.models.mlp.config import MLPConfig

    # Default config
    config = MLPConfig()

    # From YAML file
    config = MLPConfig.from_yaml("configs/mlp_default.yaml")

    # Programmatic override
    config = MLPConfig(hidden_dims=[1024, 512, 256], dropout_rates=[0.4, 0.4, 0.3])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from omegaconf import OmegaConf


@dataclass
class MLPConfig:
    """
    Full hyperparameter specification for the Deep MLP.

    Architecture:
        Input(n) → [Linear → BatchNorm → ReLU → Dropout] × n_layers → Linear(1) → Sigmoid

    Defaults are tuned for the HIGGS dataset with n_samples ≈ 500k.
    """

    # ── Architecture ──────────────────────────────────────────────────────────
    input_dim: int = 28
    """Number of input features. Set automatically from data in trainer."""

    hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128, 64])
    """Hidden layer widths. Length determines number of hidden layers."""

    output_dim: int = 1
    """Output dimension. Always 1 for binary classification with Sigmoid."""

    batch_norm: bool = True
    """Apply BatchNorm after each Linear layer (before activation)."""

    dropout_rates: list[float] = field(default_factory=lambda: [0.3, 0.3, 0.2, 0.0])
    """Dropout rates per hidden layer. Must match len(hidden_dims).
    Last layer usually has 0.0 (no dropout before output)."""

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = 100
    """Maximum training epochs."""

    batch_size: int = 4096
    """Training batch size. Large batches work well with BatchNorm."""

    learning_rate: float = 1e-3
    """Initial learning rate for AdamW."""

    weight_decay: float = 1e-4
    """L2 regularization (weight decay) for AdamW."""

    gradient_clip_val: float = 1.0
    """Gradient norm clipping. Set to 0.0 to disable."""

    mixed_precision: bool = True
    """Enable AMP (automatic mixed precision). Auto-disabled on CPU/MPS."""

    # ── Scheduler ─────────────────────────────────────────────────────────────
    scheduler_name: Literal["cosine_annealing", "reduce_on_plateau", "none"] = "cosine_annealing"
    """Learning rate scheduler."""

    scheduler_T_max: int = 100
    """CosineAnnealingLR: period length in epochs (typically = total epochs)."""

    scheduler_eta_min: float = 1e-6
    """CosineAnnealingLR: minimum learning rate."""

    scheduler_patience: int = 5
    """ReduceLROnPlateau: epochs with no improvement before reducing LR."""

    scheduler_factor: float = 0.5
    """ReduceLROnPlateau: LR reduction factor."""

    # ── Early Stopping ────────────────────────────────────────────────────────
    early_stopping: bool = True
    """Enable early stopping."""

    early_stopping_patience: int = 10
    """Epochs with no improvement before stopping."""

    early_stopping_min_delta: float = 1e-4
    """Minimum improvement to count as improvement."""

    early_stopping_monitor: Literal["val_auc", "val_loss"] = "val_auc"
    """Metric to monitor for early stopping."""

    # ── Class Imbalance ───────────────────────────────────────────────────────
    class_weights: str = "balanced"
    """
    Class weights for the loss function.
    "balanced": compute weights from label frequencies (1 / class_freq).
    "none":     no weighting.
    """

    # ── Data ──────────────────────────────────────────────────────────────────
    num_workers: int = 0
    """DataLoader workers. 0 = main process (safe on Windows)."""

    pin_memory: bool = True
    """Pin DataLoader memory to GPU (speeds up CUDA transfers)."""

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed: int = 42
    """Random seed for all PyTorch, NumPy, and Python RNG."""

    # ── Logging ───────────────────────────────────────────────────────────────
    log_every_n_epochs: int = 1
    """Log metrics every N epochs during training."""

    # ── Persistence ───────────────────────────────────────────────────────────
    checkpoint_dir: str = "models/mlp"
    """Directory to save model checkpoints."""

    def __post_init__(self) -> None:
        """Validate config consistency after construction."""
        if len(self.dropout_rates) != len(self.hidden_dims):
            raise ValueError(
                f"dropout_rates length ({len(self.dropout_rates)}) must match "
                f"hidden_dims length ({len(self.hidden_dims)}). "
                f"Got hidden_dims={self.hidden_dims}, dropout_rates={self.dropout_rates}"
            )
        if not (0.0 < self.learning_rate < 1.0):
            raise ValueError(f"learning_rate must be in (0, 1). Got {self.learning_rate}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MLPConfig":
        """Load MLPConfig from a YAML file (using OmegaConf)."""
        cfg = OmegaConf.load(path)

        arch = OmegaConf.select(cfg, "architecture", default={})
        train = OmegaConf.select(cfg, "training", default={})
        sched = OmegaConf.select(cfg, "training.scheduler", default={})
        es = OmegaConf.select(cfg, "training.early_stopping", default={})

        return cls(
            input_dim=OmegaConf.select(arch, "input_dim", default=28),
            hidden_dims=list(OmegaConf.select(arch, "hidden_dims", default=[512, 256, 128, 64])),
            batch_norm=OmegaConf.select(arch, "batch_norm", default=True),
            dropout_rates=list(OmegaConf.select(arch, "dropout_rates", default=[0.3, 0.3, 0.2, 0.0])),
            epochs=OmegaConf.select(train, "epochs", default=100),
            batch_size=OmegaConf.select(train, "batch_size", default=4096),
            learning_rate=float(OmegaConf.select(train, "learning_rate", default=1e-3)),
            weight_decay=float(OmegaConf.select(train, "weight_decay", default=1e-4)),
            gradient_clip_val=float(OmegaConf.select(train, "gradient_clip_val", default=1.0)),
            mixed_precision=OmegaConf.select(train, "mixed_precision", default=True),
            scheduler_name=OmegaConf.select(sched, "name", default="cosine_annealing"),
            scheduler_T_max=OmegaConf.select(sched, "T_max", default=100),
            scheduler_eta_min=float(OmegaConf.select(sched, "eta_min", default=1e-6)),
            early_stopping=OmegaConf.select(es, "enabled", default=True),
            early_stopping_patience=OmegaConf.select(es, "patience", default=10),
            early_stopping_min_delta=float(OmegaConf.select(es, "min_delta", default=1e-4)),
            early_stopping_monitor=OmegaConf.select(es, "monitor", default="val_auc"),
            class_weights=OmegaConf.select(train, "class_weights", default="balanced"),
            seed=OmegaConf.select(train, "seed", default=42),
        )

    def to_dict(self) -> dict:
        """Return config as a flat dict (for MLflow param logging)."""
        return {
            "input_dim": self.input_dim,
            "hidden_dims": str(self.hidden_dims),
            "batch_norm": self.batch_norm,
            "dropout_rates": str(self.dropout_rates),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip_val": self.gradient_clip_val,
            "mixed_precision": self.mixed_precision,
            "scheduler": self.scheduler_name,
            "early_stopping_patience": self.early_stopping_patience,
            "class_weights": self.class_weights,
            "seed": self.seed,
        }
