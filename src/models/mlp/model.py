"""
Deep MLP for particle physics event classification.

Architecture:
    Input(n_features)
    → Linear(512) → BatchNorm(512) → ReLU → Dropout(0.3)
    → Linear(256) → BatchNorm(256) → ReLU → Dropout(0.3)
    → Linear(128) → BatchNorm(128) → ReLU → Dropout(0.2)
    → Linear(64)  → ReLU
    → Linear(1)   → Sigmoid

Inherits from BaseModel for consistent interface with all other architectures.

Expected performance on HIGGS dataset (500k training events):
    Val AUC ≈ 0.81 (comparable to original paper's MLP result)

Reference:
    Baldi et al. (2014) achieved AUC ~0.81 with a 5-layer MLP on 5M events.
    We use 500k events for speed; full 11M gives ~0.825.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.base_model import BaseModel
from src.models.mlp.config import MLPConfig
from src.utils.logging_config import get_logger

log = get_logger(__name__)


class MLPBlock(nn.Module):
    """
    Single hidden layer block: Linear → [BatchNorm] → Activation → [Dropout].

    Args:
        in_features:  Input dimension.
        out_features: Output dimension.
        batch_norm:   Apply BatchNorm before activation.
        dropout:      Dropout probability. 0.0 = no dropout.
        activation:   Activation function (default ReLU).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        batch_norm: bool = True,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_features, out_features)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_features))
        layers.append(activation or nn.ReLU(inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeepMLP(nn.Module):
    """
    Deep MLP PyTorch module.

    The network is built dynamically from MLPConfig, allowing
    architecture search without code changes.

    Args:
        config: MLPConfig instance defining the architecture.

    Example:
        config = MLPConfig(input_dim=28, hidden_dims=[512, 256, 128, 64])
        model = DeepMLP(config)
        x = torch.randn(64, 28)
        scores = model(x)  # shape (64, 1), values in [0, 1]
    """

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        in_dim = config.input_dim

        for i, (hidden_dim, dropout_rate) in enumerate(
            zip(config.hidden_dims, config.dropout_rates)
        ):
            layers.append(
                MLPBlock(
                    in_features=in_dim,
                    out_features=hidden_dim,
                    batch_norm=config.batch_norm,
                    dropout=dropout_rate,
                    activation=nn.ReLU(inplace=True),
                )
            )
            in_dim = hidden_dim

        # Output layer: Linear → Sigmoid
        layers.append(nn.Linear(in_dim, config.output_dim))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming uniform (good for ReLU networks)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch_size, n_features).

        Returns:
            Output tensor, shape (batch_size, 1), values in [0, 1].
        """
        return self.network(x)

    def n_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_str(self) -> str:
        """Return human-readable architecture description."""
        cfg = self.config
        layers = [f"Input({cfg.input_dim})"]
        for h, d in zip(cfg.hidden_dims, cfg.dropout_rates):
            bn_str = "+BN" if cfg.batch_norm else ""
            drop_str = f"+Drop({d})" if d > 0 else ""
            layers.append(f"Linear({h}){bn_str}+ReLU{drop_str}")
        layers.append(f"Linear({cfg.output_dim})+Sigmoid")
        return " → ".join(layers)


class MLPModel(BaseModel):
    """
    Deep MLP model for particle physics event classification.

    Wraps the PyTorch DeepMLP module in the BaseModel interface so it
    is fully interchangeable with BDT, GNN, and Transformer models.

    Args:
        config: MLPConfig. If None, uses defaults.

    Example:
        model = MLPModel(MLPConfig(input_dim=28))
        history = model.fit(X_train, y_train, X_val, y_val)
        scores = model.predict_proba(X_test)  # shape (n_test,)
    """

    def __init__(self, config: MLPConfig | None = None) -> None:
        super().__init__(model_name="mlp")
        self.config = config or MLPConfig()
        self._net: DeepMLP | None = None
        self._device: torch.device | None = None

    def _build(self, input_dim: int) -> DeepMLP:
        """Instantiate the network with the correct input dimension."""
        self.config.input_dim = input_dim
        net = DeepMLP(self.config)
        log.info(
            "MLP built",
            architecture=net.architecture_str(),
            n_parameters=f"{net.n_parameters():,}",
        )
        return net

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Train the MLP. Delegates to MLPTrainer to keep model.py clean.

        Args:
            X_train, y_train: Training data.
            X_val, y_val:     Validation data.
            optuna_trial:     Optional Optuna Trial for pruning integration.
                              Pass as keyword arg: model.fit(..., optuna_trial=trial)

        Returns:
            best_val_auc (float) — the best validation AUC achieved during training.
            Full history is available via model._metadata after fitting.
        """
        from src.models.mlp.trainer import MLPTrainer

        optuna_trial = kwargs.pop("optuna_trial", None)

        trainer = MLPTrainer(config=self.config)
        history = trainer.train(
            self, X_train, y_train, X_val, y_val,
            optuna_trial=optuna_trial,
        )
        self._is_fitted = True
        self._fit_time_s = history.get("fit_time_s")
        self._metadata.update({
            "best_val_auc": history.get("best_val_auc"),
            "best_epoch": history.get("best_epoch"),
            "n_parameters": self._net.n_parameters() if self._net else None,
        })
        # Return best_val_auc so optimizer objective can use it directly
        return history.get("best_val_auc", 0.0)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Return P(signal) for each event.

        Args:
            X: Features, shape (n_events, n_features).

        Returns:
            1D float32 array of shape (n_events,), values in [0, 1].
        """
        if not self._is_fitted or self._net is None:
            raise RuntimeError("MLPModel has not been fitted. Call .fit() first.")

        self._net.eval()
        X_np = self._to_numpy(X)
        X_tensor = torch.from_numpy(X_np).to(self._device)

        with torch.no_grad():
            scores = self._net(X_tensor).squeeze(-1)

        return scores.cpu().numpy().astype(np.float32)

    def save(self, path: str | Path) -> None:
        """Save model weights, config, and metadata to directory."""
        if self._net is None:
            raise RuntimeError("Cannot save: model has not been built/fitted.")

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save PyTorch weights
        torch.save(self._net.state_dict(), save_dir / "weights.pt")

        # Save config — store lists as native JSON arrays (not string reprs)
        config_dict = {
            "input_dim": self.config.input_dim,
            "hidden_dims": list(self.config.hidden_dims),
            "dropout_rates": list(self.config.dropout_rates),
            "batch_norm": self.config.batch_norm,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "scheduler": self.config.scheduler_name,
            "seed": self.config.seed,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        # Save metadata
        self._save_metadata(save_dir)
        log.info("MLP saved", path=str(save_dir))

    def load(self, path: str | Path) -> None:
        """Load model weights and config from directory."""
        import ast

        load_dir = Path(path)

        # Load config
        with open(load_dir / "config.json") as f:
            config_dict = json.load(f)

        # Restore the full architecture config (not just input_dim)
        def parse_list(val):
            """Parse a list that may be stored as a string repr."""
            if isinstance(val, list):
                return val
            if isinstance(val, str):
                return ast.literal_eval(val)
            return val

        self.config.input_dim = int(config_dict.get("input_dim", self.config.input_dim))
        if "hidden_dims" in config_dict:
            self.config.hidden_dims = parse_list(config_dict["hidden_dims"])
        if "dropout_rates" in config_dict:
            self.config.dropout_rates = parse_list(config_dict["dropout_rates"])
        if "batch_norm" in config_dict:
            self.config.batch_norm = bool(config_dict["batch_norm"])

        self._net = DeepMLP(self.config)

        # Load weights
        state_dict = torch.load(
            load_dir / "weights.pt",
            map_location="cpu",
            weights_only=True,
        )
        self._net.load_state_dict(state_dict)

        # Restore metadata
        meta = self._load_metadata(load_dir)
        self._metadata.update(meta)
        self._is_fitted = True
        self._device = torch.device("cpu")

        log.info("MLP loaded", path=str(load_dir))

    def summary(self) -> dict[str, Any]:
        base = super().summary()
        if self._net is not None:
            base["n_parameters"] = self._net.n_parameters()
            base["architecture"] = self._net.architecture_str()
        base["config"] = self.config.to_dict()
        return base
