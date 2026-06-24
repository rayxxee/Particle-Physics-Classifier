"""
Particle Transformer (ParT) for particle physics event classification.

Architecture:
    Input: (batch, n_particles, n_features)  [e.g. (N, 4, 7) for HIGGS]
    → Linear embedding: (batch, n_particles, d_model)
    → Prepend CLS token: (batch, n_particles + 1, d_model)
    → TransformerEncoder × n_encoder_layers  [standard nn.TransformerEncoder]
    → Extract CLS token output: (batch, d_model)
    → MLP head: d_model → head_dims → 1
    → Sigmoid

The CLS token aggregates global event information via cross-attention,
similar to BERT's [CLS] token for sequence classification.

No external dependencies beyond standard PyTorch (nn.TransformerEncoder).
All torch imports are LAZY (inside functions, not module level).

References:
    Qu & Gouskos (2022) "Particle Transformer for Jet Tagging"
    Devlin et al. (2018) "BERT" (CLS token idea)

Usage:
    from src.models.transformer.config import TransformerConfig
    from src.models.transformer.model import TransformerModel

    config = TransformerConfig(n_particles=4, n_features=7, d_model=64)
    model = TransformerModel(config)
    best_auc = model.fit(X_train, y_train, X_val, y_val)
    scores = model.predict_proba(X_test)  # shape (n_test,)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel
from src.models.transformer.config import TransformerConfig
from src.utils.logging_config import get_logger

log = get_logger(__name__)


# ─── PyTorch Module (lazy builds) ─────────────────────────────────────────────

class ParticleTransformerNet:
    """
    Namespace for the PyTorch nn.Module and builder.
    Built lazily to avoid module-level torch import.
    """

    @staticmethod
    def build(config: TransformerConfig):
        """Build the Particle Transformer nn.Module. All torch imports inside."""
        import math

        import torch
        import torch.nn as nn

        class _ParticleTransformer(nn.Module):
            """
            Particle Transformer: CLS-token TransformerEncoder for event classification.

            Sequence format:
                [CLS, particle_0, particle_1, ..., particle_{n-1}]
                Length = n_particles + 1
            """

            def __init__(self, cfg: TransformerConfig) -> None:
                super().__init__()
                self.cfg = cfg

                # ── Input embedding ───────────────────────────────────────────
                self.input_proj = nn.Linear(cfg.n_features, cfg.d_model)

                # ── Learned CLS token ─────────────────────────────────────────
                self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
                nn.init.trunc_normal_(self.cls_token, std=0.02)

                # ── Positional encoding (learnable) ───────────────────────────
                seq_len = cfg.n_particles + 1  # +1 for CLS
                self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, cfg.d_model))
                nn.init.trunc_normal_(self.pos_embed, std=0.02)

                # ── Transformer encoder ───────────────────────────────────────
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=cfg.d_model,
                    nhead=cfg.n_heads,
                    dim_feedforward=cfg.dim_feedforward,
                    dropout=cfg.dropout,
                    activation="gelu",
                    batch_first=True,  # (batch, seq, d_model)
                    norm_first=True,   # Pre-LN: more stable training
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=cfg.n_encoder_layers,
                    enable_nested_tensor=False,  # Avoids warning with batch_first
                )

                # ── Classification MLP head ───────────────────────────────────
                head_layers: list[nn.Module] = []
                h_in = cfg.d_model
                for h_dim in cfg.head_dims:
                    head_layers.extend([
                        nn.LayerNorm(h_in),
                        nn.Linear(h_in, h_dim),
                        nn.GELU(),
                        nn.Dropout(p=cfg.head_dropout),
                    ])
                    h_in = h_dim
                head_layers.extend([
                    nn.LayerNorm(h_in),
                    nn.Linear(h_in, 1),
                    nn.Sigmoid(),
                ])
                self.head = nn.Sequential(*head_layers)

                self._init_weights()

            def _init_weights(self) -> None:
                """Initialize linear layers with Xavier uniform."""
                for module in self.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """
                Forward pass.

                Args:
                    x: Input tensor, shape (batch_size, n_particles, n_features).

                Returns:
                    Output tensor, shape (batch_size, 1), values in [0, 1].
                """
                batch_size = x.shape[0]

                # ─ Input embedding ─
                x = self.input_proj(x)  # (B, n_particles, d_model)

                # ─ Prepend CLS token ─
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d_model)
                x = torch.cat([cls_tokens, x], dim=1)  # (B, n_particles+1, d_model)

                # ─ Add positional embeddings ─
                x = x + self.pos_embed  # (B, n_particles+1, d_model)

                # ─ Transformer encoder ─
                x = self.encoder(x)  # (B, n_particles+1, d_model)

                # ─ Extract CLS token output ─
                cls_out = x[:, 0, :]  # (B, d_model)

                # ─ Classification head ─
                return self.head(cls_out)  # (B, 1)

            def n_parameters(self) -> int:
                return sum(p.numel() for p in self.parameters() if p.requires_grad)

            def architecture_str(self) -> str:
                cfg = self.cfg
                return (
                    f"Input({cfg.n_particles}×{cfg.n_features})"
                    f" → Embed({cfg.d_model})"
                    f" → CLS+TransformerEncoder×{cfg.n_encoder_layers}"
                    f"(heads={cfg.n_heads}, ffn={cfg.dim_feedforward})"
                    f" → MLP{cfg.head_dims}"
                    f" → Sigmoid"
                )

        return _ParticleTransformer(config)


# ─── TransformerModel (BaseModel interface) ────────────────────────────────────

class TransformerModel(BaseModel):
    """
    Particle Transformer for particle physics event classification.

    Wraps the standard PyTorch TransformerEncoder in the BaseModel interface.
    Input X:
      - 2D flat (n_events, n_features): reshaped to (n_events, n_particles, n_node_features)
      - 3D already (n_events, n_particles, n_node_features): used as-is

    For HIGGS: 28 features → 4 particles × 7 features.

    Args:
        config: TransformerConfig. Uses defaults if None.
    """

    def __init__(self, config: TransformerConfig | None = None) -> None:
        self.config = config or TransformerConfig()
        super().__init__(model_name="transformer")
        self._net: Any = None
        self._device: Any = None

    def _reshape_input(self, X: np.ndarray) -> np.ndarray:
        """
        Reshape flat 2D input to 3D (n_events, n_particles, n_features).
        Handles truncation/padding if feature count doesn't divide evenly.
        """
        if X.ndim == 3:
            return X
        n_events = X.shape[0]
        n_p = self.config.n_particles
        n_f = self.config.n_features
        expected = n_p * n_f
        if X.shape[1] != expected:
            if X.shape[1] > expected:
                X = X[:, :expected]
            else:
                pad = np.zeros((n_events, expected - X.shape[1]), dtype=X.dtype)
                X = np.hstack([X, pad])
        return X.reshape(n_events, n_p, n_f)

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> float:
        """
        Train the Particle Transformer.

        Args:
            X_train, y_train: Training data. X shape (n, n_features) or (n, n_p, n_f).
            X_val, y_val:     Validation data.
            **kwargs:         Ignored for interface compatibility.

        Returns:
            best_val_auc (float)
        """
        from src.models.transformer.trainer import TransformerTrainer

        trainer = TransformerTrainer(config=self.config)
        history = trainer.train(self, X_train, y_train, X_val, y_val)

        self._is_fitted = True
        self._fit_time_s = history.get("fit_time_s")
        self._metadata.update({
            "best_val_auc": history.get("best_val_auc"),
            "best_epoch": history.get("best_epoch"),
            "n_parameters": self._net.n_parameters() if self._net else None,
        })

        return history.get("best_val_auc", 0.0)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Return P(signal) for each event.

        Args:
            X: Features, shape (n_events, n_features) or (n_events, n_p, n_f).

        Returns:
            1D float32 array of shape (n_events,), values in [0, 1].
        """
        if not self._is_fitted or self._net is None:
            raise RuntimeError(
                "TransformerModel has not been fitted. Call .fit() first."
            )

        import torch

        X_np = self._to_numpy(X) if hasattr(X, "values") else np.asarray(X, dtype=np.float32)
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)
        X_3d = self._reshape_input(X_np)

        self._net.eval()
        all_scores = []
        bs = self.config.batch_size * 2

        for i in range(0, len(X_3d), bs):
            batch_np = X_3d[i : i + bs]
            batch_t = torch.from_numpy(batch_np).to(self._device)
            with torch.no_grad():
                scores = self._net(batch_t).squeeze(-1)
            all_scores.append(scores.cpu().numpy())

        return np.concatenate(all_scores, axis=0).astype(np.float32)

    def save(self, path: str | Path) -> None:
        """Save model weights + config to directory."""
        import torch

        if self._net is None:
            raise RuntimeError("Cannot save: model has not been fitted.")

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self._net.state_dict(), save_dir / "weights.pt")

        with open(save_dir / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        self._save_metadata(save_dir)
        log.info("Transformer saved", path=str(save_dir))

    def load(self, path: str | Path) -> None:
        """Load model weights + config from directory."""
        import torch

        load_dir = Path(path)

        with open(load_dir / "config.json") as f:
            cfg_dict = json.load(f)

        # Reconstruct config from saved params — must include ALL architecture fields
        self.config = TransformerConfig(
            n_particles=cfg_dict.get("n_particles", self.config.n_particles),
            n_features=cfg_dict.get("n_features", self.config.n_features),
            d_model=cfg_dict.get("d_model", self.config.d_model),
            n_heads=cfg_dict.get("n_heads", self.config.n_heads),
            n_encoder_layers=cfg_dict.get("n_encoder_layers", self.config.n_encoder_layers),
            dim_feedforward=cfg_dict.get("dim_feedforward", self.config.dim_feedforward),
            dropout=cfg_dict.get("dropout", self.config.dropout),
            head_dims=cfg_dict.get("head_dims", self.config.head_dims),
            head_dropout=cfg_dict.get("head_dropout", self.config.head_dropout),
            seed=cfg_dict.get("seed", self.config.seed),
        )

        self._net = ParticleTransformerNet.build(self.config)
        state_dict = torch.load(
            load_dir / "weights.pt",
            map_location="cpu",
            weights_only=True,
        )
        self._net.load_state_dict(state_dict)
        self._device = torch.device("cpu")

        meta = self._load_metadata(load_dir)
        self._metadata.update(meta)
        self._is_fitted = True

        log.info("Transformer loaded", path=str(load_dir))

    def summary(self) -> dict[str, Any]:
        base = super().summary()
        if self._net is not None:
            base["n_parameters"] = self._net.n_parameters()
            base["architecture"] = self._net.architecture_str()
        base["config"] = self.config.to_dict()
        return base
