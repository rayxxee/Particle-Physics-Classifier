from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from src.models.base_model import BaseModel
from src.models.normalizing_flow.config import NormalizingFlowConfig
from src.utils.logging_config import get_logger

log = get_logger(__name__)

class MLPNet(nn.Module):
    """Simple MLP to compute translation and scaling factors for RealNVP."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers = []
        dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(dim, h))
            layers.append(nn.ReLU())
            dim = h
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)
        
        # Initialize final layer to zero so coupling layer initially does nothing
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)

class AffineCouplingLayer(nn.Module):
    """Affine coupling layer for RealNVP."""
    def __init__(self, dim: int, hidden_dims: list[int], mask: torch.Tensor):
        super().__init__()
        self.register_buffer('mask', mask)
        self.net = MLPNet(dim, dim * 2, hidden_dims)

    def forward(self, x):
        x_masked = x * self.mask
        out = self.net(x_masked)
        s, t = out.chunk(2, dim=1)
        s = torch.tanh(s) * (1.0 - self.mask) # scale factor
        t = t * (1.0 - self.mask)             # translation factor
        
        z = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det = s.sum(dim=1)
        return z, log_det

    def inverse(self, z):
        z_masked = z * self.mask
        out = self.net(z_masked)
        s, t = out.chunk(2, dim=1)
        s = torch.tanh(s) * (1.0 - self.mask)
        t = t * (1.0 - self.mask)
        
        x = z_masked + (1 - self.mask) * ((z - t) * torch.exp(-s))
        log_det = -s.sum(dim=1)
        return x, log_det

class RealNVP(nn.Module):
    """RealNVP Normalizing Flow model."""
    def __init__(self, config: NormalizingFlowConfig):
        super().__init__()
        self.config = config
        self.dim = config.input_dim
        
        self.prior = MultivariateNormal(torch.zeros(self.dim), torch.eye(self.dim))
        self.coupling_layers = nn.ModuleList()
        
        for i in range(config.num_coupling_layers):
            mask = torch.zeros(self.dim)
            if i % 2 == 0:
                mask[::2] = 1.0
            else:
                mask[1::2] = 1.0
            self.coupling_layers.append(
                AffineCouplingLayer(self.dim, config.hidden_dims, mask)
            )

    def forward(self, x):
        z = x
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        for layer in self.coupling_layers:
            z, log_det = layer(z)
            log_det_total += log_det
        return z, log_det_total

    def inverse(self, z):
        x = z
        log_det_total = torch.zeros(z.shape[0], device=z.device)
        for layer in reversed(self.coupling_layers):
            x, log_det = layer.inverse(x)
            log_det_total += log_det
        return x, log_det_total

    def log_prob(self, x):
        z, log_det = self.forward(x)
        prior_log_prob = self.prior.log_prob(z.cpu()).to(x.device)
        return prior_log_prob + log_det

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_str(self) -> str:
        return f"RealNVP({self.config.num_coupling_layers} layers, hidden={self.config.hidden_dims})"


class NormalizingFlowModel(BaseModel):
    """
    Normalizing Flow model for anomaly-based particle physics classification.
    
    Trained only on background (y=0) to model its density.
    predict_proba returns normalized anomaly score (1.0 = most anomalous/signal).
    """
    def __init__(self, config: NormalizingFlowConfig | None = None) -> None:
        super().__init__(model_name="normalizing_flow")
        self.config = config or NormalizingFlowConfig()
        self._net: RealNVP | None = None
        self._device: torch.device | None = None
        self._min_log_prob = 0.0
        self._max_log_prob = 0.0

    def _build(self, input_dim: int) -> RealNVP:
        self.config.input_dim = input_dim
        net = RealNVP(self.config)
        log.info("Normalizing Flow built", architecture=net.architecture_str(), n_parameters=net.n_parameters())
        return net

    def fit(self, X_train, y_train, X_val, y_val, **kwargs) -> float:
        from src.models.normalizing_flow.trainer import NFTrainer
        
        # Filter for background only
        y_train_np = self._to_numpy_1d(y_train)
        y_val_np = self._to_numpy_1d(y_val)
        
        X_train_bg = self._to_numpy(X_train)[y_train_np == 0]
        X_val_bg = self._to_numpy(X_val)[y_val_np == 0]

        trainer = NFTrainer(config=self.config)
        history = trainer.train(self, X_train_bg, X_val_bg, X_val, y_val, **kwargs)
        
        self._is_fitted = True
        self._fit_time_s = history.get("fit_time_s")
        self._metadata.update({
            "best_val_loss": history.get("best_val_loss"),
            "best_epoch": history.get("best_epoch"),
            "n_parameters": self._net.n_parameters() if self._net else None,
            "min_log_prob": float(history.get("min_log_prob", 0.0)),
            "max_log_prob": float(history.get("max_log_prob", 0.0)),
        })
        self._min_log_prob = self._metadata["min_log_prob"]
        self._max_log_prob = self._metadata["max_log_prob"]
        
        return history.get("best_val_loss", 0.0)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted or self._net is None:
            raise RuntimeError("NormalizingFlowModel has not been fitted.")

        self._net.eval()
        X_np = self._to_numpy(X)
        X_tensor = torch.from_numpy(X_np).to(self._device)

        with torch.no_grad():
            log_probs = self._net.log_prob(X_tensor)
            
        log_probs_np = log_probs.cpu().numpy().astype(np.float32)
        
        # Anomaly score: 1.0 (anomalous, signal) to 0.0 (normal, background)
        # We invert the log_prob so that low log_prob = high anomaly score
        normalized = (self._max_log_prob - log_probs_np) / (self._max_log_prob - self._min_log_prob + 1e-6)
        scores = np.clip(normalized, 0.0, 1.0)
        
        return scores

    def save(self, path: str | Path) -> None:
        if self._net is None:
            raise RuntimeError("Cannot save: model has not been built/fitted.")

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(), save_dir / "weights.pt")
        
        with open(save_dir / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        self._save_metadata(save_dir)
        log.info("Normalizing Flow saved", path=str(save_dir))

    def load(self, path: str | Path) -> None:
        load_dir = Path(path)
        with open(load_dir / "config.json") as f:
            config_dict = json.load(f)

        self.config.input_dim = int(config_dict.get("input_dim", self.config.input_dim))
        self.config.num_coupling_layers = int(config_dict.get("num_coupling_layers", self.config.num_coupling_layers))
        if "hidden_dims" in config_dict:
            self.config.hidden_dims = config_dict["hidden_dims"]

        self._net = RealNVP(self.config)
        state_dict = torch.load(load_dir / "weights.pt", map_location="cpu", weights_only=True)
        self._net.load_state_dict(state_dict)

        meta = self._load_metadata(load_dir)
        self._metadata.update(meta)
        self._min_log_prob = meta.get("min_log_prob", 0.0)
        self._max_log_prob = meta.get("max_log_prob", 0.0)
        self._is_fitted = True
        self._device = torch.device("cpu")
        log.info("Normalizing Flow loaded", path=str(load_dir))

    def summary(self) -> dict[str, Any]:
        base = super().summary()
        if self._net is not None:
            base["n_parameters"] = self._net.n_parameters()
            base["architecture"] = self._net.architecture_str()
        base["config"] = self.config.to_dict()
        return base
