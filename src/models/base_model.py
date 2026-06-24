"""
Abstract base class for all model architectures.

Every model in the Model Zoo (MLP, BDT, GNN, Transformer, Normalizing Flow)
inherits from BaseModel and exposes an identical interface:
    .fit()            → Train the model
    .predict_proba()  → Return P(signal) for each event
    .predict()        → Return binary label (0/1)
    .save()           → Persist model to disk
    .load()           → Restore model from disk
    .summary()        → Return a dict of model metadata

This uniform interface is what allows:
- The training pipeline to call any model with the same code
- The FastAPI endpoint to swap models without code changes
- The experiment registry to compare all architectures identically

Usage:
    class MyModel(BaseModel):
        def _build(self) -> None: ...
        def fit(self, X_train, y_train, X_val, y_val) -> dict: ...
        def predict_proba(self, X) -> np.ndarray: ...
        def save(self, path) -> None: ...
        def load(self, path) -> None: ...
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    Abstract base class for all particle physics classifier models.

    All models must implement the abstract methods below.
    The base class provides shared utilities: path management,
    metadata serialization, and a default .predict() method.

    Args:
        model_name: Human-readable name for this model architecture.
        version:    Model version string (e.g., "1.0").
    """

    def __init__(self, model_name: str, version: str = "1.0") -> None:
        self.model_name = model_name
        self.version = version
        self._is_fitted = False
        self._fit_time_s: float | None = None
        self._metadata: dict[str, Any] = {}

    # ── Abstract interface (must implement) ────────────────────────────────────

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> float:
        """
        Train the model.

        Args:
            X_train: Training features, shape (n_train, n_features).
            y_train: Training labels, shape (n_train,). Values in {0, 1}.
            X_val:   Validation features, shape (n_val, n_features).
            y_val:   Validation labels, shape (n_val,).
            **kwargs: Model-specific kwargs (e.g., callbacks, sample_weights).

        Returns:
            best_val_auc (float): Best validation AUC achieved during training.
        """
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Return P(signal) for each event.

        Args:
            X: Features, shape (n_events, n_features).

        Returns:
            1D float array of shape (n_events,), values in [0, 1].
            Higher value = more signal-like.
        """
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """
        Persist model weights and config to disk.

        Args:
            path: Directory to save model artifacts.
                  Creates the directory if it doesn't exist.
        """
        ...

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """
        Restore model from disk.

        Args:
            path: Directory containing saved model artifacts.
        """
        ...

    # ── Provided implementations ────────────────────────────────────────────────

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Return binary predictions.

        Args:
            X:         Features, shape (n_events, n_features).
            threshold: Decision threshold. Default 0.5.

        Returns:
            Int array of shape (n_events,), values in {0, 1}.
        """
        if not self._is_fitted:
            raise RuntimeError(f"{self.model_name} has not been fitted. Call .fit() first.")
        scores = self.predict_proba(X)
        return (scores >= threshold).astype(int)

    def summary(self) -> dict[str, Any]:
        """
        Return model metadata as a dict.

        Subclasses should call super().summary() and update with their own info.
        """
        return {
            "model_name": self.model_name,
            "version": self.version,
            "is_fitted": self._is_fitted,
            "fit_time_s": self._fit_time_s,
            **self._metadata,
        }

    # ── Metadata helpers ────────────────────────────────────────────────────────

    def _save_metadata(self, directory: Path) -> None:
        """Save model metadata to a JSON file alongside the model weights."""
        meta = {
            **self.summary(),
            "saved_at": datetime.utcnow().isoformat(),
        }
        with open(directory / "model_metadata.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

    def _load_metadata(self, directory: Path) -> dict[str, Any]:
        """Load model metadata from a JSON file."""
        meta_path = directory / "model_metadata.json"
        if not meta_path.exists():
            return {}
        with open(meta_path) as f:
            return json.load(f)

    # ── Type coercion helpers ────────────────────────────────────────────────────

    @staticmethod
    def _to_numpy(X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Convert DataFrame or ndarray to float32 numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.values.astype(np.float32)
        return np.asarray(X, dtype=np.float32)

    @staticmethod
    def _to_numpy_1d(y: pd.Series | np.ndarray) -> np.ndarray:
        """Convert label Series or ndarray to float32 1D numpy array."""
        if isinstance(y, pd.Series):
            return y.values.astype(np.float32)
        return np.asarray(y, dtype=np.float32)

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name={self.model_name!r}, {status})"
