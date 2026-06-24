"""
BDT (Boosted Decision Tree) model for particle physics event classification.

Supports two backends:
  - XGBoost: `model_type='xgboost'` — GPU-capable via device='cuda'
  - LightGBM: `model_type='lightgbm'` — fast CPU-based gradient boosting

Both backends expose an identical interface via the BaseModel contract.
The sklearn-compatible API means `.fit()` and `.predict_proba()` follow
the same pattern as any scikit-learn estimator, but wrapped here to match
the project's interface (fit → float, predict_proba → 1D ndarray).

Architecture decision:
    BDTs are saved via joblib (model binary) + JSON (config + metadata).
    This avoids XGBoost/LightGBM format incompatibilities between versions.

Usage:
    from src.models.bdt.config import BDTConfig
    from src.models.bdt.model import BDTModel

    config = BDTConfig(model_type="xgboost", n_estimators=500)
    model = BDTModel(config)
    best_auc = model.fit(X_train, y_train, X_val, y_val)
    scores = model.predict_proba(X_test)  # shape (n_test,)
    model.save("models/bdt_run1")
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel
from src.models.bdt.config import BDTConfig
from src.utils.logging_config import get_logger

log = get_logger(__name__)


class BDTModel(BaseModel):
    """
    Gradient Boosted Decision Tree classifier (XGBoost or LightGBM).

    Wraps the scikit-learn compatible BDT APIs in the BaseModel interface,
    providing identical calling conventions to MLP, GNN, and Transformer.

    Args:
        config: BDTConfig instance. Uses defaults (XGBoost) if None.

    Example:
        model = BDTModel(BDTConfig(model_type="xgboost"))
        auc = model.fit(X_train, y_train, X_val, y_val)
        scores = model.predict_proba(X_test)  # 1D array
    """

    def __init__(self, config: BDTConfig | None = None) -> None:
        config = config or BDTConfig()
        super().__init__(model_name=f"bdt_{config.model_type}")
        self.config = config
        self._clf: Any = None  # The underlying XGBoost or LightGBM classifier

    def _build(self) -> Any:
        """Instantiate the underlying classifier from config."""
        if self.config.model_type == "xgboost":
            return self._build_xgboost()
        elif self.config.model_type == "lightgbm":
            return self._build_lightgbm()
        else:
            raise ValueError(
                f"Unknown model_type: {self.config.model_type}. "
                "Must be 'xgboost' or 'lightgbm'."
            )

    def _build_xgboost(self) -> Any:
        """Build XGBoost classifier, selecting GPU if available."""
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError(
                "XGBoost not installed. Run: pip install xgboost"
            ) from e

        # Try CUDA; fall back gracefully to CPU
        device = self._detect_device()
        params = self.config.xgboost_params(device=device)

        clf = xgb.XGBClassifier(**params)
        log.info(
            "XGBoost classifier built",
            device=device,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
        )
        return clf

    def _build_lightgbm(self) -> Any:
        """Build LightGBM classifier."""
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError(
                "LightGBM not installed. Run: pip install lightgbm"
            ) from e

        params = self.config.lightgbm_params()
        clf = lgb.LGBMClassifier(**params)
        log.info(
            "LightGBM classifier built",
            n_estimators=self.config.n_estimators,
            num_leaves=self.config.lgbm_num_leaves,
        )
        return clf

    @staticmethod
    def _detect_device() -> str:
        """Return 'cuda' if a CUDA GPU is available, else 'cpu'."""
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> float:
        """
        Train the BDT with early stopping on validation AUC.

        Args:
            X_train, y_train: Training data.
            X_val, y_val:     Validation data for early stopping.
            **kwargs:         Passed through (ignored for compatibility).

        Returns:
            best_val_auc (float) — best validation AUC achieved.
        """
        from sklearn.metrics import roc_auc_score

        # Convert inputs
        X_tr = self._to_numpy(X_train)
        y_tr = self._to_numpy_1d(y_train)
        X_v = self._to_numpy(X_val)
        y_v = self._to_numpy_1d(y_val)

        # Build classifier
        self._clf = self._build()

        t_start = time.time()

        # Fit with early stopping
        eval_set = [(X_v, y_v)]
        fit_kwargs: dict[str, Any] = {
            "eval_set": eval_set,
        }

        if self.config.model_type == "xgboost":
            # XGBoost >=2.0: early_stopping_rounds is a constructor arg.
            # Only pass eval_set to fit().
            pass
        elif self.config.model_type == "lightgbm":
            # LightGBM uses callbacks for early stopping
            try:
                import lightgbm as lgb
                fit_kwargs["callbacks"] = [
                    lgb.early_stopping(
                        stopping_rounds=self.config.early_stopping_rounds,
                        verbose=self.config.verbose > 0,
                    ),
                    lgb.log_evaluation(period=self.config.verbose),
                ]
            except Exception:
                pass  # callbacks unavailable in older lgb — skip

        self._clf.fit(X_tr, y_tr, **fit_kwargs)

        fit_time = time.time() - t_start

        # Compute best validation AUC
        val_scores = self._clf.predict_proba(X_v)[:, 1]
        best_val_auc = float(roc_auc_score(y_v, val_scores))

        self._is_fitted = True
        self._fit_time_s = fit_time
        self._metadata.update({
            "best_val_auc": best_val_auc,
            "fit_time_s": fit_time,
            "model_type": self.config.model_type,
        })

        log.info(
            "BDT training complete",
            model_type=self.config.model_type,
            best_val_auc=f"{best_val_auc:.4f}",
            fit_time_s=f"{fit_time:.1f}s",
        )

        return best_val_auc

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Return P(signal) for each event.

        Args:
            X: Features, shape (n_events, n_features).

        Returns:
            1D float32 array of shape (n_events,), values in [0, 1].
        """
        if not self._is_fitted or self._clf is None:
            raise RuntimeError(
                f"{self.model_name} has not been fitted. Call .fit() first."
            )
        X_np = self._to_numpy(X)
        scores = self._clf.predict_proba(X_np)[:, 1]
        return scores.astype(np.float32)

    def save(self, path: str | Path) -> None:
        """
        Save model to directory using joblib (binary) + JSON (config + metadata).

        Args:
            path: Directory to save model artifacts. Created if it doesn't exist.
        """
        import joblib

        if self._clf is None:
            raise RuntimeError("Cannot save: model has not been fitted.")

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save the sklearn-compatible classifier binary
        joblib.dump(self._clf, save_dir / "classifier.joblib")

        # Save config
        with open(save_dir / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save metadata
        self._save_metadata(save_dir)
        log.info("BDT saved", path=str(save_dir), model_type=self.config.model_type)

    def load(self, path: str | Path) -> None:
        """
        Load model from directory.

        Args:
            path: Directory containing saved model artifacts.
        """
        import joblib

        load_dir = Path(path)

        # Load config
        cfg_path = load_dir / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg_dict = json.load(f)
            # Reconstruct config from saved params
            self.config = BDTConfig(
                model_type=cfg_dict.get("model_type", self.config.model_type),
                n_estimators=cfg_dict.get("n_estimators", self.config.n_estimators),
                learning_rate=cfg_dict.get("learning_rate", self.config.learning_rate),
                max_depth=cfg_dict.get("max_depth", self.config.max_depth),
                subsample=cfg_dict.get("subsample", self.config.subsample),
                colsample_bytree=cfg_dict.get("colsample_bytree", self.config.colsample_bytree),
                reg_alpha=cfg_dict.get("reg_alpha", self.config.reg_alpha),
                reg_lambda=cfg_dict.get("reg_lambda", self.config.reg_lambda),
                seed=cfg_dict.get("seed", self.config.seed),
            )
            # Update model_name to match loaded type
            self.model_name = f"bdt_{self.config.model_type}"

        # Load binary
        self._clf = joblib.load(load_dir / "classifier.joblib")

        # Load metadata
        meta = self._load_metadata(load_dir)
        self._metadata.update(meta)
        self._is_fitted = True

        log.info("BDT loaded", path=str(load_dir), model_type=self.config.model_type)

    def summary(self) -> dict[str, Any]:
        base = super().summary()
        base["config"] = self.config.to_dict()
        if self._clf is not None and hasattr(self._clf, "best_iteration"):
            base["best_iteration"] = self._clf.best_iteration
        elif self._clf is not None and hasattr(self._clf, "best_iteration_"):
            base["best_iteration"] = self._clf.best_iteration_
        return base
