"""
BDT (Boosted Decision Tree) model configuration.

Supports both XGBoost and LightGBM backends via the same dataclass.
All hyperparameters for both frameworks are defined here.

Usage:
    from src.models.bdt.config import BDTConfig

    # XGBoost default
    config = BDTConfig(model_type="xgboost")

    # LightGBM default
    config = BDTConfig(model_type="lightgbm")

    # From YAML
    config = BDTConfig.from_yaml("configs/bdt_default.yaml")
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class BDTConfig:
    """
    Hyperparameter configuration for BDT (XGBoost or LightGBM).

    Shared parameters are identical for both frameworks where possible.
    Framework-specific parameters are prefixed with 'xgb_' or 'lgbm_'.
    """

    # ── Model selection ───────────────────────────────────────────────────────
    model_type: Literal["xgboost", "lightgbm"] = "xgboost"
    """Backend: 'xgboost' or 'lightgbm'."""

    # ── Shared hyperparameters ────────────────────────────────────────────────
    n_estimators: int = 1000
    """Number of boosting rounds (trees)."""

    learning_rate: float = 0.05
    """Step size shrinkage (eta in XGBoost)."""

    max_depth: int = 6
    """Maximum tree depth. Deeper trees = more complex, more prone to overfitting."""

    min_child_weight: float = 1.0
    """Minimum sum of instance weight in a child leaf (XGBoost) / min_child_samples (LightGBM)."""

    subsample: float = 0.8
    """Row subsampling fraction per tree."""

    colsample_bytree: float = 0.8
    """Column subsampling fraction per tree."""

    reg_alpha: float = 0.0
    """L1 regularization term."""

    reg_lambda: float = 1.0
    """L2 regularization term."""

    # ── XGBoost-specific ──────────────────────────────────────────────────────
    xgb_tree_method: str = "hist"
    """XGBoost tree construction method: 'hist' (fast) or 'exact'."""

    xgb_eval_metric: str = "auc"
    """XGBoost evaluation metric for early stopping."""

    # ── LightGBM-specific ─────────────────────────────────────────────────────
    lgbm_num_leaves: int = 127
    """LightGBM: max number of leaves per tree."""

    lgbm_min_data_in_leaf: int = 20
    """LightGBM: minimum data points per leaf."""

    lgbm_feature_fraction: float = 0.8
    """LightGBM: fraction of features per tree (alias for colsample_bytree)."""

    # ── Early stopping ────────────────────────────────────────────────────────
    early_stopping_rounds: int = 50
    """Stop if val AUC doesn't improve for this many rounds."""

    # ── Training ──────────────────────────────────────────────────────────────
    seed: int = 42
    """Random seed for reproducibility."""

    n_jobs: int = -1
    """Number of CPU threads. -1 = all cores."""

    verbose: int = 0
    """Verbosity: 0 = silent, 1 = progress."""

    # ── Persistence ───────────────────────────────────────────────────────────
    checkpoint_dir: str = "models/bdt"
    """Directory for model checkpoints."""

    def to_dict(self) -> dict:
        """Return config as a flat dict for MLflow param logging."""
        return {
            "model_type": self.model_type,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "early_stopping_rounds": self.early_stopping_rounds,
            "seed": self.seed,
            "n_jobs": self.n_jobs,
        }

    def config_hash(self) -> str:
        """SHA-256 hash of config for cache keying."""
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode()
        ).hexdigest()[:12]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BDTConfig":
        """Load BDTConfig from a YAML file."""
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})

    def xgboost_params(self, device: str = "cpu") -> dict:
        """
        Build the kwargs dict for xgboost.XGBClassifier.

        Args:
            device: 'cpu' or 'cuda' (for GPU acceleration).
        """
        params = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "tree_method": self.xgb_tree_method,
            "eval_metric": self.xgb_eval_metric,
            "random_state": self.seed,
            "n_jobs": self.n_jobs,
            "verbosity": self.verbose,
            # early_stopping_rounds moved to constructor in XGBoost >=2.0
            "early_stopping_rounds": self.early_stopping_rounds,
            "objective": "binary:logistic",
        }
        if device == "cuda":
            params["device"] = "cuda"
            params["tree_method"] = "hist"  # required for CUDA
        return params

    def lightgbm_params(self) -> dict:
        """Build the kwargs dict for lightgbm.LGBMClassifier."""
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.lgbm_num_leaves,
            "min_child_weight": self.min_child_weight,
            "min_child_samples": int(self.lgbm_min_data_in_leaf),
            "subsample": self.subsample,
            "colsample_bytree": self.lgbm_feature_fraction,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.seed,
            "n_jobs": self.n_jobs,
            "verbose": -1,
            "objective": "binary",
        }
