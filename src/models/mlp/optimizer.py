"""
Module 3b — Hyperparameter optimization for the Deep MLP.

Uses Optuna (TPE sampler) to search the joint space of:
  - Architecture: n_layers, hidden widths, dropout rates, batch norm
  - Optimisation: learning_rate, weight_decay, batch_size
  - Scheduler: type, cosine T_max / plateau patience+factor

Each trial trains a full MLP and reports val_auc to Optuna.
Optuna prunes bad trials early via MedianPruner (integrated with the
MLPTrainer's per-epoch logging callback).

Usage (CLI):
    python -m src.models.mlp.optimizer \\
        --n-trials 50 \\
        --n-epochs-per-trial 30 \\
        --study-name mlp_hpo \\
        --storage sqlite:///optuna.db

Usage (Python):
    from src.models.mlp.optimizer import MLPOptimizer, HPOConfig

    hpo = MLPOptimizer(HPOConfig(n_trials=50, n_epochs_per_trial=30))
    best_cfg = hpo.run(X_train, y_train, X_val, y_val)
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# MLPConfig and MLPModel are imported lazily inside functions to avoid
# triggering a torch/CUDA init at module import time (which hangs pytest
# collection when torch initialises the CUDA runtime).
from src.utils.logging_config import get_logger

log = get_logger(__name__)

# Silence Optuna's verbose logging — we use our own structured logger.
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─── HPO configuration ────────────────────────────────────────────────────────

@dataclass
class HPOConfig:
    """
    Configuration for the Optuna hyperparameter search.

    Attributes:
        n_trials:            Total Optuna trials to run.
        n_epochs_per_trial:  Max epochs per trial (kept low to save compute).
        early_stopping_patience: Per-trial early stopping patience.
        study_name:          Optuna study name (used for persistence).
        storage:             SQLite URL for persistent study, e.g.
                             "sqlite:///optuna.db". None = in-memory.
        direction:           Optimisation direction ("maximize" for AUC).
        n_startup_trials:    TPE random warmup before Bayesian sampling.
        n_warmup_steps:      Pruner: skip first N epochs before pruning.
        pruning_interval:    Pruner: report every N epochs.
        seed:                Random seed for TPE sampler.
        n_layers_range:      (min, max) number of hidden layers.
        hidden_dim_choices:  Layer width candidates (powers of 2).
        lr_range:            (log_low, log_high) for log-uniform lr search.
        wd_range:            (log_low, log_high) for log-uniform weight_decay.
        batch_size_choices:  Batch size candidates.
        dropout_range:       (min, max) uniform dropout range.
        save_best_config:    Path to save best config as YAML. None = skip.
        mlflow_experiment:   MLflow experiment for trial logging. None = skip.
    """
    n_trials: int = 50
    n_epochs_per_trial: int = 30
    early_stopping_patience: int = 7
    study_name: str = "mlp_hpo"
    storage: str | None = None
    direction: str = "maximize"
    n_startup_trials: int = 10
    n_warmup_steps: int = 5
    pruning_interval: int = 1
    seed: int = 42
    n_layers_range: tuple[int, int] = (2, 5)
    hidden_dim_choices: list[int] = field(
        default_factory=lambda: [64, 128, 256, 512, 1024]
    )
    lr_range: tuple[float, float] = (1e-4, 1e-2)
    wd_range: tuple[float, float] = (1e-6, 1e-2)
    batch_size_choices: list[int] = field(
        default_factory=lambda: [512, 1024, 2048, 4096]
    )
    dropout_range: tuple[float, float] = (0.0, 0.5)
    save_best_config: str | None = None
    mlflow_experiment: str | None = "mlp_hpo"


# ─── Optuna objective ─────────────────────────────────────────────────────────

class _MLPObjective:
    """
    Callable Optuna objective.

    Suggests hyperparameters for one trial, trains an MLP, and returns
    the best val_auc achieved during that trial. Integrates with Optuna
    pruning via intermediate_values.
    """

    def __init__(
        self,
        hpo_cfg: HPOConfig,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        input_dim: int,
    ) -> None:
        self.hpo_cfg = hpo_cfg
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.input_dim = input_dim

    def __call__(self, trial: optuna.Trial) -> float:
        cfg = self._suggest_config(trial)

        # Import here (lazy) to avoid torch init at collection time
        from src.models.mlp.model import MLPModel

        # Build and train model — pass Optuna trial for pruning callbacks
        model = MLPModel(cfg)
        try:
            val_auc = model.fit(
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                optuna_trial=trial,
            )
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            log.warning("Trial failed", trial=trial.number, error=str(exc))
            return 0.0

        return float(val_auc)

    def _suggest_config(self, trial: optuna.Trial):
        """Map Optuna suggestions → MLPConfig."""
        from src.models.mlp.config import MLPConfig
        cfg = self.hpo_cfg

        # Architecture
        n_layers = trial.suggest_int("n_layers", *cfg.n_layers_range)
        hidden_dims = [
            trial.suggest_categorical(f"hidden_dim_l{i}", cfg.hidden_dim_choices)
            for i in range(n_layers)
        ]
        dropout_rates = [
            round(trial.suggest_float(f"dropout_l{i}", *cfg.dropout_range), 2)
            for i in range(n_layers)
        ]
        batch_norm = trial.suggest_categorical("batch_norm", [True, False])

        # Optimisation
        lr = trial.suggest_float("lr", *cfg.lr_range, log=True)
        wd = trial.suggest_float("weight_decay", *cfg.wd_range, log=True)
        batch_size = trial.suggest_categorical("batch_size", cfg.batch_size_choices)

        # Scheduler
        scheduler = trial.suggest_categorical(
            "scheduler", ["cosine_annealing", "reduce_on_plateau", "none"]
        )

        return MLPConfig(
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            dropout_rates=dropout_rates,
            batch_norm=batch_norm,
            epochs=cfg.n_epochs_per_trial,
            batch_size=batch_size,
            learning_rate=lr,
            weight_decay=wd,
            scheduler_name=scheduler,
            scheduler_T_max=cfg.n_epochs_per_trial,
            early_stopping=True,
            early_stopping_patience=cfg.early_stopping_patience,
            early_stopping_monitor="val_auc",
            mixed_precision=False,   # disable AMP during HPO for reproducibility
            seed=cfg.seed + trial.number,
        )


# ─── Main optimizer class ─────────────────────────────────────────────────────

class MLPOptimizer:
    """
    Orchestrates Optuna HPO for the DeepMLP.

    Example::

        from src.models.mlp.optimizer import MLPOptimizer, HPOConfig

        hpo = MLPOptimizer(HPOConfig(n_trials=50))
        best_cfg = hpo.run(X_train, y_train, X_val, y_val)
        print(best_cfg)
    """

    def __init__(self, config: HPOConfig | None = None) -> None:
        self.config = config or HPOConfig()
        self._study: optuna.Study | None = None
        self._best_config: MLPConfig | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> MLPConfig:
        """
        Run the full HPO sweep and return the best MLPConfig found.

        Args:
            X_train: Training features, shape (n_train, n_features).
            y_train: Training labels, shape (n_train,).
            X_val:   Validation features.
            y_val:   Validation labels.

        Returns:
            MLPConfig with the best hyperparameters.
        """
        input_dim = X_train.shape[1]
        cfg = self.config

        log.info(
            "HPO starting",
            study=cfg.study_name,
            n_trials=cfg.n_trials,
            n_epochs_per_trial=cfg.n_epochs_per_trial,
            n_train=len(X_train),
            n_val=len(X_val),
        )

        self._study = self._create_study()

        objective = _MLPObjective(
            hpo_cfg=cfg,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            input_dim=input_dim,
        )

        self._study.optimize(
            objective,
            n_trials=cfg.n_trials,
            callbacks=[self._log_trial_callback],
            catch=(Exception,),
        )

        best_trial = self._study.best_trial
        log.info(
            "HPO complete",
            best_trial=best_trial.number,
            best_val_auc=best_trial.value,
            best_params=best_trial.params,
        )

        self._best_config = self._trial_to_config(best_trial, input_dim)

        if cfg.save_best_config:
            self._save_best_config(cfg.save_best_config)

        return self._best_config

    @property
    def study(self) -> optuna.Study | None:
        """Access the underlying Optuna study after run()."""
        return self._study

    @property
    def best_config(self) -> MLPConfig | None:
        """Best MLPConfig after run()."""
        return self._best_config

    def importance(self) -> dict[str, float]:
        """
        Return hyperparameter importance scores (requires completed study).

        Returns:
            Dict mapping param name → importance score (0-1).
        """
        if self._study is None:
            raise RuntimeError("Call run() first.")
        importances = optuna.importance.get_param_importances(self._study)
        return dict(importances)

    def get_trials_dataframe(self):
        """Return all trials as a pandas DataFrame for analysis."""
        if self._study is None:
            raise RuntimeError("Call run() first.")
        return self._study.trials_dataframe()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _create_study(self) -> optuna.Study:
        """Create (or load from storage) an Optuna study."""
        cfg = self.config
        sampler = TPESampler(
            n_startup_trials=cfg.n_startup_trials,
            seed=cfg.seed,
        )
        pruner = MedianPruner(
            n_startup_trials=cfg.n_startup_trials,
            n_warmup_steps=cfg.n_warmup_steps,
            interval_steps=cfg.pruning_interval,
        )
        return optuna.create_study(
            study_name=cfg.study_name,
            storage=cfg.storage,
            direction=cfg.direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

    def _trial_to_config(self, trial: optuna.Trial, input_dim: int):
        """Reconstruct the MLPConfig from an Optuna Trial's param dict."""
        from src.models.mlp.config import MLPConfig
        p = trial.params
        cfg = self.config

        n_layers = p["n_layers"]
        hidden_dims = [p[f"hidden_dim_l{i}"] for i in range(n_layers)]
        dropout_rates = [p[f"dropout_l{i}"] for i in range(n_layers)]

        return MLPConfig(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rates=dropout_rates,
            batch_norm=p.get("batch_norm", True),
            epochs=100,                        # full epochs for final retraining
            batch_size=p["batch_size"],
            learning_rate=p["lr"],
            weight_decay=p["weight_decay"],
            scheduler_name=p.get("scheduler", "cosine_annealing"),
            scheduler_T_max=100,
            early_stopping=True,
            early_stopping_patience=10,
            early_stopping_monitor="val_auc",
            mixed_precision=True,
            seed=cfg.seed,
        )

    def _save_best_config(self, path: str) -> None:
        """Save best config to YAML for reproducibility."""
        import yaml
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if self._best_config is None:
            return

        cfg_dict = {
            "architecture": {
                "input_dim": self._best_config.input_dim,
                "hidden_dims": self._best_config.hidden_dims,
                "batch_norm": self._best_config.batch_norm,
                "dropout_rates": self._best_config.dropout_rates,
            },
            "training": {
                "epochs": self._best_config.epochs,
                "batch_size": self._best_config.batch_size,
                "learning_rate": self._best_config.learning_rate,
                "weight_decay": self._best_config.weight_decay,
                "mixed_precision": self._best_config.mixed_precision,
                "class_weights": self._best_config.class_weights,
                "seed": self._best_config.seed,
                "scheduler": {
                    "name": self._best_config.scheduler_name,
                    "T_max": self._best_config.scheduler_T_max,
                },
                "early_stopping": {
                    "enabled": self._best_config.early_stopping,
                    "patience": self._best_config.early_stopping_patience,
                    "monitor": self._best_config.early_stopping_monitor,
                },
            },
        }
        with open(save_path, "w") as f:
            yaml.safe_dump(cfg_dict, f, default_flow_style=False)

        log.info("Best config saved", path=str(save_path))

    @staticmethod
    def _log_trial_callback(study: optuna.Study, trial: optuna.FrozenTrial) -> None:
        """Structured log after each completed trial."""
        if trial.state == optuna.trial.TrialState.COMPLETE:
            log.info(
                "Trial complete",
                trial=trial.number,
                val_auc=round(trial.value or 0.0, 4),
                best_so_far=round(study.best_value, 4),
            )
        elif trial.state == optuna.trial.TrialState.PRUNED:
            log.debug("Trial pruned", trial=trial.number)
        elif trial.state == optuna.trial.TrialState.FAIL:
            log.warning("Trial failed", trial=trial.number)


# ─── CLI entry point ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLP Hyperparameter Optimization")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-epochs-per-trial", type=int, default=30)
    parser.add_argument("--study-name", type=str, default="mlp_hpo")
    parser.add_argument("--storage", type=str, default=None,
                        help="SQLite URL e.g. sqlite:///optuna.db")
    parser.add_argument("--data-version", type=str, default=None,
                        help="ETL version hash to load from feature store")
    parser.add_argument("--feature-store-dir", type=str,
                        default="data/processed/feature_store")
    parser.add_argument("--save-best-config", type=str,
                        default="configs/mlp_best.yaml")
    parser.add_argument("--mlflow-experiment", type=str, default="mlp_hpo")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    from src.features.feature_store import FeatureConfig, FeatureStore
    from src.ingestion.etl_pipeline import ETLConfig, ETLPipeline

    log.info("Loading data for HPO")

    # Load features from store (or re-run ETL if needed)
    store = FeatureStore(
        FeatureConfig(include_derived_hl=True),
        cache_dir=args.feature_store_dir,
    )

    X_train, y_train = store.load("train")
    X_val, y_val = store.load("val")

    hpo_cfg = HPOConfig(
        n_trials=args.n_trials,
        n_epochs_per_trial=args.n_epochs_per_trial,
        study_name=args.study_name,
        storage=args.storage,
        save_best_config=args.save_best_config,
        mlflow_experiment=args.mlflow_experiment,
        seed=args.seed,
    )

    optimizer = MLPOptimizer(hpo_cfg)
    best_cfg = optimizer.run(
        X_train.values, y_train.values,
        X_val.values, y_val.values,
    )

    print("\n=== Best Config ===")
    for k, v in best_cfg.to_dict().items():
        print(f"  {k}: {v}")

    imp = optimizer.importance()
    print("\n=== Hyperparameter Importance ===")
    for k, v in sorted(imp.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
