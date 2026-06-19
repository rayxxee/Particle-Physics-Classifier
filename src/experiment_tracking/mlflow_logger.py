"""
MLflow run logging wrapper for the Particle Physics Classifier.

Provides a clean context-manager-based API for logging experiments.
Every training run automatically logs:
- All hyperparameters (from model config)
- Per-epoch metrics (loss, AUC, LR)
- Summary metrics (best AUC, training time, n_parameters)
- Artifacts: ROC curve plot, training curve plot, saved model

The wrapper handles MLflow server unavailability gracefully (warning,
not error) so training is never blocked by tracking failures.

Usage:
    from src.experiment_tracking.mlflow_logger import MLflowLogger

    logger = MLflowLogger(experiment_name="particle_physics_classifier")
    with logger.start_run(run_name="mlp_baseline") as run:
        logger.log_params(config.to_dict())
        for epoch in range(100):
            logger.log_metrics({"val_auc": 0.81, "train_loss": 0.3}, step=epoch)
        logger.log_model(model, artifact_path="model")
"""

from __future__ import annotations

import contextlib
import warnings
from pathlib import Path
from typing import Any, Generator

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
from mlflow.entities import Run
from sklearn.metrics import roc_auc_score, roc_curve

from src.utils.logging_config import get_logger

matplotlib.use("Agg")  # Non-interactive backend for server environments

log = get_logger(__name__)


class MLflowLogger:
    """
    Wrapper around MLflow for experiment tracking.

    Args:
        experiment_name:  MLflow experiment name. Created if it doesn't exist.
        tracking_uri:     MLflow server URI. Defaults to local ./mlruns.
        artifact_location: Custom artifact store location. None = default.

    Example:
        logger = MLflowLogger("particle_physics_classifier")
        with logger.start_run("mlp_v1") as run:
            logger.log_params({"lr": 0.001, "epochs": 100})
            logger.log_metrics({"val_auc": 0.82}, step=50)
    """

    def __init__(
        self,
        experiment_name: str = "particle_physics_classifier",
        tracking_uri: str | None = None,
        artifact_location: str | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or mlflow.get_tracking_uri()
        self._active_run: Run | None = None
        self._mlflow_available: bool = True

        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(
                    experiment_name,
                    artifact_location=artifact_location,
                )
                log.info("Created MLflow experiment", name=experiment_name)
            mlflow.set_experiment(experiment_name)
            log.info(
                "MLflow configured",
                experiment=experiment_name,
                tracking_uri=self.tracking_uri,
            )
        except Exception as e:
            warnings.warn(
                f"MLflow unavailable: {e}\n"
                "Training will continue but experiments will not be tracked.",
                stacklevel=2,
            )
            self._mlflow_available = False

    @contextlib.contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        nested: bool = False,
    ) -> Generator[Run | None, None, None]:
        """
        Context manager that starts and ends an MLflow run.

        Args:
            run_name: Human-readable run name (auto-generated if None).
            tags:     Key-value tags attached to the run.
            nested:   Set True for Optuna child runs.

        Yields:
            mlflow.entities.Run, or None if MLflow is unavailable.

        Example:
            with logger.start_run("mlp_baseline", tags={"model": "mlp"}) as run:
                logger.log_params({"lr": 0.001})
        """
        if not self._mlflow_available:
            yield None
            return

        try:
            with mlflow.start_run(run_name=run_name, tags=tags, nested=nested) as run:
                self._active_run = run
                log.info(
                    "MLflow run started",
                    run_id=run.info.run_id[:8],
                    run_name=run_name,
                )
                yield run
        except Exception as e:
            log.warning("MLflow run error", error=str(e))
            yield None
        finally:
            self._active_run = None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a dict of hyperparameters. Values are stringified."""
        if not self._mlflow_available:
            return
        try:
            # MLflow param values must be < 500 chars
            safe_params = {k: str(v)[:499] for k, v in params.items()}
            mlflow.log_params(safe_params)
        except Exception as e:
            log.warning("Failed to log params", error=str(e))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log a dict of metrics at a given step (epoch number)."""
        if not self._mlflow_available:
            return
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            log.warning("Failed to log metrics", error=str(e))

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a single metric."""
        self.log_metrics({key: value}, step=step)

    def log_history(self, history: dict[str, list]) -> None:
        """
        Log per-epoch training history.

        Args:
            history: Dict of metric name → list of per-epoch values.
                     E.g. {"train_auc": [0.7, 0.75, 0.8], "val_auc": [...]}
        """
        if not self._mlflow_available:
            return
        for metric_name, values in history.items():
            if isinstance(values, list):
                for epoch, value in enumerate(values):
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        self.log_metric(metric_name, float(value), step=epoch + 1)

    def log_summary(self, metrics: dict[str, float]) -> None:
        """Log summary metrics (best AUC, training time, etc.)."""
        self.log_metrics(metrics)

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: str | None = None,
    ) -> None:
        """
        Log a PyTorch model artifact to MLflow.

        Args:
            model:                  Model object (BaseModel subclass or nn.Module).
            artifact_path:          Artifact subdirectory name.
            registered_model_name:  If set, registers in MLflow Model Registry.
        """
        if not self._mlflow_available:
            return
        try:
            import torch.nn as nn

            if hasattr(model, "_net") and model._net is not None:
                # BaseModel wrapper — log the underlying PyTorch net
                mlflow.pytorch.log_model(
                    pytorch_model=model._net,
                    artifact_path=artifact_path,
                    registered_model_name=registered_model_name,
                )
            elif isinstance(model, nn.Module):
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path=artifact_path,
                    registered_model_name=registered_model_name,
                )
            log.info("Model logged to MLflow", artifact_path=artifact_path)
        except Exception as e:
            log.warning("Failed to log model", error=str(e))

    def log_artifact_file(self, local_path: str | Path, artifact_dir: str = "") -> None:
        """Log a local file as an MLflow artifact."""
        if not self._mlflow_available:
            return
        try:
            mlflow.log_artifact(str(local_path), artifact_dir)
        except Exception as e:
            log.warning("Failed to log artifact", path=str(local_path), error=str(e))

    # ── Plot utilities ────────────────────────────────────────────────────────

    def log_training_curves(
        self,
        history: dict[str, list],
        artifact_dir: str = "plots",
    ) -> None:
        """
        Generate and log training/validation loss and AUC curves.

        Args:
            history:      Training history dict from trainer.train().
            artifact_dir: MLflow artifact subdirectory.
        """
        if not self._mlflow_available:
            return

        try:
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)

                # Loss curves
                if "train_loss" in history and "val_loss" in history:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    epochs = range(1, len(history["train_loss"]) + 1)
                    ax.plot(epochs, history["train_loss"], label="Train Loss", color="#3b82f6")
                    ax.plot(epochs, history["val_loss"], label="Val Loss", color="#ef4444")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Binary Cross-Entropy Loss")
                    ax.set_title("Training and Validation Loss")
                    ax.legend()
                    ax.grid(alpha=0.3)
                    fig.tight_layout()
                    loss_path = tmp_path / "training_loss.png"
                    fig.savefig(loss_path, dpi=150)
                    plt.close(fig)
                    self.log_artifact_file(loss_path, artifact_dir)

                # AUC curves
                if "train_auc" in history and "val_auc" in history:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    epochs = range(1, len(history["train_auc"]) + 1)
                    ax.plot(epochs, history["train_auc"], label="Train AUC", color="#3b82f6")
                    ax.plot(epochs, history["val_auc"], label="Val AUC", color="#10b981")
                    best_epoch = int(np.argmax(history["val_auc"])) + 1
                    best_auc = max(history["val_auc"])
                    ax.axvline(x=best_epoch, color="#f59e0b", linestyle="--", alpha=0.7,
                               label=f"Best epoch {best_epoch} (AUC={best_auc:.4f})")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("AUC-ROC")
                    ax.set_title("Training and Validation AUC")
                    ax.set_ylim(0.5, 1.0)
                    ax.legend()
                    ax.grid(alpha=0.3)
                    fig.tight_layout()
                    auc_path = tmp_path / "training_auc.png"
                    fig.savefig(auc_path, dpi=150)
                    plt.close(fig)
                    self.log_artifact_file(auc_path, artifact_dir)

                log.info("Training curves logged", artifact_dir=artifact_dir)
        except Exception as e:
            log.warning("Failed to log training curves", error=str(e))

    def log_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        model_name: str = "model",
        artifact_dir: str = "plots",
    ) -> None:
        """
        Compute and log the ROC curve plot and AUC metric.

        Args:
            y_true:       True binary labels.
            y_scores:     Predicted scores P(signal).
            model_name:   Used in plot title.
            artifact_dir: MLflow artifact subdirectory.
        """
        if not self._mlflow_available:
            return

        try:
            import tempfile
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            auc = roc_auc_score(y_true, y_scores)

            # Log AUC as metric
            self.log_metric(f"test_auc_{model_name}", float(auc))

            with tempfile.TemporaryDirectory() as tmp:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                # ─ Standard ROC ─
                axes[0].plot(fpr, tpr, color="#3b82f6", lw=2, label=f"AUC = {auc:.4f}")
                axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
                axes[0].set_xlabel("False Positive Rate (FPR)")
                axes[0].set_ylabel("True Positive Rate (TPR = Signal Efficiency)")
                axes[0].set_title(f"ROC Curve — {model_name}")
                axes[0].legend()
                axes[0].grid(alpha=0.3)

                # ─ HEP-style: Background Rejection vs Signal Efficiency ─
                # Standard HEP plot: 1/FPR (background rejection) vs TPR (signal efficiency)
                # on log-log scale
                fpr_safe = np.maximum(fpr, 1e-6)
                axes[1].semilogy(tpr, 1.0 / fpr_safe, color="#10b981", lw=2)
                axes[1].set_xlabel("Signal Efficiency (TPR)")
                axes[1].set_ylabel("Background Rejection (1/FPR)")
                axes[1].set_title(f"Background Rejection vs Signal Efficiency — {model_name}")
                axes[1].grid(alpha=0.3, which="both")
                axes[1].set_xlim(0.0, 1.0)

                fig.tight_layout()
                roc_path = Path(tmp) / f"roc_curve_{model_name}.png"
                fig.savefig(roc_path, dpi=150)
                plt.close(fig)
                self.log_artifact_file(roc_path, artifact_dir)

            log.info("ROC curve logged", model=model_name, auc=f"{auc:.4f}")
        except Exception as e:
            log.warning("Failed to log ROC curve", error=str(e))

    # ── Convenience: log full training run ────────────────────────────────────

    def log_full_run(
        self,
        params: dict[str, Any],
        history: dict[str, list],
        y_val: np.ndarray,
        val_scores: np.ndarray,
        model: Any,
        extra_metrics: dict[str, float] | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        """
        Convenience method: log all artifacts for a completed training run.

        Logs: params, epoch metrics, training curves, ROC curve, model artifact.
        Intended to be called after training is complete.

        Args:
            params:        Model hyperparameters dict.
            history:       Training history from trainer.train().
            y_val:         Validation labels (for ROC curve).
            val_scores:    Validation P(signal) scores (for ROC curve).
            model:         Trained model (for artifact logging).
            extra_metrics: Additional scalar metrics to log.
            tags:          Run tags.
        """
        self.log_params(params)
        self.log_history(history)

        summary: dict[str, float] = {
            "best_val_auc": float(history.get("best_val_auc", 0)),
            "best_epoch": float(history.get("best_epoch", 0)),
            "fit_time_s": float(history.get("fit_time_s", 0)),
            "n_epochs_run": float(history.get("n_epochs_run", 0)),
        }
        if extra_metrics:
            summary.update(extra_metrics)
        self.log_summary(summary)

        self.log_training_curves(history)
        self.log_roc_curve(y_val, val_scores, model_name=getattr(model, "model_name", "model"))
        self.log_model(model)

        log.info("Full run logged to MLflow")
