"""
Evaluation pipeline for particle physics classifier models.

Produces a comprehensive suite of HEP + ML evaluation metrics and plots:

  ML Metrics:
    - ROC curve + AUC
    - Precision-Recall curve + Average Precision
    - Threshold sweep: accuracy, F1, precision, recall at N thresholds
    - Calibration curve (reliability diagram)

  HEP-specific:
    - Punzi Figure of Merit: FOM(t) = S(t) / (a/2 + sqrt(B(t)))
      where S = true positives, B = false positives, a = significance target

  Output:
    - PNG plots saved locally and logged to MLflow as artifacts
    - EvaluationResult dataclass with all scalar metrics as fields

Usage:
    from src.evaluation.evaluation_pipeline import EvaluationPipeline, EvaluationConfig

    cfg = EvaluationConfig(output_dir="eval_output", significance_target=5.0)
    pipeline = EvaluationPipeline(cfg)
    result = pipeline.evaluate(model, X_test, y_test, run_name="mlp_v1")
    print(f"AUC: {result.roc_auc:.4f}, Best Punzi FOM: {result.best_punzi_fom:.4f}")
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend

from src.utils.logging_config import get_logger

log = get_logger(__name__)


# ─── Config dataclass ─────────────────────────────────────────────────────────

@dataclass
class EvaluationConfig:
    """
    Configuration for the evaluation pipeline.

    Args:
        output_dir:          Directory to write PNG plots.
        significance_target: Significance level 'a' in Punzi FOM (default 5σ).
        n_threshold_steps:   Number of thresholds to evaluate in [0, 1].
        n_calibration_bins:  Number of bins for calibration curve.
        dpi:                 Plot DPI for saved figures.
        log_to_mlflow:       Whether to log plots to MLflow.
        mlflow_artifact_dir: MLflow artifact subdirectory for plots.
    """

    output_dir: str = "eval_output"
    significance_target: float = 5.0
    n_threshold_steps: int = 200
    n_calibration_bins: int = 10
    dpi: int = 150
    log_to_mlflow: bool = True
    mlflow_artifact_dir: str = "evaluation"

    def to_dict(self) -> dict:
        """Return config as flat dict for MLflow logging."""
        return {
            "eval_output_dir": self.output_dir,
            "eval_significance_target": self.significance_target,
            "eval_n_threshold_steps": self.n_threshold_steps,
            "eval_n_calibration_bins": self.n_calibration_bins,
        }

    def config_hash(self) -> str:
        """SHA-256 hash for caching keyed on config."""
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:8]


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """
    All evaluation metrics for a single model evaluation.

    Fields:
        roc_auc:           Area under the ROC curve.
        average_precision: Area under the Precision-Recall curve.
        best_f1:           Best F1 score over all thresholds.
        best_accuracy:     Best accuracy over all thresholds.
        best_threshold:    Threshold that maximises F1.
        best_punzi_fom:    Peak Punzi Figure of Merit.
        best_punzi_threshold: Threshold at peak Punzi FOM.
        calibration_ece:   Expected Calibration Error.
        n_signal:          Number of signal events in the test set.
        n_background:      Number of background events in the test set.
        model_name:        Name of the model evaluated.
        run_name:          Name of the evaluation run.
        plot_paths:        Paths to saved plot PNG files.
    """

    roc_auc: float = 0.0
    average_precision: float = 0.0
    best_f1: float = 0.0
    best_accuracy: float = 0.0
    best_threshold: float = 0.5
    best_punzi_fom: float = 0.0
    best_punzi_threshold: float = 0.5
    calibration_ece: float = 0.0
    n_signal: int = 0
    n_background: int = 0
    model_name: str = "unknown"
    run_name: str = "unknown"
    plot_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return all scalar metrics as a flat dict (for MLflow logging)."""
        return {
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
            "best_f1": self.best_f1,
            "best_accuracy": self.best_accuracy,
            "best_threshold": self.best_threshold,
            "best_punzi_fom": self.best_punzi_fom,
            "best_punzi_threshold": self.best_punzi_threshold,
            "calibration_ece": self.calibration_ece,
            "n_signal": float(self.n_signal),
            "n_background": float(self.n_background),
        }

    def summary_str(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Model:              {self.model_name}",
            f"Run:                {self.run_name}",
            f"ROC AUC:            {self.roc_auc:.4f}",
            f"Average Precision:  {self.average_precision:.4f}",
            f"Best F1:            {self.best_f1:.4f} (threshold={self.best_threshold:.3f})",
            f"Best Accuracy:      {self.best_accuracy:.4f}",
            f"Punzi FOM:          {self.best_punzi_fom:.4f} (threshold={self.best_punzi_threshold:.3f})",
            f"Calibration ECE:    {self.calibration_ece:.4f}",
            f"Signal events:      {self.n_signal:,}",
            f"Background events:  {self.n_background:,}",
        ]
        return "\n".join(lines)


# ─── Main pipeline ─────────────────────────────────────────────────────────────

class EvaluationPipeline:
    """
    Comprehensive evaluation pipeline for particle physics classifiers.

    Computes ROC, PR, threshold sweep, calibration, and Punzi FOM.
    Saves PNG plots and logs everything to MLflow.

    Args:
        config: EvaluationConfig. Uses defaults if None.

    Example:
        from src.evaluation.evaluation_pipeline import EvaluationPipeline, EvaluationConfig

        pipeline = EvaluationPipeline(EvaluationConfig())
        result = pipeline.evaluate(model, X_test, y_test, run_name="mlp_v1")
        print(result.summary_str())
    """

    def __init__(self, config: EvaluationConfig | None = None) -> None:
        self.config = config or EvaluationConfig()
        self._output_dir = Path(self.config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        model: Any,
        X_test: Any,
        y_test: Any,
        run_name: str = "evaluation",
        mlflow_logger: Any | None = None,
    ) -> EvaluationResult:
        """
        Run the full evaluation suite.

        Args:
            model:         A fitted model with .predict_proba(X) → np.ndarray.
            X_test:        Test features, shape (n_events, n_features).
            y_test:        Test labels, shape (n_events,), values {0, 1}.
            run_name:      Identifier for this evaluation (used in plot filenames).
            mlflow_logger: Optional MLflowLogger for artifact logging.

        Returns:
            EvaluationResult with all metrics and plot paths.
        """
        import numpy as np

        # Convert labels to numpy
        y_np = np.asarray(y_test, dtype=np.float32)
        model_name = getattr(model, "model_name", "model")

        log.info(
            "Evaluation started",
            model=model_name,
            run=run_name,
            n_events=len(y_np),
        )

        # Get predictions
        y_scores = model.predict_proba(X_test)
        y_scores = np.asarray(y_scores, dtype=np.float32)

        result = EvaluationResult(
            model_name=model_name,
            run_name=run_name,
            n_signal=int((y_np == 1).sum()),
            n_background=int((y_np == 0).sum()),
        )

        plot_paths: list[str] = []

        # ── 1. ROC curve ──────────────────────────────────────────────────────
        roc_path = self._plot_roc(y_np, y_scores, run_name)
        plot_paths.append(str(roc_path))
        result.roc_auc = self._compute_roc_auc(y_np, y_scores)

        # ── 2. Precision-Recall curve ─────────────────────────────────────────
        pr_path = self._plot_pr(y_np, y_scores, run_name)
        plot_paths.append(str(pr_path))
        result.average_precision = self._compute_ap(y_np, y_scores)

        # ── 3. Threshold sweep ────────────────────────────────────────────────
        thresh_path, best_f1, best_acc, best_thresh = self._plot_threshold_sweep(
            y_np, y_scores, run_name
        )
        plot_paths.append(str(thresh_path))
        result.best_f1 = best_f1
        result.best_accuracy = best_acc
        result.best_threshold = best_thresh

        # ── 4. Calibration curve ──────────────────────────────────────────────
        cal_path, ece = self._plot_calibration(y_np, y_scores, run_name)
        plot_paths.append(str(cal_path))
        result.calibration_ece = ece

        # ── 5. Punzi FOM ──────────────────────────────────────────────────────
        punzi_path, best_fom, best_punzi_thresh = self._plot_punzi_fom(
            y_np, y_scores, run_name
        )
        plot_paths.append(str(punzi_path))
        result.best_punzi_fom = best_fom
        result.best_punzi_threshold = best_punzi_thresh

        result.plot_paths = plot_paths

        log.info(
            "Evaluation complete",
            roc_auc=f"{result.roc_auc:.4f}",
            ap=f"{result.average_precision:.4f}",
            best_f1=f"{result.best_f1:.4f}",
            punzi_fom=f"{result.best_punzi_fom:.4f}",
        )

        # ── Log to MLflow ─────────────────────────────────────────────────────
        if mlflow_logger is not None and self.config.log_to_mlflow:
            self._log_to_mlflow(mlflow_logger, result)

        return result

    # ── Plot methods ──────────────────────────────────────────────────────────

    def _plot_roc(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        run_name: str,
    ) -> Path:
        """Plot ROC curve + HEP-style background rejection curve."""
        from sklearn.metrics import roc_auc_score, roc_curve

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"ROC Curve — {run_name}  (AUC = {auc:.4f})", fontsize=13, fontweight="bold")

        # Standard ROC
        axes[0].plot(fpr, tpr, color="#3b82f6", lw=2, label=f"AUC = {auc:.4f}")
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
        axes[0].fill_between(fpr, tpr, alpha=0.08, color="#3b82f6")
        axes[0].set_xlabel("False Positive Rate (FPR)", fontsize=11)
        axes[0].set_ylabel("True Positive Rate (TPR)", fontsize=11)
        axes[0].set_title("Standard ROC")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # HEP-style: background rejection vs signal efficiency (log scale)
        fpr_safe = np.maximum(fpr, 1e-7)
        axes[1].semilogy(tpr, 1.0 / fpr_safe, color="#10b981", lw=2)
        axes[1].set_xlabel("Signal Efficiency (TPR)", fontsize=11)
        axes[1].set_ylabel("Background Rejection (1/FPR)", fontsize=11)
        axes[1].set_title("HEP-style: Background Rejection")
        axes[1].grid(alpha=0.3, which="both")
        axes[1].set_xlim(0.0, 1.0)

        fig.tight_layout()
        out_path = self._output_dir / f"roc_{run_name}.png"
        fig.savefig(out_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def _plot_pr(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        run_name: str,
    ) -> Path:
        """Plot Precision-Recall curve."""
        from sklearn.metrics import average_precision_score, precision_recall_curve

        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        baseline = y_true.mean()

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color="#8b5cf6", lw=2, label=f"AP = {ap:.4f}")
        ax.axhline(y=baseline, color="#ef4444", linestyle="--", alpha=0.6,
                   label=f"Random baseline ({baseline:.3f})")
        ax.fill_between(recall, precision, alpha=0.08, color="#8b5cf6")
        ax.set_xlabel("Recall", fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_title(f"Precision-Recall Curve — {run_name}", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        fig.tight_layout()

        out_path = self._output_dir / f"pr_curve_{run_name}.png"
        fig.savefig(out_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def _plot_threshold_sweep(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        run_name: str,
    ) -> tuple[Path, float, float, float]:
        """
        Sweep thresholds and plot accuracy, F1, precision, recall.

        Returns:
            (path, best_f1, best_accuracy, best_threshold_for_f1)
        """
        from sklearn.metrics import f1_score

        thresholds = np.linspace(0.0, 1.0, self.config.n_threshold_steps)
        accuracies, f1s, precisions, recalls = [], [], [], []

        for t in thresholds:
            preds = (y_scores >= t).astype(int)
            tp = int(((preds == 1) & (y_true == 1)).sum())
            fp = int(((preds == 1) & (y_true == 0)).sum())
            tn = int(((preds == 0) & (y_true == 0)).sum())
            fn = int(((preds == 0) & (y_true == 1)).sum())

            acc = (tp + tn) / max(len(y_true), 1)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1_val = 2 * prec * rec / max(prec + rec, 1e-9)

            accuracies.append(acc)
            f1s.append(f1_val)
            precisions.append(prec)
            recalls.append(rec)

        best_f1_idx = int(np.argmax(f1s))
        best_f1 = float(f1s[best_f1_idx])
        best_acc = float(max(accuracies))
        best_thresh = float(thresholds[best_f1_idx])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Threshold Sweep — {run_name}", fontsize=13, fontweight="bold")

        # Accuracy + F1
        axes[0].plot(thresholds, accuracies, label="Accuracy", color="#3b82f6", lw=2)
        axes[0].plot(thresholds, f1s, label="F1 Score", color="#10b981", lw=2)
        axes[0].axvline(x=best_thresh, color="#f59e0b", linestyle="--", alpha=0.8,
                        label=f"Best F1 threshold ({best_thresh:.2f})")
        axes[0].set_xlabel("Decision Threshold", fontsize=11)
        axes[0].set_ylabel("Score", fontsize=11)
        axes[0].set_title("Accuracy & F1 vs Threshold")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[0].set_xlim(0, 1)

        # Precision + Recall
        axes[1].plot(thresholds, precisions, label="Precision", color="#ef4444", lw=2)
        axes[1].plot(thresholds, recalls, label="Recall", color="#8b5cf6", lw=2)
        axes[1].axvline(x=best_thresh, color="#f59e0b", linestyle="--", alpha=0.8,
                        label=f"Best F1 threshold ({best_thresh:.2f})")
        axes[1].set_xlabel("Decision Threshold", fontsize=11)
        axes[1].set_ylabel("Score", fontsize=11)
        axes[1].set_title("Precision & Recall vs Threshold")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].set_xlim(0, 1)

        fig.tight_layout()
        out_path = self._output_dir / f"threshold_sweep_{run_name}.png"
        fig.savefig(out_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path, best_f1, best_acc, best_thresh

    def _plot_calibration(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        run_name: str,
    ) -> tuple[Path, float]:
        """
        Plot calibration curve (reliability diagram).

        Returns:
            (path, ece) — path to plot, Expected Calibration Error
        """
        from sklearn.calibration import calibration_curve

        n_bins = self.config.n_calibration_bins

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frac_pos, mean_pred = calibration_curve(
                y_true, y_scores, n_bins=n_bins, strategy="uniform"
            )

        # Compute ECE (Expected Calibration Error)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(y_scores)
        for i in range(n_bins):
            in_bin = (y_scores >= bin_edges[i]) & (y_scores < bin_edges[i + 1])
            n_in_bin = in_bin.sum()
            if n_in_bin > 0:
                acc_bin = float(y_true[in_bin].mean())
                conf_bin = float(y_scores[in_bin].mean())
                ece += (n_in_bin / n) * abs(acc_bin - conf_bin)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect calibration")
        ax.plot(mean_pred, frac_pos, "o-", color="#3b82f6", lw=2, ms=6,
                label=f"Model (ECE={ece:.4f})")
        ax.fill_between(mean_pred, frac_pos, mean_pred, alpha=0.1, color="#3b82f6")
        ax.set_xlabel("Mean Predicted Probability", fontsize=11)
        ax.set_ylabel("Fraction of Positives", fontsize=11)
        ax.set_title(f"Calibration Curve — {run_name}", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()

        out_path = self._output_dir / f"calibration_{run_name}.png"
        fig.savefig(out_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path, float(ece)

    def _plot_punzi_fom(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        run_name: str,
    ) -> tuple[Path, float, float]:
        """
        Compute and plot the Punzi Figure of Merit (FOM) vs threshold.

        Punzi FOM = S / (a/2 + sqrt(B))
        where:
            S = number of true positive events (signal passing cut)
            B = number of false positive events (background passing cut)
            a = significance target (default 5σ)

        This is the standard HEP metric for optimising a cut threshold
        to maximise discovery significance.

        Returns:
            (path, best_fom, best_threshold)
        """
        a = self.config.significance_target
        thresholds = np.linspace(0.0, 1.0, self.config.n_threshold_steps)
        fom_values = []

        for t in thresholds:
            preds = y_scores >= t
            S = float(((preds) & (y_true == 1)).sum())  # true positives
            B = float(((preds) & (y_true == 0)).sum())  # false positives
            fom = S / (a / 2.0 + np.sqrt(max(B, 1e-9)))
            fom_values.append(fom)

        fom_values = np.array(fom_values)
        best_idx = int(np.argmax(fom_values))
        best_fom = float(fom_values[best_idx])
        best_thresh = float(thresholds[best_idx])

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(thresholds, fom_values, color="#f59e0b", lw=2.5, label=f"Punzi FOM (a={a})")
        ax.axvline(x=best_thresh, color="#ef4444", linestyle="--", alpha=0.8,
                   label=f"Optimal cut: {best_thresh:.3f} (FOM={best_fom:.2f})")
        ax.scatter([best_thresh], [best_fom], color="#ef4444", s=80, zorder=5)
        ax.set_xlabel("Decision Threshold", fontsize=11)
        ax.set_ylabel(f"Punzi FOM = S / ({a}/2 + √B)", fontsize=11)
        ax.set_title(f"Punzi Figure of Merit — {run_name}", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)

        # Annotate the physics formula
        ax.text(0.02, 0.95, f"FOM = S / ({a}/2 + √B)", transform=ax.transAxes,
                fontsize=9, color="gray", va="top")

        fig.tight_layout()
        out_path = self._output_dir / f"punzi_fom_{run_name}.png"
        fig.savefig(out_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path, best_fom, best_thresh

    # ── MLflow logging ────────────────────────────────────────────────────────

    def _log_to_mlflow(self, mlflow_logger: Any, result: EvaluationResult) -> None:
        """Log all metrics and plot artifacts to MLflow."""
        try:
            # Log scalar metrics
            mlflow_logger.log_metrics(result.to_dict())

            # Log plot artifacts
            for plot_path in result.plot_paths:
                path = Path(plot_path)
                if path.exists():
                    mlflow_logger.log_artifact_file(
                        path,
                        artifact_dir=self.config.mlflow_artifact_dir,
                    )

            log.info(
                "Evaluation results logged to MLflow",
                n_plots=len(result.plot_paths),
            )
        except Exception as e:
            log.warning("Failed to log evaluation to MLflow", error=str(e))

    # ── Scalar-only helpers ────────────────────────────────────────────────────

    @staticmethod
    def _compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        from sklearn.metrics import roc_auc_score
        try:
            return float(roc_auc_score(y_true, y_scores))
        except Exception:
            return 0.5

    @staticmethod
    def _compute_ap(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        from sklearn.metrics import average_precision_score
        try:
            return float(average_precision_score(y_true, y_scores))
        except Exception:
            return float(y_true.mean())
