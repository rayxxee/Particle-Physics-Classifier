"""
Experiment registry: track and promote best models per architecture.

The registry maintains a record of the best-performing run for each
model type. The "Production" model is what the FastAPI endpoint loads.

Integration with MLflow Model Registry:
    - Best model per architecture is tagged "Production"
    - API loads the Production model at startup
    - Auto-promotion rules: new model must beat production by > 0.005 AUC

Usage:
    from src.experiment_tracking.experiment_registry import ExperimentRegistry

    registry = ExperimentRegistry()
    registry.register_run(
        model_name="mlp",
        run_id="abc123",
        metrics={"val_auc": 0.823},
        promote=True,
    )
    best = registry.get_best("mlp")
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.logging_config import get_logger

log = get_logger(__name__)

PROMOTION_AUC_DELTA = 0.005   # New model must beat prod by this much to auto-promote
PRODUCTION_ALIAS = "Production"
CHALLENGER_ALIAS = "Challenger"


class ExperimentRegistry:
    """
    Manages the MLflow model registry for all architectures.

    Tracks the best run per model type and handles promotion of
    challenger models to the Production alias.

    Args:
        tracking_uri:  MLflow server URI. Defaults to local ./mlruns.
        registry_path: Local JSON file for backup (in case MLflow is down).

    Example:
        registry = ExperimentRegistry()
        registry.register_run("mlp", run_id, {"val_auc": 0.82}, promote=True)
        prod = registry.get_production("mlp")
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        registry_path: str | Path = "models/registry.json",
    ) -> None:
        self._tracking_uri = tracking_uri
        self._registry_path = Path(registry_path)
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            self._client = MlflowClient()
            self._mlflow_available = True
        except Exception as e:
            log.warning("MLflow registry unavailable", error=str(e))
            self._mlflow_available = False
            self._client = None

        # Load local registry
        self._registry = self._load_local_registry()

    # ── Registration ──────────────────────────────────────────────────────────

    def register_run(
        self,
        model_name: str,
        run_id: str,
        metrics: dict[str, float],
        model_uri: str | None = None,
        promote: bool = False,
        tags: dict[str, str] | None = None,
    ) -> None:
        """
        Register a completed run for a given model architecture.

        Args:
            model_name: Architecture name (e.g., "mlp", "gnn").
            run_id:     MLflow run ID.
            metrics:    Evaluation metrics dict. Must include "val_auc".
            model_uri:  MLflow artifact URI to the model. Auto-resolved if None.
            promote:    If True, auto-promote to Production if AUC qualifies.
            tags:       Additional tags to add to the registered model.
        """
        val_auc = metrics.get("val_auc", 0.0)
        timestamp = datetime.utcnow().isoformat()

        entry: dict[str, Any] = {
            "model_name": model_name,
            "run_id": run_id,
            "metrics": metrics,
            "val_auc": val_auc,
            "registered_at": timestamp,
            "tags": tags or {},
        }

        # Update local registry
        if model_name not in self._registry:
            self._registry[model_name] = {"runs": [], "production": None}

        self._registry[model_name]["runs"].append(entry)
        self._save_local_registry()

        log.info(
            "Run registered",
            model=model_name,
            run_id=run_id[:8],
            val_auc=f"{val_auc:.4f}",
        )

        # MLflow Model Registry
        if self._mlflow_available and model_uri:
            self._register_to_mlflow(model_name, run_id, model_uri, tags)

        # Auto-promote
        if promote:
            self._try_promote(model_name, run_id, val_auc, model_uri)

    def get_production(self, model_name: str) -> dict[str, Any] | None:
        """
        Return the Production model entry for a given architecture.

        Returns:
            Dict with run_id, metrics, and URI, or None if no production model.
        """
        if model_name not in self._registry:
            return None
        return self._registry[model_name].get("production")

    def get_best(self, model_name: str) -> dict[str, Any] | None:
        """Return the run with the highest val_auc for a given architecture."""
        if model_name not in self._registry:
            return None
        runs = self._registry[model_name].get("runs", [])
        if not runs:
            return None
        return max(runs, key=lambda r: r.get("val_auc", 0.0))

    def get_leaderboard(self) -> list[dict[str, Any]]:
        """
        Return a leaderboard of all architectures sorted by best val_auc.

        Returns:
            List of dicts, each with model_name and best_val_auc.
        """
        board = []
        for model_name, data in self._registry.items():
            best = self.get_best(model_name)
            if best:
                board.append({
                    "model_name": model_name,
                    "best_val_auc": best["val_auc"],
                    "run_id": best["run_id"],
                    "registered_at": best["registered_at"],
                    "is_production": (
                        self._registry[model_name].get("production", {}) or {}
                    ).get("run_id") == best["run_id"],
                })
        return sorted(board, key=lambda x: x["best_val_auc"], reverse=True)

    # ── Promotion ─────────────────────────────────────────────────────────────

    def _try_promote(
        self,
        model_name: str,
        run_id: str,
        val_auc: float,
        model_uri: str | None = None,
    ) -> bool:
        """
        Promote run to Production if it beats the current production model.

        Returns True if promotion occurred.
        """
        current_prod = self._registry[model_name].get("production")

        if current_prod is None:
            # No production model yet — promote automatically
            self._promote(model_name, run_id, val_auc, model_uri)
            return True

        prod_auc = current_prod.get("val_auc", 0.0)
        improvement = val_auc - prod_auc

        if improvement >= PROMOTION_AUC_DELTA:
            log.info(
                "Promoting to Production",
                model=model_name,
                new_auc=f"{val_auc:.4f}",
                old_auc=f"{prod_auc:.4f}",
                delta=f"+{improvement:.4f}",
            )
            self._promote(model_name, run_id, val_auc, model_uri)
            return True
        else:
            log.info(
                "Challenger did not beat Production",
                model=model_name,
                challenger_auc=f"{val_auc:.4f}",
                prod_auc=f"{prod_auc:.4f}",
                required_delta=PROMOTION_AUC_DELTA,
            )
            return False

    def _promote(
        self,
        model_name: str,
        run_id: str,
        val_auc: float,
        model_uri: str | None = None,
    ) -> None:
        """Mark a run as Production in local registry and MLflow."""
        self._registry[model_name]["production"] = {
            "run_id": run_id,
            "val_auc": val_auc,
            "promoted_at": datetime.utcnow().isoformat(),
            "model_uri": model_uri,
        }
        self._save_local_registry()

        # MLflow: add tag to run
        if self._mlflow_available:
            try:
                self._client.set_tag(run_id, "production", "true")
                self._client.set_tag(run_id, "model_stage", PRODUCTION_ALIAS)
                log.info("MLflow run tagged as Production", run_id=run_id[:8])
            except Exception as e:
                log.warning("Failed to tag MLflow run", error=str(e))

    # ── Local registry I/O ────────────────────────────────────────────────────

    def _load_local_registry(self) -> dict:
        if self._registry_path.exists():
            with open(self._registry_path) as f:
                return json.load(f)
        return {}

    def _save_local_registry(self) -> None:
        with open(self._registry_path, "w") as f:
            json.dump(self._registry, f, indent=2, default=str)

    # ── MLflow Model Registry ─────────────────────────────────────────────────

    def _register_to_mlflow(
        self,
        model_name: str,
        run_id: str,
        model_uri: str,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Register model version in MLflow Model Registry."""
        try:
            # Ensure registered model exists
            try:
                self._client.get_registered_model(model_name)
            except mlflow.exceptions.MlflowException:
                self._client.create_registered_model(model_name)

            # Create new version
            mv = self._client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run_id,
            )
            log.info(
                "Model version registered",
                name=model_name,
                version=mv.version,
                run_id=run_id[:8],
            )
        except Exception as e:
            log.warning("Failed to register model version", error=str(e))

    def __repr__(self) -> str:
        models = list(self._registry.keys())
        return f"ExperimentRegistry(models={models})"
