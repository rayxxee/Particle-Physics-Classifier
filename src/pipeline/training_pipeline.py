"""
End-to-end training pipeline: ingest → features → train → evaluate → log.

This is the main entry point for training any model architecture.
It orchestrates all the modules in the correct order and handles
MLflow logging throughout.

Usage:
    # Train MLP with defaults
    python -m src.pipeline.training_pipeline --model mlp

    # Train with custom config
    python -m src.pipeline.training_pipeline \\
        --model mlp \\
        --config configs/mlp_default.yaml \\
        --n-samples 500000 \\
        --force-etl

    # From Python
    from src.pipeline.training_pipeline import TrainingPipeline
    pipeline = TrainingPipeline.from_config("configs/system.yaml")
    result = pipeline.run("mlp")
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score

from src.experiment_tracking.experiment_registry import ExperimentRegistry
from src.experiment_tracking.mlflow_logger import MLflowLogger
from src.features.feature_store import FeatureConfig, FeatureStore
from src.ingestion.etl_pipeline import ETLConfig, ETLPipeline
from src.models.mlp.config import MLPConfig
from src.models.mlp.model import MLPModel
from src.utils.logging_config import configure_logging, get_logger

# Phase 2 models — imported lazily in _build_model() to keep startup fast
# from src.models.bdt.config import BDTConfig
# from src.models.bdt.model import BDTModel
# from src.models.gnn.config import GNNConfig
# from src.models.gnn.model import GNNModel
# from src.models.transformer.config import TransformerConfig
# from src.models.transformer.model import TransformerModel

log = get_logger(__name__)


class TrainingPipeline:
    """
    Orchestrates the full training workflow for a given model architecture.

    Steps:
        1. ETL: Read raw data → validate → quality cuts → normalize → split
        2. Feature engineering: Compute physics features → cache to store
        3. Load: Pull features from store
        4. Train: Fit model with MLflow tracking
        5. Evaluate: Compute test metrics
        6. Register: Update experiment registry, optionally promote to Production

    Args:
        system_config_path: Path to configs/system.yaml.

    Example:
        pipeline = TrainingPipeline()
        result = pipeline.run("mlp")
        print(f"Test AUC: {result['test_auc']:.4f}")
    """

    def __init__(
        self,
        system_config_path: str | Path = "configs/system.yaml",
    ) -> None:
        self.system_config_path = Path(system_config_path)
        if self.system_config_path.exists():
            self.cfg = OmegaConf.load(self.system_config_path)
        else:
            self.cfg = OmegaConf.create({})
            log.warning("System config not found, using defaults", path=str(system_config_path))

        # Set up MLflow
        mlflow_cfg = OmegaConf.select(self.cfg, "mlflow", default={})
        self.logger = MLflowLogger(
            experiment_name=OmegaConf.select(mlflow_cfg, "experiment_name",
                                              default="particle_physics_classifier"),
            tracking_uri=OmegaConf.select(mlflow_cfg, "tracking_uri", default=None),
        )
        self.registry = ExperimentRegistry()

    @classmethod
    def from_config(cls, config_path: str | Path = "configs/system.yaml") -> "TrainingPipeline":
        return cls(system_config_path=config_path)

    def run(
        self,
        model_type: str = "mlp",
        model_config_path: str | Path | None = None,
        n_samples: int | None = None,
        force_etl: bool = False,
        force_features: bool = False,
        run_name: str | None = None,
        promote_to_production: bool = True,
    ) -> dict[str, Any]:
        """
        Run the full training pipeline.

        Args:
            model_type:           Architecture to train ("mlp", "bdt", "gnn", etc.).
            model_config_path:    Path to model-specific YAML config. Auto-resolved if None.
            n_samples:            Override n_samples from system config.
            force_etl:            Re-run ETL even if versioned output exists.
            force_features:       Re-build feature store even if cached.
            run_name:             MLflow run name. Auto-generated if None.
            promote_to_production: Whether to auto-promote if AUC qualifies.

        Returns:
            Dict with keys: run_id, test_auc, val_auc, best_epoch, fit_time_s.
        """
        log.info(
            "Training pipeline started",
            model=model_type,
            n_samples=n_samples,
        )
        t0 = time.time()

        # ── Step 1: ETL ───────────────────────────────────────────────────────
        log.info("Step 1/5: Running ETL")
        etl_config = ETLConfig.from_omegaconf(self.cfg)
        if n_samples is not None:
            etl_config.n_samples = n_samples
        splits = ETLPipeline(etl_config).run(force=force_etl)

        # ── Step 2: Feature Engineering ────────────────────────────────────────
        log.info("Step 2/5: Building feature store")
        feature_config = FeatureConfig(
            include_low_level=True,
            include_dataset_hl=True,
            include_derived_hl=True,
        )
        store = FeatureStore(feature_config)
        store.build(splits, force=force_features)

        X_train, y_train = store.load("train")
        X_val, y_val = store.load("val")
        X_test, y_test = store.load("test")
        feature_names = store.feature_names()

        log.info(
            "Features loaded",
            n_train=len(X_train),
            n_val=len(X_val),
            n_test=len(X_test),
            n_features=len(feature_names),
        )

        # ── Step 3: Build model ────────────────────────────────────────────────
        log.info(f"Step 3/5: Building {model_type} model")
        model, model_config = self._build_model(model_type, model_config_path)
        model_config.input_dim = len(feature_names)

        # ── Step 4: Train with MLflow tracking ────────────────────────────────
        log.info("Step 4/5: Training model")
        run_name = run_name or f"{model_type}_{int(time.time())}"
        tags = {
            "model_type": model_type,
            "dataset": "higgs",
            "etl_version": splits.version,
            "n_features": str(len(feature_names)),
        }

        with self.logger.start_run(run_name=run_name, tags=tags) as run:
            run_id = run.info.run_id if run else "local"

            # Log params
            self.logger.log_params(model_config.to_dict())
            self.logger.log_params({
                "etl_version": splits.version,
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_test": len(X_test),
                "n_features": len(feature_names),
                "feature_store_version": store._version,
            })

            # Train
            history = model.fit(X_train, y_train, X_val, y_val)

            # ── Step 5: Evaluate ──────────────────────────────────────────────
            log.info("Step 5/5: Evaluating on test set")
            val_scores = model.predict_proba(X_val)
            test_scores = model.predict_proba(X_test)

            val_auc = float(roc_auc_score(y_val, val_scores))
            test_auc = float(roc_auc_score(y_test, test_scores))

            log.info(
                "Evaluation complete",
                val_auc=f"{val_auc:.4f}",
                test_auc=f"{test_auc:.4f}",
            )

            # Log full run artifacts
            self.logger.log_history(history)
            self.logger.log_summary({
                "val_auc": val_auc,
                "test_auc": test_auc,
                "best_epoch": float(history.get("best_epoch", 0)),
                "fit_time_s": float(history.get("fit_time_s", 0)),
            })
            self.logger.log_training_curves(history)
            self.logger.log_roc_curve(
                y_true=np.array(y_test),
                y_scores=test_scores,
                model_name=model_type,
            )
            self.logger.log_model(model)

        # ── Register ──────────────────────────────────────────────────────────
        self.registry.register_run(
            model_name=model_type,
            run_id=run_id,
            metrics={"val_auc": val_auc, "test_auc": test_auc},
            promote=promote_to_production,
        )

        total_time = time.time() - t0
        result = {
            "run_id": run_id,
            "model_type": model_type,
            "val_auc": val_auc,
            "test_auc": test_auc,
            "best_epoch": history.get("best_epoch"),
            "fit_time_s": history.get("fit_time_s"),
            "total_time_s": total_time,
            "n_features": len(feature_names),
        }

        log.info(
            "Pipeline complete",
            val_auc=f"{val_auc:.4f}",
            test_auc=f"{test_auc:.4f}",
            total_time=f"{total_time:.1f}s",
        )
        return result

    def _build_model(
        self,
        model_type: str,
        config_path: str | Path | None = None,
    ) -> tuple[Any, Any]:
        """Instantiate the model and its config for the given architecture."""
        if model_type == "mlp":
            if config_path is None:
                config_path = Path("configs/mlp_default.yaml")
            if Path(config_path).exists():
                config = MLPConfig.from_yaml(config_path)
            else:
                config = MLPConfig()
            model = MLPModel(config)
            return model, config

        elif model_type in ("bdt", "xgboost", "lightgbm"):
            from src.models.bdt.config import BDTConfig
            from src.models.bdt.model import BDTModel

            if config_path is not None and Path(config_path).exists():
                config = BDTConfig.from_yaml(config_path)
            else:
                # Infer model_type sub-variant from CLI flag
                bdt_backend = "lightgbm" if model_type == "lightgbm" else "xgboost"
                config = BDTConfig(model_type=bdt_backend)
            model = BDTModel(config)
            return model, config

        elif model_type == "gnn":
            from src.models.gnn.config import GNNConfig
            from src.models.gnn.model import GNNModel

            if config_path is not None and Path(config_path).exists():
                config = GNNConfig.from_yaml(config_path)
            else:
                config = GNNConfig()
            model = GNNModel(config)
            return model, config

        elif model_type == "transformer":
            from src.models.transformer.config import TransformerConfig
            from src.models.transformer.model import TransformerModel

            if config_path is not None and Path(config_path).exists():
                config = TransformerConfig.from_yaml(config_path)
            else:
                config = TransformerConfig()
            model = TransformerModel(config)
            return model, config

        elif model_type == "normalizing_flow":
            raise NotImplementedError(
                "Normalizing Flow is planned for Phase 3. "
                "Available: mlp, bdt, xgboost, lightgbm, gnn, transformer"
            )
        else:
            raise ValueError(
                f"Unknown model type: {model_type!r}. "
                f"Available: mlp, bdt, xgboost, lightgbm, gnn, transformer"
            )


def main() -> None:
    """CLI entry point for the training pipeline."""
    configure_logging(level="INFO", format="pretty")

    parser = argparse.ArgumentParser(
        description="Train a particle physics event classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.pipeline.training_pipeline --model mlp
  python -m src.pipeline.training_pipeline --model mlp --n-samples 100000
  python -m src.pipeline.training_pipeline --model mlp --force-etl --run-name my_run
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["mlp", "bdt", "xgboost", "lightgbm", "gnn", "transformer"],
        help="Model architecture to train (default: mlp)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/system.yaml",
        help="Path to system config YAML (default: configs/system.yaml)",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to model-specific config YAML",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of training events (default: from system config)",
    )
    parser.add_argument(
        "--force-etl",
        action="store_true",
        help="Re-run ETL even if versioned output exists",
    )
    parser.add_argument(
        "--force-features",
        action="store_true",
        help="Re-build feature store even if cached",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name (auto-generated if not set)",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Do not auto-promote to Production even if AUC qualifies",
    )
    args = parser.parse_args()

    pipeline = TrainingPipeline.from_config(args.config)
    result = pipeline.run(
        model_type=args.model,
        model_config_path=args.model_config,
        n_samples=args.n_samples,
        force_etl=args.force_etl,
        force_features=args.force_features,
        run_name=args.run_name,
        promote_to_production=not args.no_promote,
    )

    print("\n" + "─" * 60)
    print("Training Pipeline Results")
    print("─" * 60)
    for key, value in result.items():
        if isinstance(value, float):
            print(f"  {key:<22}: {value:.4f}")
        else:
            print(f"  {key:<22}: {value}")
    print("─" * 60)
    print(f"\n✓ Open MLflow UI: mlflow ui --host 0.0.0.0 --port 5000")


if __name__ == "__main__":
    main()
