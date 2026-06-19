"""
Full ETL pipeline: Read → Validate → Quality Cuts → Normalize → Split → Save.

This is the primary entry point for data preparation. It reads raw HIGGS data,
runs validation, applies physics quality cuts, normalizes features, splits into
train/val/test, and saves versioned parquet files.

The output parquet files are versioned by a hash of the ETL config so that
any config change produces a new, non-overwriting dataset version.

Usage:
    # From command line
    python -m src.ingestion.etl_pipeline

    # From Python
    from src.ingestion.etl_pipeline import ETLPipeline
    pipeline = ETLPipeline.from_config("configs/system.yaml")
    splits = pipeline.run()
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.ingestion.data_validator import HiggsValidator
from src.ingestion.higgs_reader import FEATURE_COLUMNS, LABEL_COLUMN, HiggsReader
from src.utils.logging_config import configure_from_config, get_logger

log = get_logger(__name__)


# ─── ETL Config ──────────────────────────────────────────────────────────────

@dataclass
class ETLConfig:
    """Configuration for the ETL pipeline."""

    # Input
    raw_filepath: str = "data/raw/HIGGS.csv"
    n_samples: int | None = 500_000  # None = full 11M

    # Quality cuts
    apply_quality_cuts: bool = True
    lepton_pt_min: float = 0.0        # Applied if > 0 and column exists
    met_min: float = 0.0

    # Normalization
    normalization: Literal["standard", "robust", "minmax", "none"] = "standard"

    # Splitting
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    random_seed: int = 42

    # Output
    processed_dir: str = "data/processed"
    dataset_name: str = "higgs"

    # Caching
    use_reader_cache: bool = True
    cache_dir: str = "data/processed/reader_cache"

    def config_hash(self) -> str:
        """SHA-256 hash of this config — used for versioned output filenames."""
        d = {k: str(v) for k, v in asdict(self).items()}
        content = json.dumps(d, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:10]

    @classmethod
    def from_omegaconf(cls, cfg: OmegaConf) -> "ETLConfig":
        """Build ETLConfig from OmegaConf system config."""
        data_cfg = OmegaConf.select(cfg, "data", default={})
        split_cfg = OmegaConf.select(cfg, "data.split", default={})
        cuts_cfg = OmegaConf.select(cfg, "data.quality_cuts", default={})
        paths_cfg = OmegaConf.select(cfg, "paths", default={})

        return cls(
            raw_filepath=str(Path(
                OmegaConf.select(paths_cfg, "raw_dir", default="data/raw")
            ) / "HIGGS.csv"),
            n_samples=OmegaConf.select(data_cfg, "higgs.n_samples_dev", default=500_000),
            apply_quality_cuts=OmegaConf.select(cuts_cfg, "apply", default=True),
            lepton_pt_min=OmegaConf.select(cuts_cfg, "lepton_pt_min", default=0.0),
            met_min=OmegaConf.select(cuts_cfg, "met_min", default=0.0),
            normalization=OmegaConf.select(data_cfg, "normalization", default="standard"),
            train_frac=OmegaConf.select(split_cfg, "train", default=0.70),
            val_frac=OmegaConf.select(split_cfg, "val", default=0.15),
            test_frac=OmegaConf.select(split_cfg, "test", default=0.15),
            random_seed=OmegaConf.select(split_cfg, "random_seed", default=42),
            processed_dir=str(OmegaConf.select(paths_cfg, "processed_dir", default="data/processed")),
        )


# ─── Dataset splits container ─────────────────────────────────────────────────

@dataclass
class DataSplits:
    """Container for train/val/test splits with metadata."""
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    scaler: StandardScaler | RobustScaler | None
    version: str
    metadata: dict = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_val(self) -> int:
        return len(self.X_val)

    @property
    def n_test(self) -> int:
        return len(self.X_test)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def summary(self) -> str:
        return (
            f"DataSplits v{self.version}: "
            f"train={self.n_train:,} | val={self.n_val:,} | test={self.n_test:,} | "
            f"features={self.n_features}"
        )


# ─── ETL Pipeline ─────────────────────────────────────────────────────────────

class ETLPipeline:
    """
    Full ETL pipeline for particle physics data preparation.

    Steps:
        1. Read raw data (HIGGS CSV)
        2. Validate schema and data quality
        3. Apply physics quality cuts
        4. Normalize features (fit on train, transform all)
        5. Split into train/val/test
        6. Save versioned parquet files

    Args:
        config: ETLConfig instance.

    Example:
        pipeline = ETLPipeline(ETLConfig(n_samples=500_000))
        splits = pipeline.run()

        # Or load from system config:
        pipeline = ETLPipeline.from_config("configs/system.yaml")
        splits = pipeline.run()
    """

    def __init__(self, config: ETLConfig) -> None:
        self.config = config
        self.processed_dir = Path(config.processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config_path: str | Path = "configs/system.yaml") -> "ETLPipeline":
        """Load ETL config from a system YAML config file."""
        cfg = OmegaConf.load(config_path)
        return cls(ETLConfig.from_omegaconf(cfg))

    def run(self, force: bool = False) -> DataSplits:
        """
        Execute the full ETL pipeline.

        Args:
            force: If True, re-run even if versioned outputs already exist.

        Returns:
            DataSplits object with train/val/test splits.
        """
        version = self.config.config_hash()
        log.info("ETL pipeline starting", version=version, config=asdict(self.config))

        # ── Check for existing outputs ────────────────────────────────────────
        if not force and self._outputs_exist(version):
            log.info("Versioned outputs found — loading from cache", version=version)
            return self._load_splits(version)

        # ── Step 1: Read ──────────────────────────────────────────────────────
        df = self._step_read()

        # ── Step 2: Validate ──────────────────────────────────────────────────
        df = self._step_validate(df)

        # ── Step 3: Quality cuts ──────────────────────────────────────────────
        df = self._step_quality_cuts(df)

        # ── Step 4: Split (before normalization to prevent data leakage) ──────
        splits_raw = self._step_split(df)

        # ── Step 5: Normalize (fit on train only) ────────────────────────────
        splits, scaler = self._step_normalize(splits_raw)

        # ── Step 6: Save ──────────────────────────────────────────────────────
        self._step_save(splits, version)

        log.info("ETL pipeline complete", summary=splits.summary())
        return splits

    # ── Pipeline steps ────────────────────────────────────────────────────────

    def _step_read(self) -> pd.DataFrame:
        log.info("Step 1/6: Reading data", source=self.config.raw_filepath)
        reader = HiggsReader(
            filepath=self.config.raw_filepath,
            cache_dir=self.config.cache_dir if self.config.use_reader_cache else None,
        )
        df = reader.read(
            n_samples=self.config.n_samples,
            random_seed=self.config.random_seed,
            use_cache=self.config.use_reader_cache,
        )
        log.info("Read complete", n_rows=len(df))
        return df

    def _step_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("Step 2/6: Validating data", n_rows=len(df))
        validator = HiggsValidator(raise_on_failure=True)
        validated = validator.validate(df)
        log.info("Validation passed")
        return validated

    def _step_quality_cuts(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.apply_quality_cuts:
            log.info("Step 3/6: Quality cuts skipped (disabled in config)")
            return df

        log.info("Step 3/6: Applying physics quality cuts")
        original_len = len(df)
        mask = pd.Series(True, index=df.index)

        # Lepton pT cut
        if self.config.lepton_pt_min > 0 and "lepton_pt" in df.columns:
            mask &= df["lepton_pt"] > self.config.lepton_pt_min

        # MET cut
        if self.config.met_min > 0 and "missing_energy_magnitude" in df.columns:
            mask &= df["missing_energy_magnitude"] > self.config.met_min

        # Remove events with physically impossible values (all-zero kinematics)
        kin_cols = ["lepton_pt", "jet1_pt"]
        existing_kin = [c for c in kin_cols if c in df.columns]
        if existing_kin:
            mask &= df[existing_kin].sum(axis=1) > 0

        df = df[mask].reset_index(drop=True)
        efficiency = len(df) / original_len
        log.info(
            "Quality cuts complete",
            before=original_len,
            after=len(df),
            efficiency=f"{efficiency:.1%}",
        )
        return df

    def _step_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        log.info("Step 4/6: Splitting train/val/test")
        X = df[FEATURE_COLUMNS]
        y = df[LABEL_COLUMN]

        # First split: train vs (val + test)
        test_val_frac = self.config.val_frac + self.config.test_frac
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=test_val_frac,
            stratify=y,
            random_state=self.config.random_seed,
        )

        # Second split: val vs test (from the held-out portion)
        val_rel_frac = self.config.val_frac / test_val_frac
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_rel_frac),
            stratify=y_temp,
            random_state=self.config.random_seed,
        )

        log.info(
            "Split complete",
            train=len(X_train),
            val=len(X_val),
            test=len(X_test),
            signal_train=f"{y_train.mean():.3f}",
        )
        return (X_train, X_val, X_test), (y_train, y_val, y_test)

    def _step_normalize(
        self,
        splits: tuple,
    ) -> tuple[DataSplits, StandardScaler | RobustScaler | None]:
        log.info("Step 5/6: Normalizing features", method=self.config.normalization)
        (X_train, X_val, X_test), (y_train, y_val, y_test) = splits

        scaler: StandardScaler | RobustScaler | None = None

        if self.config.normalization == "standard":
            scaler = StandardScaler()
        elif self.config.normalization == "robust":
            scaler = RobustScaler()
        elif self.config.normalization == "none":
            scaler = None
        else:
            raise ValueError(f"Unknown normalization: {self.config.normalization}")

        if scaler is not None:
            X_train_arr = scaler.fit_transform(X_train)
            X_val_arr = scaler.transform(X_val)
            X_test_arr = scaler.transform(X_test)
            X_train = pd.DataFrame(X_train_arr, columns=FEATURE_COLUMNS)
            X_val = pd.DataFrame(X_val_arr, columns=FEATURE_COLUMNS)
            X_test = pd.DataFrame(X_test_arr, columns=FEATURE_COLUMNS)
            log.info(
                "Normalization complete",
                feature_mean_range=f"[{X_train.mean().min():.3f}, {X_train.mean().max():.3f}]",
                feature_std_range=f"[{X_train.std().min():.3f}, {X_train.std().max():.3f}]",
            )

        version = self.config.config_hash()
        data_splits = DataSplits(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train.reset_index(drop=True),
            y_val=y_val.reset_index(drop=True),
            y_test=y_test.reset_index(drop=True),
            feature_names=FEATURE_COLUMNS,
            scaler=scaler,
            version=version,
            metadata={
                "normalization": self.config.normalization,
                "n_samples": self.config.n_samples,
                "random_seed": self.config.random_seed,
                "config_hash": version,
            },
        )
        return data_splits, scaler

    def _step_save(self, splits: DataSplits, version: str) -> None:
        log.info("Step 6/6: Saving versioned parquet files", version=version)
        out_dir = self.processed_dir / version
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save feature matrices + labels as separate parquets
        splits.X_train.assign(label=splits.y_train).to_parquet(
            out_dir / "train.parquet", index=False
        )
        splits.X_val.assign(label=splits.y_val).to_parquet(
            out_dir / "val.parquet", index=False
        )
        splits.X_test.assign(label=splits.y_test).to_parquet(
            out_dir / "test.parquet", index=False
        )

        # Save metadata
        meta = {
            **splits.metadata,
            "version": version,
            "n_train": splits.n_train,
            "n_val": splits.n_val,
            "n_test": splits.n_test,
            "n_features": splits.n_features,
            "feature_names": splits.feature_names,
            "output_dir": str(out_dir),
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Write a "latest" pointer
        with open(self.processed_dir / "latest.txt", "w") as f:
            f.write(version)

        log.info("Save complete", output_dir=str(out_dir))

    def _outputs_exist(self, version: str) -> bool:
        out_dir = self.processed_dir / version
        return (
            (out_dir / "train.parquet").exists()
            and (out_dir / "val.parquet").exists()
            and (out_dir / "test.parquet").exists()
        )

    def _load_splits(self, version: str) -> DataSplits:
        """Load existing versioned splits from disk."""
        out_dir = self.processed_dir / version

        train = pd.read_parquet(out_dir / "train.parquet")
        val = pd.read_parquet(out_dir / "val.parquet")
        test = pd.read_parquet(out_dir / "test.parquet")

        with open(out_dir / "metadata.json") as f:
            meta = json.load(f)

        feature_names = meta["feature_names"]

        return DataSplits(
            X_train=train[feature_names],
            X_val=val[feature_names],
            X_test=test[feature_names],
            y_train=train[LABEL_COLUMN],
            y_val=val[LABEL_COLUMN],
            y_test=test[LABEL_COLUMN],
            feature_names=feature_names,
            scaler=None,  # Scaler not persisted in Phase 1 (fitted fresh from raw)
            version=version,
            metadata=meta,
        )


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    """Run the ETL pipeline from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run the particle physics ETL pipeline")
    parser.add_argument(
        "--config", type=str, default="configs/system.yaml", help="Path to system config YAML"
    )
    parser.add_argument(
        "--n-samples", type=int, default=None, help="Override n_samples from config"
    )
    parser.add_argument("--force", action="store_true", help="Re-run even if outputs exist")
    args = parser.parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)
    configure_from_config(cfg)

    etl_config = ETLConfig.from_omegaconf(cfg)
    if args.n_samples is not None:
        etl_config.n_samples = args.n_samples

    pipeline = ETLPipeline(etl_config)
    splits = pipeline.run(force=args.force)
    print(f"\n✓ {splits.summary()}")
    print(f"  Output: data/processed/{splits.version}/")


if __name__ == "__main__":
    main()
