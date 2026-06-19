"""
Feature store: compute-once, cache-to-disk, version-by-config-hash.

The FeatureStore wraps feature computation so that:
- Features are computed once and cached to parquet
- Cache keys are derived from the ETL data version + feature config hash
- New model runs read from the store (fast), not raw data
- Cache invalidation is automatic when config changes

This is a lightweight alternative to Feast that is appropriate for Phase 1.
Feast integration is planned for Phase 2 (adds serving-time feature retrieval
for the FastAPI endpoint).

Usage:
    from src.features.feature_store import FeatureStore, FeatureConfig

    config = FeatureConfig(include_derived_hl=True)
    store = FeatureStore(config=config, cache_dir="data/processed/feature_store")

    # Build and cache features from ETL splits
    store.build(splits)

    # Load features (fast — hits cache on second call)
    X_train, y_train = store.load("train")
    X_val, y_val     = store.load("val")
    X_test, y_test   = store.load("test")
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.features.high_level_features import HighLevelFeatureBuilder
from src.features.low_level_features import LowLevelExtractor
from src.ingestion.etl_pipeline import DataSplits
from src.ingestion.higgs_reader import FEATURE_COLUMNS, LABEL_COLUMN
from src.utils.logging_config import get_logger

log = get_logger(__name__)


# ─── Feature Configuration ────────────────────────────────────────────────────

@dataclass
class FeatureConfig:
    """
    Configuration controlling which features are included in the feature store.

    Attributes:
        include_low_level:    Include 21 raw detector variables.
        include_dataset_hl:   Include 7 original HIGGS high-level features.
        include_derived_hl:   Include 7 derived kinematic observables.
        nan_fill_strategy:    How to handle NaN in derived features.
                              "zero" fills with 0.0, "median" fills with column median.
        dtype:                NumPy dtype for feature arrays.
    """
    include_low_level: bool = True
    include_dataset_hl: bool = True
    include_derived_hl: bool = True
    nan_fill_strategy: Literal["zero", "median"] = "median"
    dtype: str = "float32"

    def config_hash(self) -> str:
        d = {k: str(v) for k, v in asdict(self).items()}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]


# ─── Feature Store ────────────────────────────────────────────────────────────

class FeatureStore:
    """
    Compute-once, cache-to-parquet feature store.

    Combines ETL output with physics feature engineering and stores the
    result as versioned parquet files. Cache keys include both the ETL
    version and the feature config hash to ensure reproducibility.

    Args:
        config:    Feature configuration.
        cache_dir: Directory for feature caches.

    Example:
        store = FeatureStore(FeatureConfig(), cache_dir="data/processed/feature_store")
        store.build(splits)
        X_train, y_train = store.load("train")
    """

    SPLIT_NAMES = ["train", "val", "test"]

    def __init__(
        self,
        config: FeatureConfig | None = None,
        cache_dir: str | Path = "data/processed/feature_store",
    ) -> None:
        self.config = config or FeatureConfig()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._hl_builder = HighLevelFeatureBuilder(
            include_dataset_hl=self.config.include_dataset_hl,
            include_derived=self.config.include_derived_hl,
        )
        self._ll_extractor = LowLevelExtractor()
        self._version: str | None = None

    def build(
        self,
        splits: DataSplits,
        force: bool = False,
    ) -> str:
        """
        Compute features from ETL splits and cache to disk.

        Args:
            splits: DataSplits from ETLPipeline.run().
            force:  Re-compute even if cache exists.

        Returns:
            Version string (cache key).
        """
        version = self._make_version(splits.version)
        self._version = version

        if not force and self._cache_exists(version):
            log.info("Feature store cache hit", version=version)
            return version

        log.info("Building features", version=version, config=asdict(self.config))

        for split_name in self.SPLIT_NAMES:
            X_raw = getattr(splits, f"X_{split_name}")
            y = getattr(splits, f"y_{split_name}")
            df = X_raw.copy()
            df[LABEL_COLUMN] = y.values

            # Build high-level features
            df_enriched = self._hl_builder.build(df)

            # Handle NaNs from derived features (e.g. delta_r_jet1_jet2 when jet2 missing)
            df_enriched = self._fill_nans(df_enriched)

            # Select feature columns
            feature_cols = self._get_feature_columns(df_enriched)
            df_out = df_enriched[feature_cols + [LABEL_COLUMN]]

            # Save
            out_path = self._cache_path(version, split_name)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df_out.to_parquet(out_path, index=False)
            log.debug("Cached split", split=split_name, n_rows=len(df_out), n_features=len(feature_cols))

        # Save metadata
        meta = {
            "version": version,
            "etl_version": splits.version,
            "feature_config": asdict(self.config),
            "feature_names": self._get_feature_columns(df_enriched),
            "n_features": len(self._get_feature_columns(df_enriched)),
        }
        with open(self._meta_path(version), "w") as f:
            json.dump(meta, f, indent=2)

        log.info(
            "Feature store build complete",
            version=version,
            n_features=meta["n_features"],
        )
        return version

    def load(
        self,
        split: Literal["train", "val", "test"],
        version: str | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Load features for a split from cache.

        Args:
            split:   Which split to load ("train", "val", "test").
            version: Version string. If None, uses the last built version.

        Returns:
            (X, y) — feature DataFrame and label Series.
        """
        version = version or self._version or self._latest_version()
        if version is None:
            raise RuntimeError(
                "No feature store version found. Run store.build(splits) first."
            )

        path = self._cache_path(version, split)
        if not path.exists():
            raise FileNotFoundError(
                f"Feature cache not found: {path}\n"
                "Run store.build(splits) to create it."
            )

        df = pd.read_parquet(path)
        X = df.drop(columns=[LABEL_COLUMN])
        y = df[LABEL_COLUMN]
        log.debug("Loaded from feature store", split=split, n_rows=len(X), version=version)
        return X, y

    def feature_names(self, version: str | None = None) -> list[str]:
        """Return the list of feature column names for this store version."""
        version = version or self._version or self._latest_version()
        if version is None:
            return []
        with open(self._meta_path(version)) as f:
            return json.load(f)["feature_names"]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_version(self, etl_version: str) -> str:
        content = f"{etl_version}|{self.config.config_hash()}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def _cache_path(self, version: str, split: str) -> Path:
        return self.cache_dir / version / f"{split}.parquet"

    def _meta_path(self, version: str) -> Path:
        return self.cache_dir / version / "metadata.json"

    def _cache_exists(self, version: str) -> bool:
        return all(
            self._cache_path(version, s).exists()
            for s in self.SPLIT_NAMES
        )

    def _latest_version(self) -> str | None:
        """Return the most recently created version."""
        versions = [d for d in self.cache_dir.iterdir() if d.is_dir()]
        if not versions:
            return None
        latest = max(versions, key=lambda d: d.stat().st_mtime)
        return latest.name

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Return the list of feature columns to include."""
        all_cols = list(df.columns)
        exclude = {LABEL_COLUMN}

        # Build based on config
        desired: list[str] = []
        if self.config.include_low_level:
            desired += [
                c for c in FEATURE_COLUMNS[:21] if c in all_cols
            ]  # first 21 = low level
        if self.config.include_dataset_hl:
            desired += [
                c for c in FEATURE_COLUMNS[21:] if c in all_cols
            ]  # last 7 = dataset HL
        if self.config.include_derived_hl:
            desired += [
                c for c in self._hl_builder.DERIVED_COLUMNS if c in all_cols
            ]

        # Remove duplicates (maintain order)
        seen: set[str] = set()
        result = []
        for c in desired:
            if c not in seen and c not in exclude:
                seen.add(c)
                result.append(c)
        return result

    def _fill_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle NaN values according to config strategy."""
        strategy = self.config.nan_fill_strategy
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        nan_cols = [c for c in numeric_cols if df[c].isna().any()]

        if not nan_cols:
            return df

        log.debug("Filling NaN values", strategy=strategy, columns=nan_cols)

        if strategy == "zero":
            df[nan_cols] = df[nan_cols].fillna(0.0)
        elif strategy == "median":
            for col in nan_cols:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        return df
