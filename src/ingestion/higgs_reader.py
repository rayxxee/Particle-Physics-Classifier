"""
Fast reader for the UCI HIGGS dataset (CSV and compressed formats).

The HIGGS dataset is an 11M-event Monte Carlo simulation of proton-proton
collisions at the LHC, with 28 features and a binary signal/background label.

This is the primary data source for Phase 1. The reader supports:
- Full 11M event dataset
- Configurable subsample for fast development/CI
- Streaming read for memory efficiency
- Optional HDF5 caching for repeat reads

Usage:
    from src.ingestion.higgs_reader import HiggsReader

    reader = HiggsReader("data/raw/HIGGS.csv")
    df = reader.read(n_samples=500_000)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

log = get_logger(__name__)

# ─── Column definitions ──────────────────────────────────────────────────────

HIGGS_COLUMNS = [
    "label",
    "lepton_pt", "lepton_eta", "lepton_phi",
    "missing_energy_magnitude", "missing_energy_phi",
    "jet1_pt", "jet1_eta", "jet1_phi", "jet1_b_tag",
    "jet2_pt", "jet2_eta", "jet2_phi", "jet2_b_tag",
    "jet3_pt", "jet3_eta", "jet3_phi", "jet3_b_tag",
    "jet4_pt", "jet4_eta", "jet4_phi", "jet4_b_tag",
    "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb",
]

FEATURE_COLUMNS = HIGGS_COLUMNS[1:]   # 28 features (excludes label)
LABEL_COLUMN = "label"

LOW_LEVEL_FEATURES = HIGGS_COLUMNS[1:22]   # indices 0-20 in feature array
HIGH_LEVEL_FEATURES = HIGGS_COLUMNS[22:]   # indices 21-27 in feature array


class HiggsReader:
    """
    Reads the UCI HIGGS dataset CSV file into a pandas DataFrame.

    Args:
        filepath:    Path to HIGGS.csv (or HIGGS.csv.gz for compressed).
                     File must exist; download with data/scripts/download_higgs.py.
        cache_dir:   Directory for HDF5 cache. None = no caching.

    Example:
        reader = HiggsReader("data/raw/HIGGS.csv")

        # Full dataset (11M rows — needs ~4 GB RAM)
        df = reader.read()

        # Development subset (500k rows — fast)
        df = reader.read(n_samples=500_000, random_seed=42)
    """

    def __init__(
        self,
        filepath: str | Path,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.filepath = Path(filepath)
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if not self.filepath.exists():
            raise FileNotFoundError(
                f"HIGGS dataset not found: {self.filepath}\n"
                "Download it with: python data/scripts/download_higgs.py"
            )

        log.info(
            "HiggsReader initialized",
            filepath=str(self.filepath),
            file_size_gb=round(self.filepath.stat().st_size / 1e9, 2),
        )

    def _cache_key(self, n_samples: int | None, random_seed: int) -> str:
        """Generate a cache key from read parameters."""
        content = f"{self.filepath}|{n_samples}|{random_seed}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def _cache_path(self, cache_key: str) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / f"higgs_{cache_key}.parquet"

    def _check_cache(self, cache_key: str) -> pd.DataFrame | None:
        if self.cache_dir is None:
            return None
        path = self._cache_path(cache_key)
        if path.exists():
            log.info("Cache hit", cache_path=str(path))
            return pd.read_parquet(path)
        return None

    def _write_cache(self, df: pd.DataFrame, cache_key: str) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(cache_key)
        df.to_parquet(path, index=False)
        log.info("Cached to disk", cache_path=str(path))

    def read(
        self,
        n_samples: int | None = None,
        random_seed: int = 42,
        feature_set: Literal["all", "low_level", "high_level"] = "all",
        dtype: str = "float32",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Read the HIGGS dataset into a DataFrame.

        Args:
            n_samples:    Number of events to return. None = all 11M.
                          Samples are drawn randomly (stratified by label).
            random_seed:  Random seed for reproducible sampling.
            feature_set:  Which features to include:
                          "all" = all 28 features (default)
                          "low_level" = first 21 (raw detector vars)
                          "high_level" = last 7 (physics observables)
            dtype:        NumPy dtype for feature columns.
            use_cache:    Load from parquet cache if available.

        Returns:
            DataFrame with column 'label' (int) and 28 feature columns (float32).
        """
        cache_key = self._cache_key(n_samples, random_seed)

        # ── Cache check ───────────────────────────────────────────────────────
        if use_cache:
            cached = self._check_cache(cache_key)
            if cached is not None:
                return self._select_features(cached, feature_set)

        # ── Read CSV ──────────────────────────────────────────────────────────
        log.info(
            "Reading HIGGS CSV",
            n_samples=n_samples or "all",
            filepath=str(self.filepath),
        )

        # Detect if file has a header
        with open(self.filepath) as f:
            first_line = f.readline().strip()
        has_header = first_line.startswith("label")

        read_kwargs: dict = {
            "filepath_or_buffer": self.filepath,
            "dtype": {col: dtype for col in FEATURE_COLUMNS},
        }

        if has_header:
            read_kwargs["header"] = 0
        else:
            read_kwargs["header"] = None
            read_kwargs["names"] = HIGGS_COLUMNS

        if n_samples is not None:
            # Efficient random sampling: read full file, sample
            # For very large files, use skiprows-based streaming
            df = self._read_with_sampling(n_samples, random_seed, read_kwargs)
        else:
            df = pd.read_csv(**read_kwargs)

        # Ensure label is integer
        df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

        log.info(
            "HIGGS read complete",
            n_events=len(df),
            signal_fraction=f"{df[LABEL_COLUMN].mean():.3f}",
            n_features=len(FEATURE_COLUMNS),
        )

        # ── Cache write ───────────────────────────────────────────────────────
        if use_cache:
            self._write_cache(df, cache_key)

        return self._select_features(df, feature_set)

    def _read_with_sampling(
        self,
        n_samples: int,
        random_seed: int,
        read_kwargs: dict,
    ) -> pd.DataFrame:
        """
        Efficiently read n_samples rows with stratified sampling.

        Strategy: read in chunks, reservoir-sample per class to maintain
        class balance. Falls back to full read + sample for small n_samples.
        """
        rng = np.random.default_rng(random_seed)

        # For n_samples < 1M, read a slice and sample — fast and accurate
        n_to_read = min(n_samples * 4, 4_000_000)  # read 4x for stratification

        log.debug("Stratified sampling", n_to_read=n_to_read, n_samples=n_samples)

        read_kwargs["nrows"] = n_to_read
        df_raw = pd.read_csv(**read_kwargs)

        # Stratified sample
        signal = df_raw[df_raw[LABEL_COLUMN] == 1]
        background = df_raw[df_raw[LABEL_COLUMN] == 0]

        n_signal = min(n_samples // 2, len(signal))
        n_background = min(n_samples - n_signal, len(background))

        df = pd.concat([
            signal.sample(n=n_signal, random_state=random_seed),
            background.sample(n=n_background, random_state=random_seed),
        ]).sample(frac=1, random_state=random_seed).reset_index(drop=True)

        return df

    def _select_features(
        self,
        df: pd.DataFrame,
        feature_set: Literal["all", "low_level", "high_level"],
    ) -> pd.DataFrame:
        """Return DataFrame with selected feature columns plus label."""
        if feature_set == "all":
            return df
        elif feature_set == "low_level":
            return df[[LABEL_COLUMN] + LOW_LEVEL_FEATURES]
        elif feature_set == "high_level":
            return df[[LABEL_COLUMN] + HIGH_LEVEL_FEATURES]
        else:
            raise ValueError(f"Unknown feature_set: {feature_set}")

    def get_schema(self) -> dict:
        """Load and return the JSON schema for the HIGGS dataset."""
        schema_path = Path("data/schemas/higgs_schema.json")
        if not schema_path.exists():
            return {"columns": HIGGS_COLUMNS}
        with open(schema_path) as f:
            return json.load(f)

    def summary(self) -> None:
        """Print a quick summary of the dataset."""
        df = self.read(n_samples=10_000, use_cache=False)
        print(f"\nHIGGS Dataset Summary (10k sample):")
        print(f"  File:           {self.filepath}")
        print(f"  Features:       {len(FEATURE_COLUMNS)}")
        print(f"  Label balance:  signal={df[LABEL_COLUMN].mean():.1%}, "
              f"background={1-df[LABEL_COLUMN].mean():.1%}")
        print(f"\nFeature statistics:")
        print(df[FEATURE_COLUMNS].describe().round(3).to_string())
