"""
Raw 4-vector accessor helpers for the HIGGS dataset.

Extracts low-level kinematic variables (pT, η, φ) for leptons, jets, and MET
from a HIGGS-format DataFrame. These are the raw detector measurements before
any derived observables are computed.

Usage:
    from src.features.low_level_features import LowLevelExtractor

    extractor = LowLevelExtractor()
    low_level = extractor.extract(df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─── Column name maps ──────────────────────────────────────────────────────────

LEPTON_COLUMNS = {
    "pt": "lepton_pt",
    "eta": "lepton_eta",
    "phi": "lepton_phi",
}

MET_COLUMNS = {
    "magnitude": "missing_energy_magnitude",
    "phi": "missing_energy_phi",
}

JET_COLUMNS = {
    1: {"pt": "jet1_pt", "eta": "jet1_eta", "phi": "jet1_phi", "btag": "jet1_b_tag"},
    2: {"pt": "jet2_pt", "eta": "jet2_eta", "phi": "jet2_phi", "btag": "jet2_b_tag"},
    3: {"pt": "jet3_pt", "eta": "jet3_eta", "phi": "jet3_phi", "btag": "jet3_b_tag"},
    4: {"pt": "jet4_pt", "eta": "jet4_eta", "phi": "jet4_phi", "btag": "jet4_b_tag"},
}

HIGH_LEVEL_COLUMNS = ["m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"]


class LowLevelExtractor:
    """
    Extracts low-level kinematic variables from a HIGGS-format DataFrame.

    Provides named accessors for lepton, jet, and MET variables, making
    downstream feature engineering code more readable.

    Example:
        extractor = LowLevelExtractor()
        lepton_pt = extractor.lepton_pt(df)
        jet1_pt   = extractor.jet_pt(df, jet_index=1)
        met       = extractor.met(df)
    """

    # ── Lepton ────────────────────────────────────────────────────────────────

    def lepton_pt(self, df: pd.DataFrame) -> np.ndarray:
        return df["lepton_pt"].values.astype(np.float64)

    def lepton_eta(self, df: pd.DataFrame) -> np.ndarray:
        return df["lepton_eta"].values.astype(np.float64)

    def lepton_phi(self, df: pd.DataFrame) -> np.ndarray:
        return df["lepton_phi"].values.astype(np.float64)

    def lepton_4vector(self, df: pd.DataFrame, mass: float = 0.106) -> dict[str, np.ndarray]:
        """
        Return lepton 4-vector components.

        Args:
            mass: Lepton mass in GeV. Default 0.106 = muon mass.
                  Use 0.000511 for electrons.
        """
        return {
            "pt": self.lepton_pt(df),
            "eta": self.lepton_eta(df),
            "phi": self.lepton_phi(df),
            "mass": np.full(len(df), mass),
        }

    # ── MET ───────────────────────────────────────────────────────────────────

    def met_magnitude(self, df: pd.DataFrame) -> np.ndarray:
        return df["missing_energy_magnitude"].values.astype(np.float64)

    def met_phi(self, df: pd.DataFrame) -> np.ndarray:
        return df["missing_energy_phi"].values.astype(np.float64)

    def met(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        return {
            "magnitude": self.met_magnitude(df),
            "phi": self.met_phi(df),
        }

    # ── Jets ──────────────────────────────────────────────────────────────────

    def jet_pt(self, df: pd.DataFrame, jet_index: int) -> np.ndarray:
        return df[JET_COLUMNS[jet_index]["pt"]].values.astype(np.float64)

    def jet_eta(self, df: pd.DataFrame, jet_index: int) -> np.ndarray:
        return df[JET_COLUMNS[jet_index]["eta"]].values.astype(np.float64)

    def jet_phi(self, df: pd.DataFrame, jet_index: int) -> np.ndarray:
        return df[JET_COLUMNS[jet_index]["phi"]].values.astype(np.float64)

    def jet_btag(self, df: pd.DataFrame, jet_index: int) -> np.ndarray:
        return df[JET_COLUMNS[jet_index]["btag"]].values.astype(np.float64)

    def jet_4vector(
        self, df: pd.DataFrame, jet_index: int, mass: float = 0.0
    ) -> dict[str, np.ndarray]:
        """Return jet 4-vector. Jets are approximately massless in simplified models."""
        return {
            "pt": self.jet_pt(df, jet_index),
            "eta": self.jet_eta(df, jet_index),
            "phi": self.jet_phi(df, jet_index),
            "mass": np.full(len(df), mass),
        }

    def all_jet_pts(self, df: pd.DataFrame) -> np.ndarray:
        """Return array of shape (n_events, 4) with all jet pTs."""
        return np.stack(
            [self.jet_pt(df, j) for j in range(1, 5)], axis=1
        )

    def n_jets_above_threshold(self, df: pd.DataFrame, pt_threshold: float = 30.0) -> np.ndarray:
        """
        Count how many jets have pT > threshold.

        Returns:
            Integer array of shape (n_events,), values in {0, 1, 2, 3, 4}.
        """
        jet_pts = self.all_jet_pts(df)
        return (jet_pts > pt_threshold).sum(axis=1).astype(np.int32)

    def has_b_tag(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Return boolean array: True if at least one jet passes the b-tag threshold.
        """
        btag_cols = [JET_COLUMNS[j]["btag"] for j in range(1, 5)]
        return (df[btag_cols] > threshold).any(axis=1).values

    # ── High-level (pass-through) ─────────────────────────────────────────────

    def high_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the 7 high-level invariant mass features from the HIGGS dataset."""
        available = [c for c in HIGH_LEVEL_COLUMNS if c in df.columns]
        return df[available].copy()

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all low-level features as a clean DataFrame.

        Returns a DataFrame with only low-level kinematic columns, suitable
        for models that use raw detector variables only.
        """
        low_level_cols = [
            "lepton_pt", "lepton_eta", "lepton_phi",
            "missing_energy_magnitude", "missing_energy_phi",
            "jet1_pt", "jet1_eta", "jet1_phi", "jet1_b_tag",
            "jet2_pt", "jet2_eta", "jet2_phi", "jet2_b_tag",
            "jet3_pt", "jet3_eta", "jet3_phi", "jet3_b_tag",
            "jet4_pt", "jet4_eta", "jet4_phi", "jet4_b_tag",
        ]
        available = [c for c in low_level_cols if c in df.columns]
        return df[available].copy()
