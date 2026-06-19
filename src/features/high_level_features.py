"""
High-level physics features computed from the HIGGS dataset columns.

The HIGGS dataset already includes 7 "high-level" invariant mass features
(m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb) that were computed by
the original paper's authors. This module:

1. Exposes those features with their physics interpretations.
2. Derives *additional* computed features beyond the dataset's original 28:
   - Transverse mass (lepton + MET)
   - ΔR between lepton and leading jet
   - ΔR between the two leading jets
   - MET significance (MET / √H_T)
   - H_T (scalar sum of jet pTs)
   - Centrality

These derived features can improve model performance and serve as
inputs for the GNN and Transformer models in later phases.

Usage:
    from src.features.high_level_features import HighLevelFeatureBuilder

    builder = HighLevelFeatureBuilder()
    enriched_df = builder.build(df)  # Adds new columns to df
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.low_level_features import LowLevelExtractor
from src.features.physics_features import (
    azimuthal_angle_difference,
    centrality,
    delta_r,
    ht_scalar,
    missing_et_significance,
    transverse_mass,
)
from src.utils.logging_config import get_logger

log = get_logger(__name__)


class HighLevelFeatureBuilder:
    """
    Computes high-level physics features from HIGGS-format DataFrames.

    The original HIGGS dataset provides 7 high-level features (invariant masses).
    This class computes additional kinematic observables that are standard
    in HEP analyses.

    Args:
        include_dataset_hl: Include the original 7 dataset high-level features.
        include_derived:    Compute additional derived kinematic observables.

    Example:
        builder = HighLevelFeatureBuilder(include_derived=True)
        df_enriched = builder.build(df)
        print(df_enriched.columns.tolist())  # Original + derived features
    """

    # Columns this builder adds to the DataFrame
    DERIVED_COLUMNS = [
        "transverse_mass_lv",    # M_T(lepton, MET)
        "delta_r_lepton_jet1",   # ΔR(lepton, jet1)
        "delta_r_jet1_jet2",     # ΔR(jet1, jet2)
        "delta_phi_lepton_met",  # |Δφ|(lepton, MET)
        "ht",                    # Scalar H_T
        "met_significance",      # MET / √H_T
        "centrality",            # H_T / E_total (approximated)
    ]

    def __init__(
        self,
        include_dataset_hl: bool = True,
        include_derived: bool = True,
    ) -> None:
        self.include_dataset_hl = include_dataset_hl
        self.include_derived = include_derived
        self._extractor = LowLevelExtractor()

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute high-level features and return an enriched DataFrame.

        This method adds new columns to a copy of the input DataFrame.
        Existing columns are never modified.

        Args:
            df: Input DataFrame in HIGGS format.

        Returns:
            DataFrame with original columns plus computed high-level features.
        """
        df = df.copy()
        n = len(df)
        log.debug("Building high-level features", n_events=n)

        if self.include_derived:
            df = self._add_derived_features(df)

        log.debug(
            "High-level feature build complete",
            n_events=n,
            n_new_cols=len(self.DERIVED_COLUMNS) if self.include_derived else 0,
        )
        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute and add derived kinematic features."""
        ext = self._extractor

        # ── Transverse mass M_T(lepton, MET) ─────────────────────────────────
        df["transverse_mass_lv"] = transverse_mass(
            lepton_pt=ext.lepton_pt(df),
            lepton_phi=ext.lepton_phi(df),
            met=ext.met_magnitude(df),
            met_phi=ext.met_phi(df),
        )

        # ── ΔR(lepton, jet1) ──────────────────────────────────────────────────
        df["delta_r_lepton_jet1"] = delta_r(
            eta1=ext.lepton_eta(df),
            phi1=ext.lepton_phi(df),
            eta2=ext.jet_eta(df, 1),
            phi2=ext.jet_phi(df, 1),
        )

        # ── ΔR(jet1, jet2) ────────────────────────────────────────────────────
        # Only meaningful where both jets have pT > 0
        dr_j1j2 = delta_r(
            eta1=ext.jet_eta(df, 1),
            phi1=ext.jet_phi(df, 1),
            eta2=ext.jet_eta(df, 2),
            phi2=ext.jet_phi(df, 2),
        )
        # Set to NaN for events with zero-padded jets
        jet2_mask = ext.jet_pt(df, 2) == 0.0
        dr_j1j2 = dr_j1j2.astype(np.float32)
        dr_j1j2[jet2_mask] = np.nan
        df["delta_r_jet1_jet2"] = dr_j1j2

        # ── |Δφ|(lepton, MET) ─────────────────────────────────────────────────
        df["delta_phi_lepton_met"] = azimuthal_angle_difference(
            phi1=ext.lepton_phi(df),
            phi2=ext.met_phi(df),
        )

        # ── H_T (scalar jet pT sum) ───────────────────────────────────────────
        all_jet_pts = ext.all_jet_pts(df)  # shape (n_events, 4)
        ht = ht_scalar(all_jet_pts)
        df["ht"] = ht

        # ── MET significance ──────────────────────────────────────────────────
        df["met_significance"] = missing_et_significance(
            met=ext.met_magnitude(df),
            ht=ht,
        )

        # ── Centrality ────────────────────────────────────────────────────────
        # E_total ≈ H_T + lepton_pT + MET (rough approximation for massless particles)
        e_total = ht + ext.lepton_pt(df) + ext.met_magnitude(df)
        df["centrality"] = centrality(ht=ht, E_total=e_total)

        return df

    def feature_names(self) -> list[str]:
        """Return list of column names produced by this builder."""
        names = []
        if self.include_dataset_hl:
            names += ["m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"]
        if self.include_derived:
            names += self.DERIVED_COLUMNS
        return names

    def feature_descriptions(self) -> dict[str, str]:
        """Return physics description for each derived feature."""
        return {
            "transverse_mass_lv": "Transverse mass of lepton + MET system [GeV]; bounded by M_W for W→lν",
            "delta_r_lepton_jet1": "ΔR angular distance between lepton and leading jet [dimensionless]",
            "delta_r_jet1_jet2": "ΔR angular distance between jet1 and jet2 [dimensionless]",
            "delta_phi_lepton_met": "|Δφ| between lepton and MET vector [rad]; ≈π for W→lν",
            "ht": "Scalar sum of all jet pTs (H_T) [GeV]; measures event activity",
            "met_significance": "MET / √H_T [√GeV]; normalized MET discriminant",
            "centrality": "H_T / E_total [dimensionless]; measures event isotropy",
        }
