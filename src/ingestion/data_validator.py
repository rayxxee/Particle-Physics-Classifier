"""
Pandera-based data validator for the HIGGS dataset and ROOT-derived DataFrames.

Validation is a hard gate: bad data raises an exception and never passes
downstream silently. This protects training from silent data corruption.

Usage:
    from src.ingestion.data_validator import HiggsValidator, validate_dataframe

    validator = HiggsValidator()
    validator.validate(df)  # Raises SchemaError if validation fails
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

from src.utils.logging_config import get_logger

log = get_logger(__name__)


# ─── HIGGS Dataset Schema ────────────────────────────────────────────────────

def _build_higgs_schema() -> DataFrameSchema:
    """Build the Pandera schema for the HIGGS dataset."""

    def in_range(low: float, high: float) -> Check:
        return Check(lambda x: (x >= low) & (x <= high), error=f"out of range [{low}, {high}]")

    def no_inf() -> Check:
        return Check(lambda x: ~np.isinf(x), error="contains inf values")

    return DataFrameSchema(
        columns={
            "label": Column(
                int,
                checks=[Check(lambda x: x.isin([0, 1]), error="label must be 0 or 1")],
                nullable=False,
            ),
            # ── Lepton kinematics ─────────────────────────────────────────────
            "lepton_pt": Column(
                float, checks=[in_range(0.0, 2000.0), no_inf()], nullable=False
            ),
            "lepton_eta": Column(
                float, checks=[in_range(-6.0, 6.0), no_inf()], nullable=False
            ),
            "lepton_phi": Column(
                float, checks=[in_range(-4.0, 4.0), no_inf()], nullable=False
            ),
            # ── MET ───────────────────────────────────────────────────────────
            "missing_energy_magnitude": Column(
                float, checks=[in_range(0.0, 5000.0), no_inf()], nullable=False
            ),
            "missing_energy_phi": Column(
                float, checks=[in_range(-4.0, 4.0), no_inf()], nullable=False
            ),
            # ── Jet kinematics + b-tags ───────────────────────────────────────
            **{
                col: Column(float, checks=[in_range(0.0, 5000.0), no_inf()], nullable=False)
                for col in [
                    "jet1_pt", "jet2_pt", "jet3_pt", "jet4_pt",
                ]
            },
            **{
                col: Column(float, checks=[in_range(-6.0, 6.0), no_inf()], nullable=False)
                for col in [
                    "jet1_eta", "jet2_eta", "jet3_eta", "jet4_eta",
                ]
            },
            **{
                col: Column(float, checks=[in_range(-4.0, 4.0), no_inf()], nullable=False)
                for col in [
                    "jet1_phi", "jet2_phi", "jet3_phi", "jet4_phi",
                ]
            },
            **{
                col: Column(float, checks=[in_range(0.0, 10.0), no_inf()], nullable=False)
                for col in [
                    "jet1_b_tag", "jet2_b_tag", "jet3_b_tag", "jet4_b_tag",
                ]
            },
            # ── High-level features (invariant masses) ────────────────────────
            **{
                col: Column(float, checks=[in_range(0.0, 10000.0), no_inf()], nullable=False)
                for col in ["m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"]
            },
        },
        checks=[
            # Dataset-level: not empty
            Check(lambda df: len(df) > 0, error="DataFrame is empty"),
            # Dataset-level: class balance within reasonable bounds (10-90% signal)
            Check(
                lambda df: 0.10 <= df["label"].mean() <= 0.90,
                error="Label balance is extreme (< 10% or > 90% signal). Check data.",
            ),
        ],
        coerce=True,  # Auto-coerce dtypes where possible
        strict=False,  # Allow extra columns (e.g. derived features added later)
    )


# ─── Validator class ─────────────────────────────────────────────────────────

class HiggsValidator:
    """
    Validates HIGGS dataset DataFrames against the expected schema.

    Raises pa.errors.SchemaError on validation failure — never silently passes.

    Args:
        raise_on_failure: If True (default), raises SchemaError on failure.
                          If False, logs the error and returns a report dict.

    Example:
        validator = HiggsValidator()
        validated_df = validator.validate(df)
    """

    def __init__(self, raise_on_failure: bool = True) -> None:
        self.schema = _build_higgs_schema()
        self.raise_on_failure = raise_on_failure

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate a DataFrame against the HIGGS schema.

        Args:
            df: Input DataFrame to validate.

        Returns:
            The validated (and optionally coerced) DataFrame.

        Raises:
            pa.errors.SchemaError: If any validation check fails.
        """
        log.info("Starting data validation", n_rows=len(df), n_cols=len(df.columns))

        # Run pre-validation checks
        self._check_no_nulls(df)
        self._check_no_inf(df)

        try:
            validated = self.schema.validate(df, lazy=True)
            log.info("Data validation passed", n_rows=len(validated))
            return validated

        except pa.errors.SchemaErrors as e:
            n_failures = len(e.failure_cases)
            log.error(
                "Data validation FAILED",
                n_failures=n_failures,
                failure_cases=e.failure_cases.to_dict("records")[:5],  # show first 5
            )
            if self.raise_on_failure:
                raise
            return df  # Return original if not raising

    def _check_no_nulls(self, df: pd.DataFrame) -> None:
        """Check for null values in any column."""
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0]
        if not null_cols.empty:
            msg = f"Found null values in columns: {null_cols.to_dict()}"
            log.error("Null values detected", null_counts=null_cols.to_dict())
            if self.raise_on_failure:
                raise ValueError(msg)

    def _check_no_inf(self, df: pd.DataFrame) -> None:
        """Check for inf values in numeric columns."""
        numeric = df.select_dtypes(include=[np.number])
        inf_mask = np.isinf(numeric)
        inf_cols = inf_mask.sum()[inf_mask.sum() > 0]
        if not inf_cols.empty:
            msg = f"Found inf values in columns: {inf_cols.to_dict()}"
            log.error("Inf values detected", inf_counts=inf_cols.to_dict())
            if self.raise_on_failure:
                raise ValueError(msg)

    def validation_report(self, df: pd.DataFrame) -> dict:
        """
        Return a validation report without raising on failures.

        Returns:
            Dict with keys: passed, n_failures, failure_details, summary_stats.
        """
        report: dict[str, Any] = {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "passed": True,
            "n_failures": 0,
            "failure_details": [],
            "summary": {},
        }

        try:
            self.schema.validate(df, lazy=True)
            report["summary"]["null_count"] = int(df.isnull().sum().sum())
            report["summary"]["signal_fraction"] = float(df["label"].mean())
        except pa.errors.SchemaErrors as e:
            report["passed"] = False
            report["n_failures"] = len(e.failure_cases)
            report["failure_details"] = e.failure_cases.to_dict("records")[:20]

        return report


def validate_dataframe(df: pd.DataFrame, raise_on_failure: bool = True) -> pd.DataFrame:
    """
    Convenience function: validate a HIGGS DataFrame.

    Args:
        df:                Input DataFrame.
        raise_on_failure:  Whether to raise on failure.

    Returns:
        Validated DataFrame.
    """
    return HiggsValidator(raise_on_failure=raise_on_failure).validate(df)
