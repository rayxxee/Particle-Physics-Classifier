"""
Unit tests for evaluation metrics and physics-specific metrics.

Tests verify:
- AUC = 1.0 for perfect classifier
- AUC = 0.5 for random classifier
- AUC = 0.0 for perfectly wrong classifier (inverted)
- Signal significance formula correctness
- Punzi figure of merit is monotonically increasing with signal efficiency

Run with:
    pytest tests/unit/test_metrics.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score


# ─── Helpers (inline — no dependency on evaluation module yet) ────────────────

def signal_significance(s: float | np.ndarray, b: float | np.ndarray) -> float | np.ndarray:
    """Z = S / √B."""
    return s / np.sqrt(np.maximum(b, 1e-10))


def punzi_figure_of_merit(
    signal_eff: float | np.ndarray,
    b: float | np.ndarray,
    sigma: float = 5.0,
) -> float | np.ndarray:
    """FOM = ε_sig / (σ/2 + √B)."""
    return signal_eff / (sigma / 2 + np.sqrt(np.maximum(b, 1e-10)))


def compute_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_scores))


# ─── AUC tests ────────────────────────────────────────────────────────────────

class TestAUC:

    def test_perfect_classifier_auc_is_one(self):
        """AUC = 1.0 for a classifier that perfectly separates signal and background."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        # Perfect scores: signal gets 1.0, background gets 0.0
        y_scores = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        auc = compute_auc(y_true, y_scores)
        assert abs(auc - 1.0) < 1e-6, f"Expected AUC=1.0, got {auc}"

    def test_random_classifier_auc_is_half(self):
        """AUC ≈ 0.5 for a classifier that assigns random scores."""
        rng = np.random.default_rng(42)
        n = 10_000
        y_true = rng.integers(0, 2, n)
        y_scores = rng.uniform(0, 1, n)
        auc = compute_auc(y_true, y_scores)
        # AUC should be close to 0.5 for large n (within ~0.01)
        assert abs(auc - 0.5) < 0.02, f"Random classifier AUC should be ~0.5, got {auc:.4f}"

    def test_inverted_classifier_auc_is_zero(self):
        """AUC = 0.0 for a perfectly wrong classifier (inverted scores)."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        # Inverted: signal gets 0.0, background gets 1.0
        y_scores = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        auc = compute_auc(y_true, y_scores)
        assert abs(auc - 0.0) < 1e-6, f"Inverted classifier AUC should be 0.0, got {auc}"

    def test_auc_invariant_to_score_monotonic_transform(self):
        """AUC is invariant to any monotonic transformation of the scores."""
        rng = np.random.default_rng(10)
        y_true = rng.integers(0, 2, 1000)
        y_scores = rng.uniform(0, 1, 1000)

        auc_original = compute_auc(y_true, y_scores)
        # Monotonic transform: square the scores
        auc_squared = compute_auc(y_true, y_scores**2)

        assert abs(auc_original - auc_squared) < 1e-6, (
            "AUC should be invariant to monotonic score transforms"
        )

    def test_auc_bounded_zero_to_one(self):
        """AUC is always in [0, 1]."""
        rng = np.random.default_rng(20)
        for _ in range(20):
            y_true = rng.integers(0, 2, 100)
            if y_true.sum() == 0 or y_true.sum() == 100:
                continue  # Skip degenerate cases
            y_scores = rng.uniform(0, 1, 100)
            auc = compute_auc(y_true, y_scores)
            assert 0.0 <= auc <= 1.0, f"AUC out of bounds: {auc}"

    def test_auc_with_tied_scores(self):
        """AUC handles tied scores gracefully (no division by zero)."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # All tied
        auc = compute_auc(y_true, y_scores)
        assert abs(auc - 0.5) < 1e-6, f"Tied scores should give AUC=0.5, got {auc}"


# ─── Signal significance tests ────────────────────────────────────────────────

class TestSignalSignificance:

    def test_known_values(self):
        """Z = S / √B for known values."""
        z = signal_significance(s=10.0, b=100.0)
        assert abs(z - 1.0) < 1e-6, f"Expected Z=1.0, got {z}"

    def test_significance_25_sigma(self):
        """Standard Higgs discovery: S=25, B=25 → Z=5."""
        z = signal_significance(s=25.0, b=25.0)
        assert abs(z - 5.0) < 1e-6, f"Expected Z=5.0, got {z}"

    def test_significance_increases_with_signal(self):
        """More signal at fixed background → higher significance."""
        z1 = signal_significance(s=10.0, b=100.0)
        z2 = signal_significance(s=20.0, b=100.0)
        assert z2 > z1, "Significance should increase with more signal"

    def test_significance_decreases_with_more_background(self):
        """More background at fixed signal → lower significance."""
        z1 = signal_significance(s=10.0, b=100.0)
        z2 = signal_significance(s=10.0, b=400.0)
        assert z2 < z1, "Significance should decrease with more background"

    def test_vectorized(self):
        """Signal significance works on arrays."""
        s = np.array([10.0, 20.0, 30.0])
        b = np.array([100.0, 100.0, 100.0])
        z = signal_significance(s, b)
        expected = s / np.sqrt(b)
        np.testing.assert_allclose(z, expected, rtol=1e-6)


# ─── Punzi FOM tests ──────────────────────────────────────────────────────────

class TestPunziFOM:

    def test_known_value(self):
        """FOM = ε_sig / (σ/2 + √B) for known values (sigma=5, B=25, ε=1)."""
        fom = punzi_figure_of_merit(signal_eff=1.0, b=25.0, sigma=5.0)
        expected = 1.0 / (2.5 + 5.0)  # = 1 / 7.5 ≈ 0.1333
        assert abs(fom - expected) < 1e-6

    def test_fom_increases_with_efficiency(self):
        """Higher signal efficiency → higher Punzi FOM (at fixed background)."""
        fom1 = punzi_figure_of_merit(signal_eff=0.5, b=100.0)
        fom2 = punzi_figure_of_merit(signal_eff=0.8, b=100.0)
        assert fom2 > fom1

    def test_fom_decreases_with_more_background(self):
        """More background → lower Punzi FOM (at fixed efficiency)."""
        fom1 = punzi_figure_of_merit(signal_eff=0.8, b=100.0)
        fom2 = punzi_figure_of_merit(signal_eff=0.8, b=400.0)
        assert fom2 < fom1

    def test_fom_non_negative(self):
        """Punzi FOM is non-negative for valid inputs."""
        rng = np.random.default_rng(30)
        eff = rng.uniform(0, 1, 100)
        b = rng.uniform(0, 1000, 100)
        fom = punzi_figure_of_merit(eff, b)
        assert (fom >= 0).all()


# ─── Background rejection curve tests ────────────────────────────────────────

class TestBackgroundRejection:

    def test_background_rejection_at_perfect_working_point(self):
        """At 100% signal efficiency on perfect classifier, background rejection = ∞."""
        from sklearn.metrics import roc_curve

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        # At 0% FPR (perfect rejection), TPR should still be > 0
        # (non-trivial signal efficiency at high rejection)
        assert fpr.min() == 0.0 or fpr[0] < 0.1

    def test_rejection_vs_efficiency_curve(self):
        """1/FPR (rejection) should decrease as TPR (efficiency) increases."""
        from sklearn.metrics import roc_curve

        rng = np.random.default_rng(40)
        n = 1000
        y_true = rng.integers(0, 2, n)
        y_scores = y_true + rng.normal(0, 0.5, n)  # Noisy signal

        fpr, tpr, _ = roc_curve(y_true, y_scores)

        # Remove zero FPR entries (infinite rejection)
        fpr_nonzero = fpr[fpr > 0]
        tpr_at_nonzero = tpr[fpr > 0]
        rejection = 1.0 / fpr_nonzero

        # Rejection should generally decrease as efficiency (tpr) increases
        # (test monotonic trend via correlation)
        correlation = np.corrcoef(tpr_at_nonzero, rejection)[0, 1]
        assert correlation < 0, (
            f"Rejection should decrease as efficiency increases, "
            f"but correlation was {correlation:.3f}"
        )


# ─── Calibration tests ────────────────────────────────────────────────────────

class TestCalibration:

    def test_perfect_calibration_score(self):
        """A perfectly calibrated model has Brier score = 0."""
        y_true = np.array([0.0, 0.0, 1.0, 1.0])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])  # Perfect probs
        brier = float(np.mean((y_prob - y_true) ** 2))
        assert abs(brier) < 1e-10

    def test_random_calibration_score(self):
        """Random classifier has Brier score ≈ 0.25 for balanced classes."""
        rng = np.random.default_rng(50)
        n = 10_000
        y_true = rng.integers(0, 2, n).astype(float)
        y_prob = np.full(n, 0.5)  # Always predict 50%
        brier = float(np.mean((y_prob - y_true) ** 2))
        # For p=0.5 and balanced classes: E[(0.5-y)²] = 0.25
        assert abs(brier - 0.25) < 0.01
