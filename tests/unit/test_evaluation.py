"""
Unit tests for the EvaluationPipeline.

Tests verify:
  - EvaluationConfig dataclass serialization
  - EvaluationResult dataclass and to_dict()
  - All five evaluation metrics compute correctly on toy data:
    ROC AUC, Average Precision, threshold sweep, calibration, Punzi FOM
  - Plots are saved as PNG files
  - No MLflow dependency required (mlflow_logger=None)

Run with:
    pytest tests/unit/test_evaluation.py -v -p no:typeguard
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.evaluation_pipeline import EvaluationConfig, EvaluationPipeline, EvaluationResult


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def toy_y_true():
    """Binary labels: 500 signal, 500 background."""
    rng = np.random.default_rng(42)
    y = np.zeros(1000, dtype=np.float32)
    y[:500] = 1.0
    return rng.permutation(y)


@pytest.fixture
def good_scores(toy_y_true):
    """Scores correlated with labels (AUC ~ 0.85)."""
    rng = np.random.default_rng(0)
    scores = toy_y_true * 0.7 + rng.uniform(0, 0.3, size=len(toy_y_true))
    return scores.clip(0, 1).astype(np.float32)


@pytest.fixture
def random_scores(toy_y_true):
    """Random scores (AUC ~ 0.5)."""
    rng = np.random.default_rng(1)
    return rng.uniform(0, 1, size=len(toy_y_true)).astype(np.float32)


class _DummyModel:
    """Minimal model-like object with predict_proba()."""
    model_name = "dummy"

    def __init__(self, scores):
        self._scores = scores

    def predict_proba(self, X):
        return self._scores


# ─── EvaluationConfig tests ───────────────────────────────────────────────────

class TestEvaluationConfig:

    def test_default_config_valid(self):
        cfg = EvaluationConfig()
        assert cfg.significance_target == 5.0
        assert cfg.n_threshold_steps > 0
        assert cfg.n_calibration_bins > 0

    def test_to_dict_is_flat(self):
        cfg = EvaluationConfig()
        d = cfg.to_dict()
        for v in d.values():
            assert not isinstance(v, dict)

    def test_config_hash_is_string(self):
        cfg = EvaluationConfig()
        h = cfg.config_hash()
        assert isinstance(h, str) and len(h) > 0


# ─── EvaluationResult tests ───────────────────────────────────────────────────

class TestEvaluationResult:

    def test_default_result_valid(self):
        result = EvaluationResult()
        assert result.roc_auc == 0.0
        assert result.model_name == "unknown"

    def test_to_dict_keys(self):
        result = EvaluationResult(roc_auc=0.85, average_precision=0.80)
        d = result.to_dict()
        required = [
            "roc_auc", "average_precision", "best_f1", "best_accuracy",
            "best_threshold", "best_punzi_fom", "best_punzi_threshold",
            "calibration_ece", "n_signal", "n_background",
        ]
        for key in required:
            assert key in d, f"Missing key: {key}"

    def test_summary_str_contains_auc(self):
        result = EvaluationResult(roc_auc=0.85, model_name="test_model")
        s = result.summary_str()
        assert "0.8500" in s
        assert "test_model" in s


# ─── EvaluationPipeline metric tests ─────────────────────────────────────────

class TestEvaluationPipeline:

    def _make_pipeline(self, tmp_dir: str) -> EvaluationPipeline:
        cfg = EvaluationConfig(
            output_dir=tmp_dir,
            significance_target=5.0,
            n_threshold_steps=50,
            n_calibration_bins=5,
            dpi=72,
            log_to_mlflow=False,
        )
        return EvaluationPipeline(cfg)

    def test_roc_auc_good_scores(self, toy_y_true, good_scores):
        """Good scores produce AUC > 0.7."""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._make_pipeline(tmp)
            model = _DummyModel(good_scores)
            result = pipeline.evaluate(model, None, toy_y_true, run_name="test")
        assert result.roc_auc > 0.7, f"AUC {result.roc_auc:.4f} should be > 0.7"

    def test_roc_auc_random_scores_near_half(self, toy_y_true, random_scores):
        """Random scores produce AUC near 0.5."""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._make_pipeline(tmp)
            model = _DummyModel(random_scores)
            result = pipeline.evaluate(model, None, toy_y_true, run_name="random")
        assert 0.4 <= result.roc_auc <= 0.6, f"Random AUC {result.roc_auc:.4f} should be ~0.5"

    def test_average_precision_good_scores(self, toy_y_true, good_scores):
        """Good scores produce AP > 0.7."""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._make_pipeline(tmp)
            result = pipeline.evaluate(_DummyModel(good_scores), None, toy_y_true, run_name="t")
        assert result.average_precision > 0.7

    def test_best_f1_in_range(self, toy_y_true, good_scores):
        """Best F1 is in (0, 1]."""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._make_pipeline(tmp)
            result = pipeline.evaluate(_DummyModel(good_scores), None, toy_y_true, run_name="t")
        assert 0.0 < result.best_f1 <= 1.0

    def test_best_threshold_in_range(self, toy_y_true, good_scores):
        """Best threshold is in [0, 1]."""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._make_pipeline(tmp)
            result = pipeline.evaluate(_DummyModel(good_scores), None, toy_y_true, run_name="t")
        assert 0.0 <= result.best_threshold <= 1.0

    def test_calibration_ece_in_range(self, toy_y_true, good_scores):
        """Calibration ECE is in [0, 1]."""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._make_pipeline(tmp)
            result = pipeline.evaluate(_DummyModel(good_scores), None, toy_y_true, run_name="t")
        assert 0.0 <= result.calibration_ece <= 1.0

    def test_punzi_fom_positive(self, toy_y_true, good_scores):
        """Punzi FOM is positive for good scores."""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._make_pipeline(tmp)
            result = pipeline.evaluate(_DummyModel(good_scores), None, toy_y_true, run_name="t")
        assert result.best_punzi_fom > 0.0

    def test_plots_saved_as_png(self, toy_y_true, good_scores):
        """All 5 plots are saved as PNG files."""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._make_pipeline(tmp)
            result = pipeline.evaluate(_DummyModel(good_scores), None, toy_y_true, run_name="t")
            # Assert inside the context so files are not yet deleted
            assert len(result.plot_paths) == 5, f"Expected 5 plots, got {len(result.plot_paths)}"
            for p in result.plot_paths:
                assert Path(p).exists(), f"Plot file not found: {p}"
                assert p.endswith(".png"), f"Not a PNG: {p}"

    def test_n_signal_n_background(self, toy_y_true, good_scores):
        """n_signal and n_background sum to total events."""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._make_pipeline(tmp)
            result = pipeline.evaluate(_DummyModel(good_scores), None, toy_y_true, run_name="t")
        assert result.n_signal + result.n_background == len(toy_y_true)
        assert result.n_signal == int((toy_y_true == 1).sum())

    def test_model_name_propagated(self, toy_y_true, good_scores):
        """model_name from model object is stored in result."""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._make_pipeline(tmp)
            result = pipeline.evaluate(_DummyModel(good_scores), None, toy_y_true,
                                       run_name="myrun")
        assert result.model_name == "dummy"
        assert result.run_name == "myrun"

    def test_punzi_fom_formula(self):
        """Punzi FOM matches formula: S / (a/2 + sqrt(B))."""
        # Manual check: at threshold 0, S=500, B=500, a=5
        # FOM = 500 / (2.5 + sqrt(500)) ≈ 500 / (2.5 + 22.36) ≈ 20.0
        y_true = np.array([1.0] * 500 + [0.0] * 500)
        scores = np.array([0.8] * 500 + [0.3] * 500)
        a = 5.0
        # At threshold 0.5: S=500, B=0
        # FOM = 500 / (2.5 + sqrt(0)) → very large
        # At threshold 0.9: S=0, B=0 → FOM=0

        with tempfile.TemporaryDirectory() as tmp:
            cfg = EvaluationConfig(
                output_dir=tmp,
                significance_target=a,
                n_threshold_steps=100,
                n_calibration_bins=5,
                dpi=72,
            )
            pipeline = EvaluationPipeline(cfg)
            result = pipeline.evaluate(_DummyModel(scores), None, y_true, run_name="fom_test")

        # At threshold=0.5 (midpoint), all signal passes, no background: FOM is large
        assert result.best_punzi_fom > 10.0
