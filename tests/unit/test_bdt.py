"""
Unit tests for the BDT model (XGBoost + LightGBM).

Tests verify:
  - BDTConfig dataclass validation and serialization
  - BDTModel.fit() trains and returns best_val_auc float
  - BDTModel.predict_proba() returns correct shape and range
  - Save/load roundtrip preserves predictions
  - XGBoost and LightGBM backends both work

Run with:
    pytest tests/unit/test_bdt.py -v -p no:typeguard
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from src.models.bdt.config import BDTConfig
from src.models.bdt.model import BDTModel


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def toy_data():
    """Linearly separable toy data for fast BDT tests."""
    rng = np.random.default_rng(42)
    n = 400
    X_sig = rng.normal(loc=1.0, scale=0.5, size=(n // 2, 10)).astype(np.float32)
    X_bkg = rng.normal(loc=-1.0, scale=0.5, size=(n // 2, 10)).astype(np.float32)
    X = np.vstack([X_sig, X_bkg])
    y = np.concatenate([np.ones(n // 2), np.zeros(n // 2)]).astype(np.float32)
    idx = rng.permutation(n)
    return X[idx], y[idx]


@pytest.fixture
def small_xgb_config():
    """Tiny XGBoost config for fast unit tests."""
    return BDTConfig(
        model_type="xgboost",
        n_estimators=20,
        max_depth=3,
        learning_rate=0.3,
        early_stopping_rounds=5,
        verbose=0,
    )


@pytest.fixture
def small_lgbm_config():
    """Tiny LightGBM config for fast unit tests."""
    return BDTConfig(
        model_type="lightgbm",
        n_estimators=20,
        max_depth=3,
        learning_rate=0.3,
        lgbm_num_leaves=8,
        early_stopping_rounds=5,
        verbose=0,
    )


# ─── BDTConfig tests ──────────────────────────────────────────────────────────

class TestBDTConfig:

    def test_default_config_is_xgboost(self):
        config = BDTConfig()
        assert config.model_type == "xgboost"

    def test_to_dict_is_flat(self):
        config = BDTConfig()
        d = config.to_dict()
        for v in d.values():
            assert not isinstance(v, dict), "to_dict() must return flat dict"

    def test_config_hash_is_string(self):
        config = BDTConfig()
        h = config.config_hash()
        assert isinstance(h, str) and len(h) > 0

    def test_xgboost_params_cpu(self):
        config = BDTConfig(model_type="xgboost", n_estimators=100)
        params = config.xgboost_params(device="cpu")
        assert params["n_estimators"] == 100
        assert params["objective"] == "binary:logistic"
        assert "device" not in params  # cpu: no device key

    def test_xgboost_params_cuda(self):
        config = BDTConfig(model_type="xgboost")
        params = config.xgboost_params(device="cuda")
        assert params["device"] == "cuda"
        assert params["tree_method"] == "hist"

    def test_lightgbm_params(self):
        config = BDTConfig(model_type="lightgbm", n_estimators=100, lgbm_num_leaves=31)
        params = config.lightgbm_params()
        assert params["n_estimators"] == 100
        assert params["num_leaves"] == 31
        assert params["objective"] == "binary"


# ─── BDTModel tests ───────────────────────────────────────────────────────────

class TestBDTModelXGBoost:

    def test_fit_returns_float(self, small_xgb_config, toy_data):
        """fit() returns a float (best_val_auc), not a dict."""
        pytest.importorskip("xgboost")
        X, y = toy_data
        model = BDTModel(small_xgb_config)
        result = model.fit(X[:300], y[:300], X[300:], y[300:])
        assert isinstance(result, float), f"Expected float, got {type(result)}"
        assert 0.0 <= result <= 1.0

    def test_predict_proba_shape(self, small_xgb_config, toy_data):
        """predict_proba() returns 1D array of correct length."""
        pytest.importorskip("xgboost")
        X, y = toy_data
        model = BDTModel(small_xgb_config)
        model.fit(X[:300], y[:300], X[300:], y[300:])
        scores = model.predict_proba(X)
        assert scores.shape == (len(X),), f"Expected ({len(X)},), got {scores.shape}"

    def test_predict_proba_range(self, small_xgb_config, toy_data):
        """predict_proba() outputs are in [0, 1]."""
        pytest.importorskip("xgboost")
        X, y = toy_data
        model = BDTModel(small_xgb_config)
        model.fit(X[:300], y[:300], X[300:], y[300:])
        scores = model.predict_proba(X)
        assert scores.min() >= 0.0 and scores.max() <= 1.0

    def test_training_improves_auc(self, small_xgb_config, toy_data):
        """XGBoost AUC > 0.5 on linearly separable data."""
        pytest.importorskip("xgboost")
        from sklearn.metrics import roc_auc_score
        X, y = toy_data
        model = BDTModel(small_xgb_config)
        model.fit(X[:300], y[:300], X[300:], y[300:])
        scores = model.predict_proba(X[300:])
        auc = roc_auc_score(y[300:], scores)
        assert auc > 0.5, f"AUC {auc:.4f} should be > 0.5 on separable data"

    def test_save_load_roundtrip(self, small_xgb_config, toy_data):
        """Save/load roundtrip preserves predictions."""
        pytest.importorskip("xgboost")
        X, y = toy_data
        model = BDTModel(small_xgb_config)
        model.fit(X[:300], y[:300], X[300:], y[300:])
        scores_before = model.predict_proba(X)

        with tempfile.TemporaryDirectory() as tmp:
            model.save(tmp)
            loaded = BDTModel()
            loaded.load(tmp)
            scores_after = loaded.predict_proba(X)

        np.testing.assert_allclose(
            scores_before, scores_after, rtol=1e-5,
            err_msg="Predictions differ after save/load",
        )

    def test_predict_before_fit_raises(self, small_xgb_config):
        """predict_proba() raises RuntimeError before fitting."""
        pytest.importorskip("xgboost")
        model = BDTModel(small_xgb_config)
        X = np.random.randn(10, 10).astype(np.float32)
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_proba(X)

    def test_accepts_dataframe(self, small_xgb_config, toy_data):
        """BDTModel accepts pandas DataFrame input."""
        pytest.importorskip("xgboost")
        import pandas as pd
        X, y = toy_data
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        y_s = pd.Series(y)
        model = BDTModel(small_xgb_config)
        model.fit(X_df.iloc[:300], y_s.iloc[:300], X_df.iloc[300:], y_s.iloc[300:])
        scores = model.predict_proba(X_df)
        assert scores.shape == (len(X_df),)

    def test_summary_contains_required_keys(self, small_xgb_config, toy_data):
        """summary() returns dict with required keys."""
        pytest.importorskip("xgboost")
        X, y = toy_data
        model = BDTModel(small_xgb_config)
        model.fit(X[:300], y[:300], X[300:], y[300:])
        s = model.summary()
        for key in ["model_name", "is_fitted", "fit_time_s", "config"]:
            assert key in s, f"Missing key: {key}"


class TestBDTModelLightGBM:

    def test_fit_returns_float(self, small_lgbm_config, toy_data):
        """LightGBM fit() returns float."""
        pytest.importorskip("lightgbm")
        X, y = toy_data
        model = BDTModel(small_lgbm_config)
        result = model.fit(X[:300], y[:300], X[300:], y[300:])
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_predict_proba_shape(self, small_lgbm_config, toy_data):
        """LightGBM predict_proba() returns correct shape."""
        pytest.importorskip("lightgbm")
        X, y = toy_data
        model = BDTModel(small_lgbm_config)
        model.fit(X[:300], y[:300], X[300:], y[300:])
        scores = model.predict_proba(X)
        assert scores.shape == (len(X),)

    def test_training_improves_auc(self, small_lgbm_config, toy_data):
        """LightGBM AUC > 0.5 on separable data."""
        pytest.importorskip("lightgbm")
        from sklearn.metrics import roc_auc_score
        X, y = toy_data
        model = BDTModel(small_lgbm_config)
        model.fit(X[:300], y[:300], X[300:], y[300:])
        scores = model.predict_proba(X[300:])
        auc = roc_auc_score(y[300:], scores)
        assert auc > 0.5

    def test_save_load_roundtrip(self, small_lgbm_config, toy_data):
        """LightGBM save/load roundtrip."""
        pytest.importorskip("lightgbm")
        X, y = toy_data
        model = BDTModel(small_lgbm_config)
        model.fit(X[:300], y[:300], X[300:], y[300:])
        scores_before = model.predict_proba(X)

        with tempfile.TemporaryDirectory() as tmp:
            model.save(tmp)
            loaded = BDTModel(BDTConfig(model_type="lightgbm"))
            loaded.load(tmp)
            scores_after = loaded.predict_proba(X)

        np.testing.assert_allclose(scores_before, scores_after, rtol=1e-5)
