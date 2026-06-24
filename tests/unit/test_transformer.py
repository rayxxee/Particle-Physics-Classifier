"""
Unit tests for the Particle Transformer (ParT) model.

Tests verify:
  - TransformerConfig dataclass validation
  - Input reshaping: 2D flat → 3D (n_events, n_particles, n_features)
  - ParticleTransformerNet forward pass shape and value range
  - TransformerModel.fit() returns best_val_auc float
  - predict_proba() shape and range
  - Save/load roundtrip preserves predictions

All torch imports inside test functions (lazy) to prevent pytest
collection hangs on this Windows machine.

Run with:
    pytest tests/unit/test_transformer.py -v -p no:typeguard
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from src.models.transformer.config import TransformerConfig


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def toy_data_flat():
    """Flat (2D) toy data: (n_events, 28) → reshaped to (n_events, 4, 7)."""
    rng = np.random.default_rng(42)
    n = 300
    X = rng.normal(size=(n, 28)).astype(np.float32)
    X[:n // 2] += 0.5
    y = np.zeros(n, dtype=np.float32)
    y[:n // 2] = 1.0
    idx = rng.permutation(n)
    return X[idx], y[idx]


@pytest.fixture
def small_config():
    """Tiny Transformer config for fast unit tests."""
    return TransformerConfig(
        n_particles=4,
        n_features=7,
        d_model=16,
        n_heads=4,
        n_encoder_layers=2,
        dim_feedforward=32,
        dropout=0.0,
        head_dims=[32],
        head_dropout=0.0,
        epochs=3,
        batch_size=64,
        learning_rate=1e-2,
        warmup_epochs=1,
        early_stopping=False,
        mixed_precision=False,
        seed=42,
    )


# ─── TransformerConfig tests ──────────────────────────────────────────────────

class TestTransformerConfig:

    def test_default_config_valid(self):
        config = TransformerConfig()
        assert config.d_model % config.n_heads == 0

    def test_d_model_not_divisible_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            TransformerConfig(d_model=17, n_heads=4)  # 17 % 4 != 0

    def test_to_dict_is_flat(self):
        config = TransformerConfig()
        d = config.to_dict()
        for v in d.values():
            assert not isinstance(v, dict)

    def test_config_hash_is_string(self):
        config = TransformerConfig()
        h = config.config_hash()
        assert isinstance(h, str) and len(h) >= 8

    def test_different_configs_different_hashes(self):
        c1 = TransformerConfig(d_model=64)
        c2 = TransformerConfig(d_model=128, n_heads=8)
        assert c1.config_hash() != c2.config_hash()


# ─── TransformerModel reshape tests (no heavy deps) ───────────────────────────

class TestTransformerModelReshape:

    def test_reshape_2d_to_3d(self, small_config):
        """_reshape_input converts (n, 28) → (n, 4, 7)."""
        from src.models.transformer.model import TransformerModel
        model = TransformerModel(small_config)
        X = np.random.randn(100, 28).astype(np.float32)
        X_3d = model._reshape_input(X)
        assert X_3d.shape == (100, 4, 7)

    def test_reshape_3d_unchanged(self, small_config):
        """_reshape_input returns 3D input unchanged."""
        from src.models.transformer.model import TransformerModel
        model = TransformerModel(small_config)
        X = np.random.randn(100, 4, 7).astype(np.float32)
        X_out = model._reshape_input(X)
        assert X_out.shape == (100, 4, 7)

    def test_predict_before_fit_raises(self, small_config):
        """predict_proba() raises RuntimeError before fitting."""
        from src.models.transformer.model import TransformerModel
        model = TransformerModel(small_config)
        X = np.random.randn(10, 28).astype(np.float32)
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_proba(X)


# ─── ParticleTransformerNet architecture tests ────────────────────────────────

class TestParticleTransformerNet:

    def test_forward_output_shape(self, small_config):
        """Forward pass returns (batch_size, 1) tensor."""
        import torch
        from src.models.transformer.model import ParticleTransformerNet
        net = ParticleTransformerNet.build(small_config)
        x = torch.randn(16, small_config.n_particles, small_config.n_features)
        out = net(x)
        assert out.shape == (16, 1), f"Expected (16, 1), got {out.shape}"

    def test_forward_output_range(self, small_config):
        """Forward pass outputs are in [0, 1] (Sigmoid)."""
        import torch
        from src.models.transformer.model import ParticleTransformerNet
        net = ParticleTransformerNet.build(small_config)
        x = torch.randn(100, small_config.n_particles, small_config.n_features)
        with torch.no_grad():
            out = net(x)
        assert (out >= 0).all() and (out <= 1).all()

    def test_n_parameters_positive(self, small_config):
        """Model has positive number of parameters."""
        from src.models.transformer.model import ParticleTransformerNet
        net = ParticleTransformerNet.build(small_config)
        assert net.n_parameters() > 0

    def test_architecture_str(self, small_config):
        """architecture_str() includes key dimensions."""
        from src.models.transformer.model import ParticleTransformerNet
        net = ParticleTransformerNet.build(small_config)
        arch = net.architecture_str()
        assert str(small_config.d_model) in arch
        assert str(small_config.n_encoder_layers) in arch

    def test_gradient_flows(self, small_config):
        """Gradients flow through the Transformer to input."""
        import torch
        from src.models.transformer.model import ParticleTransformerNet
        net = ParticleTransformerNet.build(small_config)
        x = torch.randn(8, small_config.n_particles, small_config.n_features,
                        requires_grad=True)
        out = net(x)
        loss = out.mean()
        loss.backward()
        assert x.grad is not None


# ─── TransformerModel full pipeline tests ────────────────────────────────────

class TestTransformerModel:

    def test_fit_returns_float(self, small_config, toy_data_flat):
        """fit() returns a float (best_val_auc)."""
        from src.models.transformer.model import TransformerModel
        X, y = toy_data_flat
        model = TransformerModel(small_config)
        result = model.fit(X[:200], y[:200], X[200:], y[200:])
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_predict_proba_shape(self, small_config, toy_data_flat):
        """predict_proba() returns 1D array of correct length."""
        from src.models.transformer.model import TransformerModel
        X, y = toy_data_flat
        model = TransformerModel(small_config)
        model.fit(X[:200], y[:200], X[200:], y[200:])
        scores = model.predict_proba(X)
        assert scores.shape == (len(X),)

    def test_predict_proba_range(self, small_config, toy_data_flat):
        """predict_proba() outputs are in [0, 1]."""
        from src.models.transformer.model import TransformerModel
        X, y = toy_data_flat
        model = TransformerModel(small_config)
        model.fit(X[:200], y[:200], X[200:], y[200:])
        scores = model.predict_proba(X)
        assert scores.min() >= 0.0 and scores.max() <= 1.0

    def test_training_improves_auc(self, small_config, toy_data_flat):
        """Transformer AUC > 0.5 on separable data."""
        from sklearn.metrics import roc_auc_score
        from src.models.transformer.model import TransformerModel
        # Increase epochs slightly for AUC signal on 3 epochs
        config = TransformerConfig(
            n_particles=4, n_features=7,
            d_model=16, n_heads=4, n_encoder_layers=2,
            dim_feedforward=32, dropout=0.0, head_dims=[32], head_dropout=0.0,
            epochs=5, batch_size=64, learning_rate=5e-3,
            warmup_epochs=1, early_stopping=False, mixed_precision=False, seed=0,
        )
        X, y = toy_data_flat
        model = TransformerModel(config)
        model.fit(X[:200], y[:200], X[200:], y[200:])
        scores = model.predict_proba(X[200:])
        auc = roc_auc_score(y[200:], scores)
        assert auc > 0.5, f"AUC {auc:.4f} should be > 0.5"

    def test_save_load_roundtrip(self, small_config, toy_data_flat):
        """Save and load preserves predictions exactly."""
        from src.models.transformer.model import TransformerModel
        X, y = toy_data_flat
        model = TransformerModel(small_config)
        model.fit(X[:200], y[:200], X[200:], y[200:])
        scores_before = model.predict_proba(X)

        with tempfile.TemporaryDirectory() as tmp:
            model.save(tmp)
            loaded = TransformerModel(small_config)
            loaded.load(tmp)
            scores_after = loaded.predict_proba(X)

        np.testing.assert_allclose(
            scores_before, scores_after, rtol=1e-5,
            err_msg="Predictions differ after save/load roundtrip",
        )

    def test_accepts_dataframe(self, small_config, toy_data_flat):
        """TransformerModel accepts pandas DataFrame input."""
        import pandas as pd
        from src.models.transformer.model import TransformerModel
        X, y = toy_data_flat
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        y_s = pd.Series(y)
        model = TransformerModel(small_config)
        model.fit(X_df.iloc[:200], y_s.iloc[:200], X_df.iloc[200:], y_s.iloc[200:])
        scores = model.predict_proba(X_df)
        assert scores.shape == (len(X_df),)

    def test_summary_contains_required_keys(self, small_config, toy_data_flat):
        """summary() returns dict with required keys after fitting."""
        from src.models.transformer.model import TransformerModel
        X, y = toy_data_flat
        model = TransformerModel(small_config)
        model.fit(X[:200], y[:200], X[200:], y[200:])
        s = model.summary()
        for key in ["model_name", "is_fitted", "n_parameters", "config", "architecture"]:
            assert key in s
