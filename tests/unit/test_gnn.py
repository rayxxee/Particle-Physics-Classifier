"""
Unit tests for the GNN model (EdgeConv / DGCNN-style).

Tests verify:
  - GNNConfig dataclass validation and serialization
  - Input reshaping: 2D flat → 3D (n_events, n_particles, n_features)
  - GNNModel.fit() trains and returns best_val_auc float (with torch_geometric)
  - predict_proba() shape and range
  - Save/load roundtrip

All tests that require torch_geometric use pytest.importorskip("torch_geometric")
so they are skipped gracefully when torch_geometric is not installed.

Run with:
    pytest tests/unit/test_gnn.py -v -p no:typeguard
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from src.models.gnn.config import GNNConfig


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def toy_data_flat():
    """Flat (2D) toy data: (n_events, 28) → reshaped to (n_events, 4, 7)."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, 28)).astype(np.float32)
    # Signal: slightly different distribution
    X[:n // 2] += 0.5
    y = np.zeros(n, dtype=np.float32)
    y[:n // 2] = 1.0
    idx = rng.permutation(n)
    return X[idx], y[idx]


@pytest.fixture
def toy_data_3d():
    """3D toy data: (n_events, n_particles, n_node_features)."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, 4, 7)).astype(np.float32)
    X[:n // 2] += 0.5
    y = np.zeros(n, dtype=np.float32)
    y[:n // 2] = 1.0
    idx = rng.permutation(n)
    return X[idx], y[idx]


@pytest.fixture
def small_gnn_config():
    """Tiny GNN config for fast unit tests."""
    return GNNConfig(
        k_neighbors=3,          # small k for tiny graphs
        n_particles=4,
        n_node_features=7,
        n_edge_conv_layers=2,
        hidden_dim=16,
        mlp_head_dims=[32],
        dropout=0.0,
        epochs=3,
        batch_size=32,
        learning_rate=1e-2,
        early_stopping=False,
        mixed_precision=False,
        seed=42,
    )


# ─── GNNConfig tests ──────────────────────────────────────────────────────────

class TestGNNConfig:

    def test_default_config_valid(self):
        config = GNNConfig()
        assert config.k_neighbors > 0
        assert config.n_particles > 0
        assert config.n_node_features > 0

    def test_to_dict_is_flat(self):
        config = GNNConfig()
        d = config.to_dict()
        for v in d.values():
            assert not isinstance(v, dict)

    def test_config_hash_is_string(self):
        config = GNNConfig()
        h = config.config_hash()
        assert isinstance(h, str) and len(h) >= 8

    def test_different_configs_have_different_hashes(self):
        c1 = GNNConfig(k_neighbors=4)
        c2 = GNNConfig(k_neighbors=8)
        assert c1.config_hash() != c2.config_hash()


# ─── GNNModel input reshape tests (no torch_geometric needed) ─────────────────

class TestGNNModelReshape:

    def test_reshape_flat_to_3d(self, small_gnn_config):
        """_reshape_input converts (n, 28) → (n, 4, 7)."""
        from src.models.gnn.model import GNNModel
        model = GNNModel(small_gnn_config)
        X = np.random.randn(100, 28).astype(np.float32)
        X_3d = model._reshape_input(X)
        assert X_3d.shape == (100, 4, 7), f"Expected (100, 4, 7), got {X_3d.shape}"

    def test_reshape_already_3d_unchanged(self, small_gnn_config):
        """_reshape_input returns 3D input as-is."""
        from src.models.gnn.model import GNNModel
        model = GNNModel(small_gnn_config)
        X = np.random.randn(100, 4, 7).astype(np.float32)
        X_out = model._reshape_input(X)
        assert X_out.shape == (100, 4, 7)

    def test_reshape_with_truncation(self, small_gnn_config):
        """_reshape_input truncates extra features."""
        from src.models.gnn.model import GNNModel
        model = GNNModel(small_gnn_config)
        X = np.random.randn(50, 35).astype(np.float32)  # extra features
        X_out = model._reshape_input(X)
        assert X_out.shape == (50, 4, 7)

    def test_predict_before_fit_raises(self, small_gnn_config):
        """predict_proba() raises if model not fitted."""
        from src.models.gnn.model import GNNModel
        model = GNNModel(small_gnn_config)
        X = np.random.randn(10, 28).astype(np.float32)
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_proba(X)


# ─── GNNModel training tests (requires torch_geometric) ───────────────────────

class TestGNNModelWithPyG:

    def test_fit_returns_float(self, small_gnn_config, toy_data_flat):
        """fit() returns float best_val_auc."""
        pytest.importorskip("torch_geometric")
        from src.models.gnn.model import GNNModel
        X, y = toy_data_flat
        model = GNNModel(small_gnn_config)
        result = model.fit(X[:150], y[:150], X[150:], y[150:])
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_predict_proba_shape(self, small_gnn_config, toy_data_flat):
        """predict_proba() returns 1D array of correct length."""
        pytest.importorskip("torch_geometric")
        from src.models.gnn.model import GNNModel
        X, y = toy_data_flat
        model = GNNModel(small_gnn_config)
        model.fit(X[:150], y[:150], X[150:], y[150:])
        scores = model.predict_proba(X)
        assert scores.shape == (len(X),)

    def test_predict_proba_range(self, small_gnn_config, toy_data_flat):
        """predict_proba() outputs are in [0, 1]."""
        pytest.importorskip("torch_geometric")
        from src.models.gnn.model import GNNModel
        X, y = toy_data_flat
        model = GNNModel(small_gnn_config)
        model.fit(X[:150], y[:150], X[150:], y[150:])
        scores = model.predict_proba(X)
        assert scores.min() >= 0.0 and scores.max() <= 1.0

    def test_fit_with_3d_input(self, small_gnn_config, toy_data_3d):
        """GNNModel accepts 3D input directly."""
        pytest.importorskip("torch_geometric")
        from src.models.gnn.model import GNNModel
        X, y = toy_data_3d
        model = GNNModel(small_gnn_config)
        result = model.fit(X[:150], y[:150], X[150:], y[150:])
        assert isinstance(result, float)

    def test_save_load_roundtrip(self, small_gnn_config, toy_data_flat):
        """Save/load roundtrip preserves predictions."""
        pytest.importorskip("torch_geometric")
        from src.models.gnn.model import GNNModel
        X, y = toy_data_flat
        model = GNNModel(small_gnn_config)
        model.fit(X[:150], y[:150], X[150:], y[150:])
        scores_before = model.predict_proba(X)

        with tempfile.TemporaryDirectory() as tmp:
            model.save(tmp)
            loaded = GNNModel(small_gnn_config)
            loaded.load(tmp)
            scores_after = loaded.predict_proba(X)

        np.testing.assert_allclose(scores_before, scores_after, rtol=1e-5)

    def test_summary_contains_required_keys(self, small_gnn_config, toy_data_flat):
        """summary() includes expected keys after fitting."""
        pytest.importorskip("torch_geometric")
        from src.models.gnn.model import GNNModel
        X, y = toy_data_flat
        model = GNNModel(small_gnn_config)
        model.fit(X[:150], y[:150], X[150:], y[150:])
        s = model.summary()
        for key in ["model_name", "is_fitted", "n_parameters", "config"]:
            assert key in s
