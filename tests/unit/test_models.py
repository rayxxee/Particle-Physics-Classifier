"""
Unit tests for the MLP model architecture and BaseModel interface.

Tests verify:
- Forward pass produces correct output shape and value range
- Model can be trained on toy data and produces AUC > 0.5
- Save/load roundtrip preserves predictions exactly
- BaseModel interface contract is upheld

Run with:
    pytest tests/unit/test_models.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.base_model import BaseModel
from src.models.mlp.config import MLPConfig
from src.models.mlp.model import DeepMLP, MLPBlock, MLPModel


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def small_config():
    """Tiny MLP config for fast unit tests."""
    return MLPConfig(
        input_dim=10,
        hidden_dims=[32, 16],
        dropout_rates=[0.0, 0.0],
        batch_norm=True,
        epochs=3,           # Just enough to verify training loop runs
        batch_size=32,
        learning_rate=1e-3,
        early_stopping=False,
        mixed_precision=False,  # No CUDA in CI
        seed=42,
    )


@pytest.fixture
def toy_data():
    """Generate linearly separable toy data for model tests."""
    rng = np.random.default_rng(42)
    n = 200
    # Signal: features centered at +1
    X_sig = rng.normal(loc=1.0, scale=0.5, size=(n // 2, 10)).astype(np.float32)
    y_sig = np.ones(n // 2, dtype=np.float32)
    # Background: features centered at -1
    X_bkg = rng.normal(loc=-1.0, scale=0.5, size=(n // 2, 10)).astype(np.float32)
    y_bkg = np.zeros(n // 2, dtype=np.float32)

    X = np.vstack([X_sig, X_bkg])
    y = np.concatenate([y_sig, y_bkg])

    # Shuffle
    idx = rng.permutation(n)
    return X[idx], y[idx]


@pytest.fixture
def mlp_model(small_config):
    """MLPModel instance with small config."""
    return MLPModel(small_config)


# ─── DeepMLP architecture tests ───────────────────────────────────────────────

class TestDeepMLP:

    def test_output_shape(self, small_config):
        """Forward pass returns correct output shape (batch_size, 1)."""
        net = DeepMLP(small_config)
        x = torch.randn(64, small_config.input_dim)
        out = net(x)
        assert out.shape == (64, 1), f"Expected (64, 1), got {out.shape}"

    def test_output_range(self, small_config):
        """All outputs are in [0, 1] (Sigmoid activation)."""
        net = DeepMLP(small_config)
        x = torch.randn(1000, small_config.input_dim)
        with torch.no_grad():
            out = net(x)
        assert (out >= 0).all() and (out <= 1).all(), "Outputs must be in [0, 1]"

    def test_n_parameters_positive(self, small_config):
        """Model has positive number of trainable parameters."""
        net = DeepMLP(small_config)
        assert net.n_parameters() > 0

    def test_architecture_matches_config(self, small_config):
        """Network has the correct number of hidden layers."""
        net = DeepMLP(small_config)
        n_hidden = len(small_config.hidden_dims)
        # Count MLPBlock modules
        n_blocks = sum(1 for m in net.modules() if isinstance(m, MLPBlock))
        assert n_blocks == n_hidden, f"Expected {n_hidden} blocks, got {n_blocks}"

    def test_batch_norm_present(self, small_config):
        """BatchNorm layers exist when batch_norm=True."""
        net = DeepMLP(small_config)
        bn_layers = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm1d)]
        assert len(bn_layers) == len(small_config.hidden_dims), (
            f"Expected {len(small_config.hidden_dims)} BN layers, got {len(bn_layers)}"
        )

    def test_no_batch_norm_when_disabled(self, small_config):
        """No BatchNorm layers when batch_norm=False."""
        cfg = MLPConfig(
            input_dim=10,
            hidden_dims=[32, 16],
            dropout_rates=[0.0, 0.0],
            batch_norm=False,
            epochs=1,
            early_stopping=False,
        )
        net = DeepMLP(cfg)
        bn_layers = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm1d)]
        assert len(bn_layers) == 0

    def test_architecture_str_contains_hidden_dims(self, small_config):
        """architecture_str() includes the hidden layer dimensions."""
        net = DeepMLP(small_config)
        arch_str = net.architecture_str()
        for dim in small_config.hidden_dims:
            assert str(dim) in arch_str

    def test_gradient_flows(self, small_config):
        """Gradients flow back to input during backward pass."""
        net = DeepMLP(small_config)
        x = torch.randn(16, small_config.input_dim, requires_grad=True)
        out = net(x)
        loss = out.mean()
        loss.backward()
        assert x.grad is not None, "Gradients did not flow to input"


# ─── MLPBlock tests ───────────────────────────────────────────────────────────

class TestMLPBlock:

    def test_output_shape(self):
        """MLPBlock output shape is (batch, out_features)."""
        block = MLPBlock(in_features=64, out_features=32)
        x = torch.randn(16, 64)
        out = block(x)
        assert out.shape == (16, 32)

    def test_dropout_in_training_mode(self):
        """Dropout has effect in training mode (stochastic outputs)."""
        block = MLPBlock(in_features=64, out_features=64, dropout=0.5, batch_norm=False)
        block.train()
        x = torch.ones(100, 64)
        torch.manual_seed(0)
        out1 = block(x)
        torch.manual_seed(1)
        out2 = block(x)
        # With 50% dropout, outputs should differ between seeds
        assert not torch.allclose(out1, out2), "Dropout should produce different outputs"

    def test_dropout_disabled_in_eval_mode(self):
        """Dropout is disabled in eval mode (deterministic outputs)."""
        block = MLPBlock(in_features=64, out_features=64, dropout=0.5, batch_norm=False)
        block.eval()
        x = torch.randn(100, 64)
        with torch.no_grad():
            out1 = block(x)
            out2 = block(x)
        assert torch.allclose(out1, out2), "Dropout should be disabled in eval mode"


# ─── MLPModel (BaseModel interface) tests ─────────────────────────────────────

class TestMLPModel:

    def test_predict_proba_output_range(self, mlp_model, toy_data):
        """predict_proba() returns values in [0, 1]."""
        X, y = toy_data
        mlp_model.fit(X, y, X[:50], y[:50])
        scores = mlp_model.predict_proba(X)
        assert scores.min() >= 0.0 and scores.max() <= 1.0

    def test_predict_proba_shape(self, mlp_model, toy_data):
        """predict_proba() returns 1D array of length n_events."""
        X, y = toy_data
        mlp_model.fit(X, y, X[:50], y[:50])
        scores = mlp_model.predict_proba(X)
        assert scores.shape == (len(X),), f"Expected ({len(X)},), got {scores.shape}"

    def test_predict_binary(self, mlp_model, toy_data):
        """predict() returns binary labels in {0, 1}."""
        X, y = toy_data
        mlp_model.fit(X, y, X[:50], y[:50])
        labels = mlp_model.predict(X)
        assert set(labels.tolist()).issubset({0, 1})

    def test_predict_requires_fit(self, small_config):
        """predict_proba() raises if model not fitted."""
        model = MLPModel(small_config)
        X = np.random.randn(10, 10).astype(np.float32)
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_proba(X)

    def test_training_improves_auc(self, mlp_model, toy_data):
        """Model AUC exceeds 0.5 on linearly separable data after training."""
        from sklearn.metrics import roc_auc_score
        X, y = toy_data
        n_train = int(0.8 * len(X))
        mlp_model.fit(X[:n_train], y[:n_train], X[n_train:], y[n_train:])
        scores = mlp_model.predict_proba(X[n_train:])
        auc = roc_auc_score(y[n_train:], scores)
        assert auc > 0.5, f"AUC {auc:.4f} should be > 0.5 on separable data"

    def test_save_load_roundtrip(self, mlp_model, toy_data):
        """Save and load preserves predictions exactly."""
        X, y = toy_data
        mlp_model.fit(X, y, X[:50], y[:50])
        scores_before = mlp_model.predict_proba(X)

        with tempfile.TemporaryDirectory() as tmp:
            mlp_model.save(tmp)

            loaded_model = MLPModel()
            loaded_model.load(tmp)
            scores_after = loaded_model.predict_proba(X)

        np.testing.assert_allclose(
            scores_before, scores_after, rtol=1e-5,
            err_msg="Predictions differ after save/load roundtrip",
        )

    def test_summary_contains_required_keys(self, mlp_model, toy_data):
        """summary() includes expected metadata keys."""
        X, y = toy_data
        mlp_model.fit(X, y, X[:50], y[:50])
        summary = mlp_model.summary()
        required_keys = ["model_name", "version", "is_fitted", "n_parameters"]
        for key in required_keys:
            assert key in summary, f"Missing key in summary: {key}"

    def test_accepts_dataframe_input(self, mlp_model, toy_data):
        """Model accepts pandas DataFrame as input."""
        import pandas as pd
        X, y = toy_data
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)
        mlp_model.fit(X_df, y_series, X_df.iloc[:50], y_series.iloc[:50])
        scores = mlp_model.predict_proba(X_df)
        assert scores.shape == (len(X_df),)


# ─── MLPConfig tests ──────────────────────────────────────────────────────────

class TestMLPConfig:

    def test_default_config_valid(self):
        """Default MLPConfig is valid (passes post-init check)."""
        config = MLPConfig()  # Should not raise
        assert len(config.hidden_dims) == len(config.dropout_rates)

    def test_mismatched_dims_raises(self):
        """MLPConfig raises if hidden_dims and dropout_rates lengths don't match."""
        with pytest.raises(ValueError, match="dropout_rates length"):
            MLPConfig(
                hidden_dims=[512, 256],
                dropout_rates=[0.3],  # wrong length
            )

    def test_to_dict_is_flat(self):
        """to_dict() returns a flat dict (no nested dicts)."""
        config = MLPConfig()
        d = config.to_dict()
        for key, value in d.items():
            assert not isinstance(value, dict), f"Nested dict found at key: {key}"

    def test_from_yaml_with_defaults(self, tmp_path):
        """MLPConfig.from_yaml() loads config without errors using a minimal YAML."""
        yaml_content = """
architecture:
  input_dim: 28
  hidden_dims: [512, 256]
  dropout_rates: [0.3, 0.0]
  batch_norm: true
training:
  epochs: 50
  batch_size: 1024
  learning_rate: 0.001
  weight_decay: 0.0001
  gradient_clip_val: 1.0
  mixed_precision: false
  class_weights: balanced
  seed: 42
  scheduler:
    name: cosine_annealing
    T_max: 50
    eta_min: 0.000001
  early_stopping:
    enabled: true
    patience: 10
    monitor: val_auc
    min_delta: 0.0001
"""
        yaml_file = tmp_path / "test_mlp.yaml"
        yaml_file.write_text(yaml_content)
        config = MLPConfig.from_yaml(yaml_file)
        assert config.input_dim == 28
        assert config.hidden_dims == [512, 256]
        assert config.epochs == 50
