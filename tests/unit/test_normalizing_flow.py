import numpy as np
import pytest
from src.models.normalizing_flow.config import NormalizingFlowConfig
from src.models.normalizing_flow.model import NormalizingFlowModel

def test_normalizing_flow_initialization():
    config = NormalizingFlowConfig(input_dim=10, num_coupling_layers=2)
    model = NormalizingFlowModel(config)
    assert model.model_name == "normalizing_flow"
    assert model.config.input_dim == 10

def test_normalizing_flow_fit_predict(tmp_path):
    config = NormalizingFlowConfig(input_dim=5, num_coupling_layers=2, epochs=2, batch_size=16)
    model = NormalizingFlowModel(config)
    
    # Dummy data
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = np.random.randint(0, 2, 100)
    X_val = np.random.randn(50, 5)
    y_val = np.random.randint(0, 2, 50)
    
    # Test fitting
    loss = model.fit(X_train, y_train, X_val, y_val)
    assert isinstance(loss, float)
    assert model._is_fitted is True
    
    # Test predicting probabilities
    scores = model.predict_proba(X_val)
    assert scores.shape == (50,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    
    # Test save/load
    save_path = tmp_path / "nf_model"
    model.save(save_path)
    
    loaded_model = NormalizingFlowModel()
    loaded_model.load(save_path)
    
    assert loaded_model._is_fitted is True
    assert loaded_model.config.input_dim == 5
    
    loaded_scores = loaded_model.predict_proba(X_val)
    np.testing.assert_allclose(scores, loaded_scores, rtol=1e-5)
