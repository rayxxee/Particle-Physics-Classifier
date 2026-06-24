import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.lime_explainer import LIMEExplainer, LIME_AVAILABLE
from src.explainability.physics_interpretability import interpret_shap_physics

class MockModel:
    def __init__(self, name="mlp"):
        self.model_name = name
        self._net = None
        self._device = None
        
    def _to_numpy(self, X):
        if isinstance(X, pd.DataFrame):
            return X.values.astype(np.float32)
        return np.asarray(X, dtype=np.float32)

    def predict_proba(self, X):
        X_np = self._to_numpy(X)
        return np.random.uniform(0, 1, X_np.shape[0])

def test_shap_explainer():
    model = MockModel("mlp")
    X_bg = np.random.randn(10, 5)
    
    explainer = SHAPExplainer(model, X_bg)
    assert explainer.model_name == "mlp"
    
    # Test explain
    X_test = np.random.randn(2, 5)
    shap_vals = explainer.explain(X_test)
    
    # KernelExplainer returns array of shape (n_samples, n_features)
    assert shap_vals.shape == (2, 5)

@pytest.mark.skipif(not LIME_AVAILABLE, reason="LIME is not installed")
def test_lime_explainer():
    model = MockModel("mlp")
    X_bg = np.random.randn(50, 5)
    feature_names = [f"f{i}" for i in range(5)]
    
    explainer = LIMEExplainer(model, X_bg, feature_names)
    
    x_test = np.random.randn(5)
    exp = explainer.explain(x_test, num_features=3)
    
    # LIME explanation should have a list of (feature_name, weight)
    assert len(exp.as_list()) == 3

def test_physics_interpretability():
    shap_values = {"m_bb": 0.5, "met": -0.2, "n_jets": 0.1, "jet1_pt": -0.05}
    feature_names = ["m_bb", "met", "n_jets", "jet1_pt"]
    
    explanations = interpret_shap_physics(shap_values, feature_names, top_k=2)
    
    assert len(explanations) == 2
    assert "m_bb" in explanations[0]
    assert "signal-like" in explanations[0]
    assert "met" in explanations[1]
    assert "background-like" in explanations[1]
