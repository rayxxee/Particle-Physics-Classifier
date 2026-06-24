import numpy as np
import pandas as pd
import shap
from typing import Any

class SHAPExplainer:
    """Wrapper for SHAP explainers across different model architectures."""
    def __init__(self, model_wrapper: Any, X_background: pd.DataFrame | np.ndarray):
        """
        Args:
            model_wrapper: The BaseModel instance (MLP, BDT, etc.).
            X_background: Background dataset to integrate over.
        """
        self.model_wrapper = model_wrapper
        self.model_name = model_wrapper.model_name
        self.X_background = model_wrapper._to_numpy(X_background)
        self.explainer = self._build_explainer()

    def _build_explainer(self):
        if self.model_name == "bdt":
            # For XGBoost/LightGBM
            if hasattr(self.model_wrapper, '_model'):
                return shap.TreeExplainer(self.model_wrapper._model)
        elif self.model_name in ["mlp", "transformer", "normalizing_flow"]:
            if hasattr(self.model_wrapper, '_net') and self.model_wrapper._net is not None:
                import torch
                # Create a wrapper function that takes numpy array and returns numpy array of predictions
                def predict_fn(X):
                    device = self.model_wrapper._device or torch.device('cpu')
                    X_tensor = torch.from_numpy(X).to(device)
                    self.model_wrapper._net.eval()
                    with torch.no_grad():
                        if self.model_name == "normalizing_flow":
                            scores = self.model_wrapper.predict_proba(X)
                            return scores
                        else:
                            scores = self.model_wrapper._net(X_tensor).squeeze(-1)
                            return scores.cpu().numpy()
                return shap.KernelExplainer(predict_fn, shap.sample(self.X_background, 100))
        
        # Fallback to KernelExplainer using predict_proba
        return shap.KernelExplainer(self.model_wrapper.predict_proba, shap.sample(self.X_background, 100))

    def explain(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Get SHAP values for the given features.
        """
        X_np = self.model_wrapper._to_numpy(X)
        shap_values = self.explainer.shap_values(X_np)
        
        # TreeExplainer might return a list for binary classification, Kernel returns array
        if isinstance(shap_values, list):
            return shap_values[1] # Return values for class 1
        return shap_values
