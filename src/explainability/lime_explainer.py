import numpy as np
import pandas as pd
try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

from typing import Any, List

class LIMEExplainer:
    """Wrapper for LIME explanations."""
    def __init__(self, model_wrapper: Any, X_train: pd.DataFrame | np.ndarray, feature_names: List[str]):
        """
        Args:
            model_wrapper: The BaseModel instance.
            X_train: Training data to fit the LIME explainer.
            feature_names: List of feature names.
        """
        if not LIME_AVAILABLE:
            raise ImportError("Please install lime via `pip install lime` to use LIMEExplainer.")
            
        self.model_wrapper = model_wrapper
        X_train_np = model_wrapper._to_numpy(X_train)
        
        self.explainer = LimeTabularExplainer(
            X_train_np,
            feature_names=feature_names,
            class_names=["background", "signal"],
            mode="classification",
            discretize_continuous=True
        )

    def _predict_fn(self, X: np.ndarray) -> np.ndarray:
        """
        LIME requires a prediction function that returns probabilities for all classes.
        Since we return P(signal), we return [1 - P(signal), P(signal)].
        """
        scores = self.model_wrapper.predict_proba(X)
        return np.vstack([1.0 - scores, scores]).T

    def explain(self, x: pd.Series | np.ndarray, num_features: int = 10):
        """
        Explain a single instance.
        """
        if isinstance(x, pd.Series):
            x_np = x.values
        else:
            x_np = np.asarray(x).flatten()
            
        exp = self.explainer.explain_instance(
            x_np,
            self._predict_fn,
            num_features=num_features
        )
        return exp
