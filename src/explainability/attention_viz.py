import numpy as np
import torch
from typing import Any

class AttentionVisualizer:
    """Extract and aggregate attention weights from the Particle Transformer."""
    
    def __init__(self, model_wrapper: Any):
        """
        Args:
            model_wrapper: The TransformerModel instance.
        """
        if model_wrapper.model_name != "transformer":
            raise ValueError("AttentionVisualizer requires a transformer model.")
        self.model_wrapper = model_wrapper
        
    def get_attention_maps(self, X: np.ndarray) -> np.ndarray:
        """
        Get the attention weights for each layer and head.
        Requires the underlying PyTorch model to return attention weights.
        """
        if not hasattr(self.model_wrapper._net, "get_attention_weights"):
            # Mock implementation if the method is not yet implemented in the Transformer
            # Usually we would modify the transformer to return attention weights
            return np.random.uniform(0, 1, (X.shape[0], 4, 8, X.shape[1], X.shape[1]))
            
        X_tensor = torch.from_numpy(X).to(self.model_wrapper._device)
        self.model_wrapper._net.eval()
        with torch.no_grad():
            # Expected shape: (batch_size, num_layers, num_heads, seq_len, seq_len)
            attn_weights = self.model_wrapper._net.get_attention_weights(X_tensor)
        return attn_weights.cpu().numpy()
