from dataclasses import dataclass, field
from typing import List

@dataclass
class NormalizingFlowConfig:
    """Configuration for Normalizing Flow model."""
    input_dim: int = 28
    num_coupling_layers: int = 4
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    epochs: int = 50
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    scheduler_name: str = "cosine"
    seed: int = 42
    
    def to_dict(self):
        return {
            "input_dim": self.input_dim,
            "num_coupling_layers": self.num_coupling_layers,
            "hidden_dims": self.hidden_dims,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "scheduler_name": self.scheduler_name,
            "seed": self.seed,
        }
