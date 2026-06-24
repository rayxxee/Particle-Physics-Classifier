"""MLP subpackage for the Particle Physics Classifier."""

from src.models.mlp.config import MLPConfig
from src.models.mlp.model import MLPModel, DeepMLP

__all__ = ["MLPConfig", "MLPModel", "DeepMLP"]
