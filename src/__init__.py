"""
Particle Physics Event Classifier — top-level package.

This is a production-grade ML platform for classifying particle physics
collision events from CERN data. It implements:

- Module 1: Data ingestion & ETL (src.ingestion)
- Module 2: Physics feature engineering (src.features)
- Module 3a: Deep MLP baseline (src.models.mlp)
- Module 4: Experiment tracking via MLflow (src.experiment_tracking)
- Module 5+: Coming in Phase 2 (evaluation, explainability, API, frontend, monitoring)
"""

__version__ = "0.1.0"
__author__ = "Rayya"
