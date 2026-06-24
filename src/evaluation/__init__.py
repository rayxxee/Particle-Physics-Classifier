"""
Evaluation pipeline for particle physics classifier models.

Computes standard HEP and ML evaluation metrics:
  - ROC AUC
  - Precision-Recall AUC (Average Precision)
  - Threshold sweep (accuracy, F1, precision, recall)
  - Calibration curve
  - Punzi Figure of Merit (FOM)

Usage:
    from src.evaluation.evaluation_pipeline import EvaluationPipeline, EvaluationConfig

    cfg = EvaluationConfig(output_dir="eval_output", significance_target=5.0)
    pipeline = EvaluationPipeline(cfg)
    result = pipeline.evaluate(model, X_test, y_test, run_name="mlp_v1")
"""

from src.evaluation.evaluation_pipeline import EvaluationConfig, EvaluationPipeline, EvaluationResult

__all__ = ["EvaluationConfig", "EvaluationPipeline", "EvaluationResult"]
