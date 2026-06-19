"""Data ingestion package for the Particle Physics Classifier."""

from src.ingestion.higgs_reader import HiggsReader, HIGGS_COLUMNS, FEATURE_COLUMNS, LABEL_COLUMN
from src.ingestion.etl_pipeline import ETLPipeline, ETLConfig, DataSplits
from src.ingestion.data_validator import HiggsValidator, validate_dataframe

__all__ = [
    "HiggsReader",
    "HIGGS_COLUMNS",
    "FEATURE_COLUMNS",
    "LABEL_COLUMN",
    "ETLPipeline",
    "ETLConfig",
    "DataSplits",
    "HiggsValidator",
    "validate_dataframe",
]
