from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class PredictRequest(BaseModel):
    features: List[float] = Field(..., min_length=28, max_length=28)
    model: str = Field(default="mlp")

class PredictResponse(BaseModel):
    score: float
    label: str
    confidence: str

class BatchPredictRequest(BaseModel):
    events: List[List[float]]
    model: str = Field(default="transformer")

class BatchPredictResponse(BaseModel):
    predictions: List[float]
    latencies_ms: List[float]

class ExplainRequest(BaseModel):
    features: List[float] = Field(..., min_length=28, max_length=28)
    model: str = Field(default="mlp")

class ExplainResponse(BaseModel):
    shap_values: Dict[str, float]
    top_features: List[str]
    physics_notes: List[str]
