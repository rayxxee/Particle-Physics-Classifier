from fastapi import APIRouter
from src.serving.api.schemas import PredictRequest, PredictResponse, BatchPredictRequest, BatchPredictResponse

router = APIRouter()

@router.post("/", response_model=PredictResponse)
async def predict(request: PredictRequest):
    # Dummy implementation for now
    # In production, this would load the model and call model.predict_proba()
    score = 0.85
    label = "signal" if score >= 0.5 else "background"
    confidence = "high" if abs(score - 0.5) > 0.3 else "low"
    
    return PredictResponse(
        score=score,
        label=label,
        confidence=confidence
    )

@router.post("/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    # Dummy implementation
    predictions = [0.85] * len(request.events)
    latencies = [1.5] * len(request.events)
    
    return BatchPredictResponse(
        predictions=predictions,
        latencies_ms=latencies
    )
