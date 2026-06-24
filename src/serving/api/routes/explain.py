from fastapi import APIRouter
from src.serving.api.schemas import ExplainRequest, ExplainResponse

router = APIRouter()

@router.post("/", response_model=ExplainResponse)
async def explain(request: ExplainRequest):
    # Dummy implementation
    shap_values = {
        "m_bb": 0.45,
        "met": -0.12,
        "deltaR_lep_jet": 0.08
    }
    top_features = ["m_bb", "met", "deltaR_lep_jet"]
    physics_notes = [
        "**m_bb** pushed the prediction towards signal-like. Invariant mass of the two b-tagged jets.",
        "**met** pushed the prediction towards background-like. Missing transverse energy."
    ]
    
    return ExplainResponse(
        shap_values=shap_values,
        top_features=top_features,
        physics_notes=physics_notes
    )
