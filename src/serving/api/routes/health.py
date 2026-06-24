from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def health_check():
    return {
        "status": "ok",
        "loaded_model": "mlp",
        "uptime_s": 3600
    }
