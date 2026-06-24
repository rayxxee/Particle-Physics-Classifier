import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from src.serving.api.routes import predict, explain, health

app = FastAPI(
    title="Particle Physics Classifier API",
    description="REST API for model inference and explainability",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

app.include_router(predict.router, prefix="/v1/predict", tags=["Prediction"])
app.include_router(explain.router, prefix="/v1/explain", tags=["Explainability"])
app.include_router(health.router, prefix="/v1/health", tags=["Health"])

def start():
    import uvicorn
    uvicorn.run("src.serving.api.main:app", host="0.0.0.0", port=8000, reload=True)
