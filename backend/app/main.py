import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import METRICS_PATH, SHAP_PATH
from app.model_service import ModelNotReadyError, feature_schema, model_status, predict_soc, read_json
from app.schemas import PredictionRequest, PredictionResponse

app = FastAPI(
    title="Soil Organic Carbon XGBoost API",
    description="SOC percentage prediction with XGBoost, Optuna, and SHAP explainability.",
    version="1.0.0",
)

origins = [
    item.strip().rstrip("/")
    for item in os.getenv("CORS_ORIGINS", "*").split(",")
    if item.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Soil Organic Carbon XGBoost API", "docs": "/docs"}


@app.get("/api/status")
def status() -> dict:
    return model_status()


@app.get("/api/features")
def features() -> list[dict]:
    try:
        return feature_schema()
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        value = predict_soc(payload.features)
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictionResponse(soc_percent=value, model="XGBoost Regressor")


@app.get("/api/metrics")
def metrics() -> dict:
    return read_json(METRICS_PATH)


@app.get("/api/artifacts/shap-beeswarm")
def shap_beeswarm() -> FileResponse:
    if not SHAP_PATH.exists():
        raise HTTPException(status_code=404, detail="SHAP beeswarm plot not found. Train the model first.")
    return FileResponse(SHAP_PATH, media_type="image/png", filename="shap_beeswarm.png")
