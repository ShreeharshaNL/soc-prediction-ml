import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.config import METRICS_PATH, MODEL_PATH, SHAP_PATH


class ModelNotReadyError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_bundle() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise ModelNotReadyError("Model artifact not found. Run backend/train.py first.")
    return joblib.load(MODEL_PATH)


def clear_model_cache() -> None:
    load_bundle.cache_clear()


def predict_soc(features: dict[str, Any]) -> float:
    bundle = load_bundle()
    raw_features: list[str] = bundle["raw_features"]
    frame = pd.DataFrame([{name: features.get(name) for name in raw_features}])
    prediction = bundle["pipeline"].predict(frame)[0]
    return round(float(prediction), 4)


def feature_schema() -> list[dict[str, Any]]:
    bundle = load_bundle()
    return bundle.get("feature_schema", [])


def model_status() -> dict[str, Any]:
    metrics = read_json(METRICS_PATH)
    return {
        "model_ready": MODEL_PATH.exists(),
        "metrics_ready": METRICS_PATH.exists(),
        "shap_ready": SHAP_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "metrics": metrics,
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
