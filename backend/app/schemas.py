from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="Raw feature values keyed by training column name")


class PredictionResponse(BaseModel):
    soc_percent: float
    unit: str = "SOC %"
    model: str


class FeatureSpec(BaseModel):
    name: str
    kind: str
    example: Any
    categories: list[str] | None = None
