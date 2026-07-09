"""FastAPI request/response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictHolder(BaseModel):
    proxy_wallet: str
    outcome_index: Literal[0, 1]


class PredictRequest(BaseModel):
    tag_id: int
    days_left: int = Field(ge=0)
    holders: list[PredictHolder]


class PredictResponse(BaseModel):
    predict_probs: list[float]


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class FeaturesResponse(BaseModel):
    features: list[FeatureImportance]


class TagsResponse(BaseModel):
    tag_ids: list[int]
