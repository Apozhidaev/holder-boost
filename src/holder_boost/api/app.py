"""FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request

from holder_boost.api.schemas import (
    FeatureImportance,
    FeaturesResponse,
    PredictRequest,
    PredictResponse,
)
from holder_boost.config import get_settings
from holder_boost.ml.artifacts import load_feature_importance, model_dir_for_tag
from holder_boost.ml.inference import LoadedModel, load_model_artifacts, predict_probs

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.model_cache = {}
    app.state.models_base_dir = Path(get_settings().train.model_dir)
    yield


def _get_loaded_model(request: Request, tag_id: int) -> LoadedModel:
    cache: dict[int, LoadedModel] = request.app.state.model_cache
    if tag_id in cache:
        return cache[tag_id]

    model_dir = model_dir_for_tag(request.app.state.models_base_dir, tag_id)
    try:
        loaded = load_model_artifacts(model_dir)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"model not found for tag_id={tag_id}",
        ) from exc

    cache[tag_id] = loaded
    return loaded


app = FastAPI(title="holder-boost", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/features/{tag_id}", response_model=FeaturesResponse)
def features(tag_id: int, request: Request) -> FeaturesResponse:
    model_dir = model_dir_for_tag(request.app.state.models_base_dir, tag_id)
    try:
        rows = load_feature_importance(model_dir)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"features not found for tag_id={tag_id}",
        ) from exc
    return FeaturesResponse(
        features=[
            FeatureImportance(feature=feature, importance=importance)
            for feature, importance in rows
        ]
    )


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    loaded = _get_loaded_model(request, body.tag_id)
    holders = {holder.proxy_wallet: holder.outcome_index for holder in body.holders}
    try:
        probs = predict_probs(loaded, days_left=body.days_left, holders=holders)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictResponse(predict_probs=probs)
