"""FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request

from holder_boost.api.schemas import PredictRequest, PredictResponse
from holder_boost.config import get_settings
from holder_boost.ml.inference import LoadedModel, load_model_artifacts, predict_probs

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    model_dir = Path(settings.train.model_dir)
    app.state.loaded_model = load_model_artifacts(model_dir)
    yield


app = FastAPI(title="holder-boost", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    loaded: LoadedModel = request.app.state.loaded_model
    holders = {holder.proxy_wallet: holder.outcome_index for holder in body.holders}
    try:
        probs = predict_probs(loaded, days_left=body.days_left, holders=holders)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictResponse(predict_probs=probs)
