"""FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Query, Request

from holder_boost.api.sample import build_predict_request
from holder_boost.api.schemas import PredictRequest, PredictResponse
from holder_boost.config import get_settings
from holder_boost.ml.inference import LoadedModel, load_model_artifacts, predict_probs
from holder_boost.storage.client import get_database
from holder_boost.storage.repos import markets as markets_repo

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from motor.motor_asyncio import AsyncIOMotorDatabase


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    model_dir = Path(settings.train.model_dir)
    app.state.loaded_model = load_model_artifacts(model_dir)
    app.state.db = await get_database()
    yield


app = FastAPI(title="holder-boost", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/predict/sample", response_model=PredictRequest)
async def predict_sample(
    request: Request,
    condition_id: str,
    days_left: int = Query(ge=0),
) -> PredictRequest:
    db: AsyncIOMotorDatabase = request.app.state.db
    market = await markets_repo.fetch_market_by_condition_id(db, condition_id)
    if market is None:
        raise HTTPException(status_code=404, detail=f"market not found: {condition_id}")
    try:
        return build_predict_request(market, days_left=days_left)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    loaded: LoadedModel = request.app.state.loaded_model
    holders = {holder.proxy_wallet: holder.outcome_index for holder in body.holders}
    try:
        probs = predict_probs(loaded, days_left=body.days_left, holders=holders)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictResponse(predict_probs=probs)

