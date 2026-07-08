"""``train`` command: CatBoost model on resolved market take-prices."""

from __future__ import annotations

import asyncio
from pathlib import Path

from motor.motor_asyncio import AsyncIOMotorDatabase

from holder_boost.config import get_settings
from holder_boost.logging import get_logger
from holder_boost.ml.artifacts import OUTCOME_INDICES, outcome_model_dir, save_artifacts
from holder_boost.ml.trainer import train_model
from holder_boost.storage.repos.markets import fetch_all_markets


async def run_train(db: AsyncIOMotorDatabase, *, skip_tuning: bool = False) -> None:
    """Load markets from Mongo, train CatBoost for both outcomes, save artifacts."""
    log = get_logger("cli.train")
    cfg = get_settings().train
    model_dir = Path(cfg.model_dir)

    markets = await fetch_all_markets(db, neg_risk=cfg.neg_risk)
    log.info(
        "train data loaded",
        markets=len(markets),
        neg_risk=cfg.neg_risk,
    )

    summaries: list[str] = []
    for outcome_index in OUTCOME_INDICES:
        result = await asyncio.to_thread(
            train_model,
            markets,
            cfg,
            outcome_index=outcome_index,
            skip_tuning=skip_tuning,
        )
        outcome_dir = outcome_model_dir(model_dir, outcome_index)
        save_artifacts(
            outcome_dir,
            model=result.model,
            mlb=result.mlb,
            metadata=result.metadata,
            feature_importance_csv=result.feature_importance_csv,
        )

        auc_msg = (
            f" auc={result.metadata.best_cv_auc:.5f}"
            if result.metadata.best_cv_auc is not None
            else ""
        )
        summaries.append(
            f"outcome_{outcome_index}: rows={result.metadata.n_rows} "
            f"features={result.metadata.n_features} tags={len(result.metadata.frequent_tags)}"
            f"{auc_msg}"
        )
        log.info(
            "train outcome complete",
            outcome_index=outcome_index,
            model_dir=str(outcome_dir),
            rows=result.metadata.n_rows,
            features=result.metadata.n_features,
            tags=len(result.metadata.frequent_tags),
            best_cv_auc=result.metadata.best_cv_auc,
        )

    print(f"train: markets={len(markets)} saved={model_dir}")
    for line in summaries:
        print(f"  {line}")
