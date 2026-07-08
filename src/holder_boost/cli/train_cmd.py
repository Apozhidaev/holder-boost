"""``train`` command: CatBoost model on holder distribution rows."""

from __future__ import annotations

import asyncio
from pathlib import Path

from motor.motor_asyncio import AsyncIOMotorDatabase

from holder_boost.config import get_settings
from holder_boost.logging import get_logger
from holder_boost.ml.artifacts import save_artifacts
from holder_boost.ml.trainer import train_model
from holder_boost.storage.repos.markets import fetch_all_markets


async def run_train(db: AsyncIOMotorDatabase, *, skip_tuning: bool = False) -> None:
    """Load markets from Mongo, train CatBoost on holder rows, save artifacts."""
    log = get_logger("cli.train")
    cfg = get_settings().train
    model_dir = Path(cfg.model_dir)

    markets = await fetch_all_markets(db, neg_risk=cfg.neg_risk)
    log.info(
        "train data loaded",
        markets=len(markets),
        neg_risk=cfg.neg_risk,
    )

    result = await asyncio.to_thread(
        train_model,
        markets,
        cfg,
        skip_tuning=skip_tuning,
    )
    save_artifacts(
        model_dir,
        model=result.model,
        frequent_wallets=result.frequent_wallets,
        metadata=result.metadata,
        feature_importance_csv=result.feature_importance_csv,
    )

    auc_msg = (
        f" auc={result.metadata.best_cv_auc:.5f}"
        if result.metadata.best_cv_auc is not None
        else ""
    )
    summary = (
        f"rows={result.metadata.n_rows} "
        f"features={result.metadata.n_features} "
        f"wallets={len(result.metadata.frequent_wallets)}"
        f"{auc_msg}"
    )
    log.info(
        "train complete",
        model_dir=str(model_dir),
        rows=result.metadata.n_rows,
        features=result.metadata.n_features,
        wallets=len(result.metadata.frequent_wallets),
        best_cv_auc=result.metadata.best_cv_auc,
    )

    print(f"train: markets={len(markets)} saved={model_dir}")
    print(f"  {summary}")
