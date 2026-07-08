"""``train`` command: CatBoost model on holder distribution rows."""

from __future__ import annotations

import asyncio

import httpx

from holder_boost.config import get_settings
from holder_boost.logging import get_logger
from holder_boost.ml.artifacts import model_dir_for_tag, save_artifacts
from holder_boost.ml.trainer import train_model
from holder_boost.poly_house.enrich import fetch_resolved_markets_with_holders


async def run_train(
    client: httpx.AsyncClient,
    *,
    skip_tuning: bool = False,
    max_pages: int | None = None,
) -> None:
    """Fetch markets from poly-house, train CatBoost on holder rows, save artifacts."""
    log = get_logger("cli.train")
    cfg = get_settings().train
    model_dir = model_dir_for_tag(cfg.model_dir, cfg.tag_id)

    markets = await fetch_resolved_markets_with_holders(
        client,
        tag_id=cfg.tag_id,
        neg_risk=cfg.neg_risk,
        max_pages=max_pages,
    )
    log.info(
        "train data loaded",
        tag_id=cfg.tag_id,
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

    print(f"train: tag_id={cfg.tag_id} markets={len(markets)} saved={model_dir}")
    print(f"  {summary}")
