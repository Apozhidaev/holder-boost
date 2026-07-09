"""``dataset`` command: build training parquet from poly-house."""

from __future__ import annotations

import asyncio

import httpx

from holder_boost.config import get_settings
from holder_boost.logging import get_logger
from holder_boost.ml.dataset import build_training_frame
from holder_boost.ml.dataset_store import dataset_dir_for_tag, save_training_frame
from holder_boost.poly_house.enrich import fetch_resolved_markets_with_holders


async def run_dataset(
    client: httpx.AsyncClient,
    *,
    max_pages: int | None = None,
) -> None:
    """Fetch markets from poly-house and save training frame to parquet."""
    log = get_logger("cli.dataset")
    cfg = get_settings().train
    out_dir = dataset_dir_for_tag(cfg.dataset_dir, cfg.tag_id)

    markets = await fetch_resolved_markets_with_holders(
        client,
        tag_id=cfg.tag_id,
        neg_risk=cfg.neg_risk,
        max_pages=max_pages,
    )
    log.info(
        "dataset source loaded",
        tag_id=cfg.tag_id,
        markets=len(markets),
        neg_risk=cfg.neg_risk,
    )

    X, y, frequent_wallets = await asyncio.to_thread(
        build_training_frame,
        markets,
        min_wallet_count=cfg.min_wallet_count,
    )
    await asyncio.to_thread(
        save_training_frame,
        out_dir,
        X=X,
        y=y,
        frequent_wallets=frequent_wallets,
        tag_id=cfg.tag_id,
        min_wallet_count=cfg.min_wallet_count,
    )

    log.info(
        "dataset saved",
        dataset_dir=str(out_dir),
        rows=len(X),
        features=len(X.columns),
        wallets=len(frequent_wallets),
    )
    print(
        f"dataset: tag_id={cfg.tag_id} markets={len(markets)} "
        f"rows={len(X)} saved={out_dir}"
    )
