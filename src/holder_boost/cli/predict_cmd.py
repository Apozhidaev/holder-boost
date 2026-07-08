"""``predict`` command: Yes win rate per tag from Mongo ``markets``."""

from __future__ import annotations

from datetime import UTC, datetime

from motor.motor_asyncio import AsyncIOMotorDatabase

from holder_boost.analysis.predict import compute_predict
from holder_boost.config import get_settings
from holder_boost.logging import get_logger
from holder_boost.storage.repos.markets import fetch_markets_by_tag_id
from holder_boost.storage.repos.predicts import upsert_predict
from holder_boost.storage.repos.tags import fetch_all_tags


async def run_predict(db: AsyncIOMotorDatabase) -> None:
    """Compute Yes probability per tag and upsert ``predicts`` collection."""
    log = get_logger("cli.predict")
    cfg = get_settings().predict
    tags = await fetch_all_tags(db)
    computed_at = datetime.now(UTC)
    updated = 0
    skipped = 0

    for tag in tags:
        markets = await fetch_markets_by_tag_id(db, tag.id, neg_risk=cfg.neg_risk)
        row = compute_predict(tag, markets, computed_at=computed_at)
        if row is None:
            skipped += 1
            continue
        await upsert_predict(db, row)
        updated += 1

    log.info(
        "predict complete",
        tags=len(tags),
        updated=updated,
        skipped=skipped,
    )
    print(f"predict: tags={len(tags)} updated={updated} skipped={skipped}")
