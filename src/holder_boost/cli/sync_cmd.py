"""``sync`` command: load resolved markets from poly-house and insert into Mongo ``markets``."""

from __future__ import annotations

import asyncio

import httpx
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.operations import UpdateOne

from holder_boost.config import get_settings
from holder_boost.http import HTTPError
from holder_boost.logging import get_logger
from holder_boost.poly_house.activity import fetch_market_holder_distribution
from holder_boost.poly_house.catalog import fetch_resolved_markets_and_tags
from holder_boost.poly_house.trades import (
    fetch_market_stop_cent_prices,
    fetch_market_trade_cent_prices,
)
from holder_boost.storage.repos.markets import insert_market_if_absent, market_exists
from holder_boost.storage.repos.tags import insert_tag_if_absent


async def _refresh_tag_markets_counts(db: AsyncIOMotorDatabase) -> None:
    """Set `markets_count` on each tag doc from cached `markets` (distinct markets per tag)."""
    pipeline = [
        {"$match": {"tags": {"$exists": True, "$ne": []}}},
        {"$unwind": "$tags"},
        {"$group": {"_id": "$tags", "markets_count": {"$sum": 1}}},
    ]
    cursor = db.markets.aggregate(pipeline)
    counts: dict[int, int] = {doc["_id"]: doc["markets_count"] async for doc in cursor}

    tag_docs = await db.tags.find({}, projection={"id": 1}).to_list(length=None)
    if not tag_docs:
        return

    ops = [
        UpdateOne(
            {"id": doc["id"]},
            {"$set": {"markets_count": counts.get(doc["id"], 0)}},
        )
        for doc in tag_docs
    ]
    await db.tags.bulk_write(ops)


async def run_sync(
    db: AsyncIOMotorDatabase,
    client: httpx.AsyncClient,
    *,
    max_pages: int | None = None,
) -> None:
    """Fetch resolved markets from poly-house and insert new rows into ``markets``."""
    log = get_logger("cli.sync")
    cfg = get_settings().sync

    markets, tags = await fetch_resolved_markets_and_tags(
        client,
        tag_ids=cfg.tag_ids if cfg.tag_ids else None,
        neg_risk=cfg.neg_risk,
        max_pages=max_pages,
    )

    for tag in tags:
        await insert_tag_if_absent(db, tag)

    inserted = 0
    skipped = 0
    trade_failed = 0
    for market in markets:
        if await market_exists(db, market.id):
            skipped += 1
            continue
        try:
            trade_prices, stop_prices, holder_data = await asyncio.gather(
                fetch_market_trade_cent_prices(
                    client,
                    condition_id=market.condition_id,
                ),
                fetch_market_stop_cent_prices(
                    client,
                    condition_id=market.condition_id,
                    outcome_index=market.resolved_outcome_index,
                ),
                fetch_market_holder_distribution(
                    client,
                    condition_id=market.condition_id,
                ),
            )
            holders, max_days = holder_data
        except HTTPError as exc:
            trade_failed += 1
            log.warning(
                "skipping market: trade/stop/holders fetch failed",
                market_id=market.id,
                error=str(exc),
            )
            continue
        enriched = market.model_copy(
            update={
                "trade_cent_prices": trade_prices,
                "stop_prices": stop_prices,
                "holders": holders,
                "max_days": max_days,
            }
        )
        if await insert_market_if_absent(db, enriched):
            inserted += 1
            log.info(
                "market inserted",
                market_id=market.id,
                trade_yes=len(trade_prices[0]),
                trade_no=len(trade_prices[1]),
                stop_buckets=len(stop_prices),
                holders=len(holders),
                max_days=max_days,
            )

    await _refresh_tag_markets_counts(db)

    log.info(
        "sync complete",
        fetched=len(markets),
        inserted=inserted,
        skipped=skipped,
        trade_failed=trade_failed,
    )
    print(
        f"sync: fetched={len(markets)} inserted={inserted} "
        f"skipped={skipped} trade_failed={trade_failed} resolved markets"
    )
