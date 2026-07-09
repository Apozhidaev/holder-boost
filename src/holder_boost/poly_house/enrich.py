"""Load resolved markets with holder-distribution from poly-house."""

from __future__ import annotations

import httpx

from holder_boost.http import HTTPError
from holder_boost.logging import get_logger
from holder_boost.models.markets import ResolvedMarket
from holder_boost.poly_house.activity import fetch_market_holder_distribution
from holder_boost.poly_house.markets import fetch_resolved_markets

log = get_logger("poly_house.enrich")


async def fetch_resolved_markets_with_holders(
    client: httpx.AsyncClient,
    *,
    tag_id: int,
    neg_risk: bool,
    max_pages: int | None = None,
) -> list[ResolvedMarket]:
    """Fetch closed markets for *tag_id* and enrich each with holder distribution."""
    markets = await fetch_resolved_markets(
        client,
        tag_ids=[tag_id],
        neg_risk=neg_risk,
        max_pages=max_pages,
    )

    enriched: list[ResolvedMarket] = []
    enrich_failed = 0
    for market in markets:
        try:
            holders, max_days_left = await fetch_market_holder_distribution(
                client,
                condition_id=market.condition_id,
                tag_id=tag_id
            )
        except HTTPError as exc:
            enrich_failed += 1
            log.warning(
                "skipping market: holder-distribution fetch failed",
                market_id=market.id,
                error=str(exc),
            )
            continue
        enriched.append(
            market.model_copy(
                update={
                    "holders": holders,
                    "max_days_left": max_days_left,
                }
            )
        )

    log.info(
        "resolved markets enriched",
        tag_id=tag_id,
        fetched=len(markets),
        enriched=len(enriched),
        enrich_failed=enrich_failed,
    )
    return enriched
