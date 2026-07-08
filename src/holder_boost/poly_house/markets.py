"""poly-house markets API client."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from holder_boost.config import get_settings
from holder_boost.http import get_json_with_body
from holder_boost.logging import get_logger
from holder_boost.models.markets import ResolvedMarket

log = get_logger("poly_house.markets")

_PAGE_LIMIT = 500


def _resolved_market_from_house(raw: dict[str, Any]) -> ResolvedMarket:
    outcome_prices = raw["outcome_prices"]
    return ResolvedMarket(
        id=raw["_id"],
        slug=raw["slug"],
        event_id=raw["event_id"],
        condition_id=raw["condition_id"],
        clob_token_ids=raw["clob_token_ids"],
        neg_risk=raw.get("neg_risk") or False,
        tags=raw["tag_ids"],
        resolved_outcome_index=outcome_prices.index(max(outcome_prices)),
        trade_cent_prices=[[], []],
        stop_prices={},
    )


async def fetch_resolved_markets(
    client: httpx.AsyncClient,
    *,
    tag_ids: Sequence[int] | None = None,
    neg_risk: bool = False,
    max_pages: int | None = None,
) -> list[ResolvedMarket]:
    """Load closed markets from poly-house ``GET /markets``."""
    base = get_settings().poly_house.base_url.rstrip("/")
    url = f"{base}/markets"
    body: list[int] | None = list(tag_ids) if tag_ids else None

    out: list[ResolvedMarket] = []
    offset = 0
    page_no = 0

    while True:
        payload = await get_json_with_body(
            client,
            url,
            params={
                "closed": True,
                "neg_risk": neg_risk,
                "limit": _PAGE_LIMIT,
                "offset": offset,
            },
            json_body=body,
        )
        items = payload["items"]
        if not items:
            break

        page_no += 1
        out.extend(_resolved_market_from_house(raw) for raw in items)

        log.info(
            "poly-house markets loaded",
            page=page_no,
            page_markets=len(items),
            market_count=len(out),
            max_pages=max_pages,
            neg_risk=neg_risk,
        )

        if max_pages is not None and page_no >= max_pages:
            break
        if len(items) < _PAGE_LIMIT:
            break
        offset += len(items)

    log.info("poly-house resolved markets built", market_count=len(out))
    return out
