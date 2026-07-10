"""poly-house markets API client."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from holder_boost.config import get_settings
from holder_boost.http import get_json
from holder_boost.logging import get_logger
from holder_boost.models.markets import ResolvedMarket

log = get_logger("poly_house.markets")

_PAGE_LIMIT = 500
_MINUTES_PER_DAY = 1440


def _resolved_market_from_house(raw: dict[str, Any]) -> ResolvedMarket:
    return ResolvedMarket(
        id=raw["_id"],
        slug=raw["slug"],
        condition_id=raw["condition_id"],
        resolved_outcome_index=raw["resolved_outcome_index"],
        duration_days=raw["duration_minutes"] // _MINUTES_PER_DAY
    )


async def fetch_resolved_markets(
    client: httpx.AsyncClient,
    *,
    tag_ids: Sequence[int] | None = None,
    neg_risk: bool = False,
    exclude_tag_ids: Sequence[int] | None = None,
    min_duration_days: int = 0,
    max_pages: int | None = None,
) -> list[ResolvedMarket]:
    """Load closed markets from poly-house ``GET /markets``."""
    base = get_settings().poly_house.base_url.rstrip("/")
    url = f"{base}/markets"

    out: list[ResolvedMarket] = []
    offset = 0
    page_no = 0

    include_ids: list[int] | None = list(tag_ids) if tag_ids else None
    exclude_ids = list(exclude_tag_ids) if exclude_tag_ids else []
    duration_from = min_duration_days * _MINUTES_PER_DAY

    while True:
        params: dict[str, Any] = {
            "resolved": True,
            "temporal": True,

            "neg_risk": neg_risk,
            "limit": _PAGE_LIMIT,
            "offset": offset,
        }
        if include_ids:
            params["tag_ids"] = include_ids
        if exclude_ids:
            params["exclude_tag_ids"] = exclude_ids
        if min_duration_days > 0:
            params["duration_from"] = duration_from

        payload = await get_json(
            client,
            url,
            params=params,
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
            exclude_tag_ids=exclude_ids,
            min_duration_days=min_duration_days,
        )

        if max_pages is not None and page_no >= max_pages:
            break
        if len(items) < _PAGE_LIMIT:
            break
        offset += len(items)

    log.info("poly-house resolved markets built", market_count=len(out))
    return out
