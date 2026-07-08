"""poly-house catalog: resolved markets and tags for ``sync``."""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from holder_boost.models.markets import ResolvedMarket
from holder_boost.models.tag import Tag
from holder_boost.poly_house.markets import fetch_resolved_markets
from holder_boost.poly_house.tags import fetch_tags_for_markets


async def fetch_resolved_markets_and_tags(
    client: httpx.AsyncClient,
    *,
    tag_ids: Sequence[int] | None = None,
    neg_risk: bool = False,
    max_pages: int | None = None,
) -> tuple[list[ResolvedMarket], list[Tag]]:
    """Load resolved markets and their tags from poly-house."""
    markets = await fetch_resolved_markets(
        client,
        tag_ids=tag_ids,
        neg_risk=neg_risk,
        max_pages=max_pages,
    )
    tags = await fetch_tags_for_markets(client, markets)
    return markets, tags
