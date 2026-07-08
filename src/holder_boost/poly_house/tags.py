"""poly-house tags API client."""

from __future__ import annotations

import asyncio

import httpx

from holder_boost.config import get_settings
from holder_boost.http import get_json
from holder_boost.logging import get_logger
from holder_boost.models.markets import ResolvedMarket
from holder_boost.models.tag import Tag

log = get_logger("poly_house.tags")


def _tag_from_house(raw: dict) -> Tag:
    return Tag(id=raw["_id"], slug=raw["slug"], label=raw["label"])


async def _fetch_tag(client: httpx.AsyncClient, *, base: str, tag_id: int) -> Tag:
    payload = await get_json(client, f"{base}/tags/{tag_id}")
    return _tag_from_house(payload)


async def fetch_tags_for_markets(
    client: httpx.AsyncClient,
    markets: list[ResolvedMarket],
) -> list[Tag]:
    """Load tag details for unique ``tag_ids`` referenced by *markets*."""
    tag_ids = sorted({tag_id for market in markets for tag_id in market.tags})
    if not tag_ids:
        return []

    base = get_settings().poly_house.base_url.rstrip("/")
    tags = await asyncio.gather(
        *(_fetch_tag(client, base=base, tag_id=tag_id) for tag_id in tag_ids)
    )
    log.info("poly-house tags loaded", count=len(tags))
    return list(tags)
