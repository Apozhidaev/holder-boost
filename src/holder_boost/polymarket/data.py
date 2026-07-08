"""Polymarket Data API client: user positions."""

from __future__ import annotations

from typing import Any

import httpx

from holder_boost.config import get_settings
from holder_boost.logging import get_logger
from holder_boost.polymarket.http import PolymarketHTTPError, get_json

log = get_logger("polymarket.data")

_POSITIONS_PAGE_LIMIT = 500


def _position_key(raw: dict[str, Any]) -> tuple[str, int] | None:
    """Map one ``/positions`` row to ``(condition_id, outcome_index)`` if open."""
    condition_id = raw.get("conditionId")
    outcome_index = raw.get("outcomeIndex")
    size = raw.get("size")
    if condition_id is None or outcome_index is None or size is None:
        return None
    try:
        if float(size) <= 0:
            return None
        return (str(condition_id), int(outcome_index))
    except (TypeError, ValueError):
        return None


def _flatten_holders(payload: Any) -> list[dict[str, Any]]:
    """Parse ``GET /holders`` response into a flat list of holder dicts."""
    if not isinstance(payload, list):
        return []
    out: list[dict[str, Any]] = []
    for group in payload:
        if not isinstance(group, dict):
            continue
        holders = group.get("holders")
        if not isinstance(holders, list):
            continue
        for row in holders:
            if isinstance(row, dict):
                out.append(row)
    return out


async def fetch_open_position_keys(
    client: httpx.AsyncClient,
    *,
    user: str,
) -> set[tuple[str, int]]:
    """Load all open position keys for *user* via paginated ``GET /positions``."""
    base = get_settings().polymarket.data_url.rstrip("/")
    url = f"{base}/positions"
    keys: set[tuple[str, int]] = set()
    offset = 0
    pages = 0

    while True:
        payload = await get_json(
            client,
            url,
            params={
                "user": user,
                "limit": _POSITIONS_PAGE_LIMIT,
                "offset": offset,
                "sizeThreshold": 0,
            },
        )
        if not isinstance(payload, list):
            raise PolymarketHTTPError(
                f"unexpected Data API /positions response: expected list, got {type(payload)!r}"
            )
        pages += 1
        if not payload:
            break
        for row in payload:
            if not isinstance(row, dict):
                continue
            key = _position_key(row)
            if key is not None:
                keys.add(key)
        if len(payload) < _POSITIONS_PAGE_LIMIT:
            break
        offset += len(payload)

    log.info("data positions loaded", user=user, positions=len(keys), pages=pages)
    return keys


async def fetch_holders(
    client: httpx.AsyncClient,
    condition_id: str,
) -> list[dict[str, Any]]:
    """Load top holders for *condition_id* (single request, no pagination)."""
    base = get_settings().polymarket.data_url.rstrip("/")
    url = f"{base}/holders"
    limit = get_settings().search_holder.holders_limit
    payload = await get_json(
        client,
        url,
        params={
            "market": condition_id,
            "limit": limit,
            "offset": 0,
        },
    )
    return _flatten_holders(payload)
