"""Gamma API client: open markets lookup for ``search``."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from holder_boost.config import get_settings
from holder_boost.logging import get_logger
from holder_boost.models.contract_error import ContractError
from holder_boost.models.markets import OpenMarket
from holder_boost.models.tag import Tag
from holder_boost.polymarket.http import PolymarketHTTPError, get_json
from holder_boost.polymarket.tz import (
    deadline_from_end_date_iso,
    deadline_from_group_item_title,
    deadline_from_none,
    series_calendar_date,
    series_day_end_et,
)

log = get_logger("polymarket.gamma")

_GAMMA_MARKETS_KEYSET_MAX_LIMIT = 100


def _parse_json_array(raw: Any, field: str, market_id: str) -> list[str]:
    """Parse a Gamma string-encoded JSON array into a Python list."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PolymarketHTTPError(
                f"market {market_id}: malformed JSON in {field}: {raw!r}"
            ) from exc
        if not isinstance(decoded, list):
            raise PolymarketHTTPError(
                f"market {market_id}: {field} is not a JSON array: {raw!r}"
            )
        return [str(x) for x in decoded]
    raise PolymarketHTTPError(
        f"market {market_id}: unexpected {field} type: {type(raw)!r}"
    )


def _tags_from_raw(raw: dict) -> list[Tag]:
    """Map Gamma ``tags`` on an event payload into :class:`Tag` rows."""
    tags_raw = raw.get("tags")
    if not isinstance(tags_raw, list):
        return []
    out: list[Tag] = []
    for item in tags_raw:
        if not isinstance(item, dict):
            raise ContractError(f"invalid tag: {item!r}")
        out.append(
            Tag(
                id=int(item["id"]),
                slug=str(item["slug"]),
                label=str(item.get("label", "")),
            )
        )
    return out


def _raw_matches_tag_slugs(raw: dict[str, Any], wanted: frozenset[str]) -> bool:
    """True if some Gamma tag slug on ``raw`` is in ``wanted``."""
    return any(t.slug in wanted for t in _tags_from_raw(raw))


def _open_market_from_raw(raw: dict[str, Any]) -> OpenMarket:
    """Map one Gamma ``/markets/keyset`` row to :class:`OpenMarket`."""
    market_id = str(raw["id"])
    events_raw = raw.get("events")
    if not isinstance(events_raw, list) or not events_raw:
        raise ContractError(f"market {market_id}: missing events")
    first_event = events_raw[0]
    if not isinstance(first_event, dict) or "slug" not in first_event:
        raise ContractError(f"market {market_id}: events[0] missing slug")
    event_slug = str(first_event["slug"])
    event_title = str(first_event["title"])

    outcomes = _parse_json_array(raw.get("outcomes"), "outcomes", market_id)
    outcome_prices = _parse_json_array(
        raw.get("outcomePrices"), "outcomePrices", market_id
    )
    clob_token_ids = _parse_json_array(
        raw.get("clobTokenIds"), "clobTokenIds", market_id
    )
    if not outcomes or len(outcomes) != len(outcome_prices) or len(outcomes) != len(
        clob_token_ids
    ):
        raise ContractError(
            f"market {market_id}: outcomes/outcomePrices/clobTokenIds missing or length mismatch"
        )

    iso_raw = raw.get("endDateIso")
    end_date = None if iso_raw is None else str(iso_raw).strip() or None
    gt_raw = raw.get("groupItemTitle")
    group_item_title = None if gt_raw is None else (str(gt_raw).strip() or None)

    deadline = deadline_from_group_item_title(group_item_title)
    if deadline is None:
        deadline = deadline_from_end_date_iso(end_date)
    if deadline is None:
        deadline = deadline_from_none()

    return OpenMarket(
        id=market_id,
        slug=str(raw.get("slug", "")),
        question=str(raw.get("question", "")),
        event_slug=event_slug,
        event_title=event_title,
        condition_id=str(raw.get("conditionId", "")),
        outcomes=outcomes,
        outcome_prices=outcome_prices,
        clob_token_ids=clob_token_ids,
        end_date=end_date,
        group_item_title=group_item_title,
        accepting_orders=bool(raw.get("acceptingOrders", False)),
        neg_risk=bool(raw.get("negRisk", False)),
        tags=_tag_ids_from_market_raw(raw),
        deadline=deadline,
    )


def _end_date_max_iso(deadline_in_days: int) -> str:
    """Upper bound for Gamma ``end_date_max``: end of (today ET + *days*)."""
    today = series_calendar_date(datetime.now(UTC))
    max_day = today + timedelta(days=deadline_in_days)
    return series_day_end_et(max_day).isoformat()


def _tag_ids_from_tags_raw(tags_raw: Any) -> list[int]:
    if not isinstance(tags_raw, list):
        return []
    out: list[int] = []
    for item in tags_raw:
        if not isinstance(item, dict) or "id" not in item:
            continue
        try:
            out.append(int(item["id"]))
        except (TypeError, ValueError):
            continue
    return out


def _tag_ids_from_market_raw(raw: dict[str, Any]) -> list[int]:
    """Tag ids from market row; with ``include_tag=true`` Gamma puts them on ``tags``."""
    tags = _tag_ids_from_tags_raw(raw.get("tags"))
    if tags:
        return tags
    events_raw = raw.get("events")
    if isinstance(events_raw, list) and events_raw:
        first_event = events_raw[0]
        if isinstance(first_event, dict):
            return _tag_ids_from_tags_raw(first_event.get("tags"))
    return []


async def _fetch_row_markets_by_id(
    client: httpx.AsyncClient,
    *,
    closed: bool | None = None,
    tag_ids: Sequence[int] | None = None,
    end_date_max: str | None = None,
    start_date_min: str | None = None,
    uma_resolution_status: str | None = None,
    max_pages: int | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Load markets from Gamma ``GET /markets/keyset`` into RAM.

    Args:
        client: HTTP client
        closed: if True, only closed markets are loaded
        uma_resolution_status: if not None, only markets with this UMA resolution status are loaded
        start_date_min: if not None, only markets with start date greater than or equal to this date are loaded
    """
    base = get_settings().polymarket.gamma_url.rstrip("/")
    url = f"{base}/markets/keyset"

    after_cursor: str | None = None
    out: dict[str, dict[str, Any]] = {}
    page_no = 0

    while True:
        params: dict[str, Any] = {
            "limit": _GAMMA_MARKETS_KEYSET_MAX_LIMIT,
            "include_tag": "true",
        }
        if closed is not None:
            params["closed"] = "true" if closed else "false"
        if tag_ids:
            params["tag_id"] = list(tag_ids)
        if end_date_max is not None:
            params["end_date_max"] = end_date_max
        if start_date_min is not None:
            params["start_date_min"] = start_date_min
        if uma_resolution_status is not None:
            params["uma_resolution_status"] = uma_resolution_status
        if after_cursor is not None:
            params["after_cursor"] = after_cursor

        payload = await get_json(client, url, params=params)
        if not isinstance(payload, dict):
            raise PolymarketHTTPError(
                "unexpected Gamma /markets/keyset response: top-level JSON is not an object"
            )

        raw_markets = payload.get("markets")
        if raw_markets is None:
            raw_markets = []
        if not isinstance(raw_markets, list):
            raise PolymarketHTTPError(
                "unexpected Gamma /markets/keyset response: markets is not a list"
            )

        if not raw_markets:
            break

        page_no += 1

        for item in raw_markets:
            if not isinstance(item, dict) or "id" not in item:
                raise PolymarketHTTPError(
                    "Gamma /markets/keyset markets entry missing id or not an object"
                )
            out[str(item["id"])] = item

        log.info(
            "gamma keyset markets loaded",
            page=page_no,
            page_markets=len(raw_markets),
            market_count=len(out),
            max_pages=max_pages,
        )

        if max_pages is not None and page_no >= max_pages:
            break

        next_cursor = payload.get("next_cursor")
        if not next_cursor:
            break
        after_cursor = str(next_cursor)

    return out


async def fetch_open_markets(
    client: httpx.AsyncClient,
    *,
    tag_ids: Sequence[int] | None = None,
    deadline_in_days: int | None = None,
) -> list[OpenMarket]:
    """Load open Gamma markets via keyset and map to :class:`OpenMarket`."""
    end_date_max = (
        _end_date_max_iso(deadline_in_days) if deadline_in_days is not None else None
    )
    raw_by_id = await _fetch_row_markets_by_id(
        client,
        closed=False,
        tag_ids=tag_ids,
        end_date_max=end_date_max,
    )
    out: list[OpenMarket] = []
    for raw in raw_by_id.values():
        if not isinstance(raw, dict):
            continue
        market_id = raw.get("id")
        try:
            out.append(_open_market_from_raw(raw))
        except (ContractError, PolymarketHTTPError) as exc:
            log.warning(
                "skipping open market",
                market_id=market_id,
                error=str(exc),
            )
    log.info(
        "gamma open markets built",
        market_count=len(out),
        end_date_max=end_date_max,
    )
    return out
