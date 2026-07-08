"""poly-house trades API client."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from holder_boost.config import get_settings
from holder_boost.http import get_json
from holder_boost.logging import get_logger
from holder_boost.models.markets import StopPriceDetails

log = get_logger("poly_house.trades")


def _take_cent_prices(payload: dict[str, Any]) -> list[int]:
    return [round(float(price) * 100) for price in payload["items"]]


def _stop_cent_prices(payload: dict[str, Any]) -> dict[int, StopPriceDetails]:
    out: dict[int, StopPriceDetails] = {}
    for item in payload["items"]:
        cent = round(float(item["price"]) * 100)
        out[cent] = StopPriceDetails(
            count=int(item["count"]),
            min_price=round(float(item["min_price"]) * 100),
            min_mean=round(float(item["min_mean"]) * 100),
        )
    return out


async def _fetch_outcome_take_cent_prices(
    client: httpx.AsyncClient,
    *,
    url: str,
    condition_id: str,
    outcome_index: int,
) -> list[int]:
    payload = await get_json(
        client,
        url,
        params={
            "condition_id": condition_id,
            "outcome_index": outcome_index,
            "side": "BUY",
        },
    )
    prices = _take_cent_prices(payload)
    log.info(
        "poly-house take-prices loaded",
        market=condition_id,
        outcome_index=outcome_index,
        prices=len(prices),
    )
    return prices


async def fetch_market_trade_cent_prices(
    client: httpx.AsyncClient,
    *,
    condition_id: str,
) -> list[list[int]]:
    """Load BUY take prices for *condition_id* from poly-house ``GET /trades/take-prices``."""
    base = get_settings().poly_house.base_url.rstrip("/")
    url = f"{base}/trades/take-prices"
    yes_prices, no_prices = await asyncio.gather(
        _fetch_outcome_take_cent_prices(
            client, url=url, condition_id=condition_id, outcome_index=0
        ),
        _fetch_outcome_take_cent_prices(
            client, url=url, condition_id=condition_id, outcome_index=1
        ),
    )
    return [yes_prices, no_prices]


async def fetch_market_stop_cent_prices(
    client: httpx.AsyncClient,
    *,
    condition_id: str,
    outcome_index: int,
) -> dict[int, StopPriceDetails]:
    """Load BUY stop prices for *condition_id* from poly-house ``GET /trades/stop-prices``."""
    base = get_settings().poly_house.base_url.rstrip("/")
    url = f"{base}/trades/stop-prices"
    payload = await get_json(
        client,
        url,
        params={
            "condition_id": condition_id,
            "outcome_index": outcome_index,
            "side": "BUY",
        },
    )
    prices = _stop_cent_prices(payload)
    log.info(
        "poly-house stop-prices loaded",
        market=condition_id,
        outcome_index=outcome_index,
        buckets=len(prices),
    )
    return prices
