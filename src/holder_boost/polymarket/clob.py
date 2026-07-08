"""CLOB API client: order book lookup for an asset (CLOB token id).

Order books from the CLOB endpoint arrive in worst-first order (bids ascending,
asks descending). We normalize to best-first so callers can safely index
``bids[0]`` and ``asks[0]``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from holder_boost.config import get_settings
from holder_boost.logging import get_logger
from holder_boost.models.markets import OpenMarket
from holder_boost.models.orderbook import OrderBook, OrderLevel
from holder_boost.polymarket.http import PolymarketHTTPError

log = get_logger("polymarket.clob")


def _level_from_raw(raw: dict[str, Any]) -> OrderLevel:
    return OrderLevel(price=float(raw["price"]), size=float(raw["size"]))


def _orderbook_from_raw(raw: dict[str, Any]) -> OrderBook:
    asset_id = str(raw["asset_id"])
    raw_bids = [_level_from_raw(b) for b in raw.get("bids") or []]
    raw_asks = [_level_from_raw(a) for a in raw.get("asks") or []]

    bids = sorted(raw_bids, key=lambda lvl: lvl.price, reverse=True)
    asks = sorted(raw_asks, key=lambda lvl: lvl.price)
    return OrderBook(asset_id=asset_id, bids=bids, asks=asks)


async def tradeable_markets_with_orderbooks(
    client: httpx.AsyncClient,
    markets: list[OpenMarket],
) -> tuple[list[OpenMarket], dict[str, OrderBook]]:
    """Markets where every outcome has a CLOB book with a best ask; merged books map."""
    books: dict[str, OrderBook] = {}
    out: list[OpenMarket] = []
    for market in markets:
        if not market.accepting_orders:
            continue
        local: list[tuple[str, OrderBook]] = []
        ok = True
        for i in (0, 1):
            book = await get_orderbook(client, market.clob_token_ids[i])
            if book.best_ask is None:
                ok = False
                break
            local.append((market.clob_token_ids[i], book))
        if ok:
            for aid, b in local:
                books[aid] = b
            out.append(market)
    out.sort(key=lambda m: m.id)
    return out, books


async def get_orderbook(client: httpx.AsyncClient, asset_id: str) -> OrderBook:
    """Fetch and normalize the order book for a CLOB token id.

    A **404** (no orderbook for token) yields an empty book so callers can skip
    the market instead of failing the whole pipeline.
    """
    base = get_settings().polymarket.clob_url.rstrip("/")
    url = f"{base}/book"
    params = {"token_id": asset_id}
    last_err: BaseException | None = None
    for attempt in range(3):
        try:
            response = await client.get(url, params=params)
        except (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
        ) as exc:
            last_err = exc
            if attempt < 2:
                await asyncio.sleep(0.5)
            continue
        if response.status_code == 404:
            log.info("clob no orderbook", asset_id=asset_id)
            return OrderBook(asset_id=asset_id, bids=[], asks=[])
        if 500 <= response.status_code < 600:
            last_err = PolymarketHTTPError(
                f"GET {url} transient {response.status_code}: {response.text[:200]}"
            )
            if attempt < 2:
                await asyncio.sleep(0.5)
            continue
        if response.status_code >= 400:
            raise PolymarketHTTPError(
                f"GET {url} failed: {response.status_code} {response.text[:200]}"
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise PolymarketHTTPError(
                f"unexpected CLOB response shape for asset_id={asset_id}"
            )
        book = _orderbook_from_raw(payload)
        log.info(
            "clob orderbook fetched",
            asset_id=asset_id,
            bids=len(book.bids),
            asks=len(book.asks),
            best_bid=book.best_bid.price if book.best_bid else None,
            best_ask=book.best_ask.price if book.best_ask else None,
        )
        return book
    raise PolymarketHTTPError(
        f"GET {url} failed after retries: {last_err}"
    ) from last_err
