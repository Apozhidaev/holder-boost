"""poly-house activity API client."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import httpx

from holder_boost.config import get_settings
from holder_boost.http import HTTPError, get_json
from holder_boost.logging import get_logger
from holder_boost.models.markets import HolderDistribution

log = get_logger("poly_house.activity")


def compute_market_max_days_left(holders: Sequence[HolderDistribution]) -> int:
    values = [holder.max_days_left for holder in holders if holder.max_days_left >= 0]
    return max(values) if values else -1


def _proxy_wallets_from_payload(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        raise HTTPError(
            f"unexpected holders response: expected object, got {type(payload)!r}"
        )
    proxy_wallets_raw = payload.get("proxy_wallets")
    if proxy_wallets_raw is None:
        return []
    if not isinstance(proxy_wallets_raw, list):
        raise HTTPError("unexpected holders response: proxy_wallets is not a list")
    return [str(wallet) for wallet in proxy_wallets_raw]


async def fetch_market_proxy_wallets(
    client: httpx.AsyncClient,
    *,
    condition_id: str,
) -> list[str]:
    """Load proxy wallet list for *condition_id* from poly-house ``GET /activity/holders``."""
    base = get_settings().poly_house.base_url.rstrip("/")
    url = f"{base}/activity/holders"
    payload = await get_json(client, url, params={"condition_id": condition_id})
    proxy_wallets = _proxy_wallets_from_payload(payload)
    log.info(
        "poly-house holders loaded",
        market=condition_id,
        proxy_wallets=len(proxy_wallets),
    )
    return proxy_wallets


async def fetch_holder_distribution(
    client: httpx.AsyncClient,
    *,
    condition_id: str,
    proxy_wallet: str,
) -> HolderDistribution:
    """Load one holder distribution from poly-house ``GET /activity/hold-distribution``."""
    base = get_settings().poly_house.base_url.rstrip("/")
    url = f"{base}/activity/hold-distribution"
    payload = await get_json(
        client,
        url,
        params={"condition_id": condition_id, "proxy_wallet": proxy_wallet},
    )
    if not isinstance(payload, dict):
        raise HTTPError(
            f"unexpected hold-distribution response: expected object, got {type(payload)!r}"
        )
    return HolderDistribution.model_validate(payload)


async def _fetch_holder_distribution_safe(
    client: httpx.AsyncClient,
    *,
    condition_id: str,
    proxy_wallet: str,
) -> HolderDistribution | None:
    try:
        return await fetch_holder_distribution(
            client,
            condition_id=condition_id,
            proxy_wallet=proxy_wallet,
        )
    except HTTPError as exc:
        log.warning(
            "skipping wallet: holder-distribution fetch failed",
            market=condition_id,
            proxy_wallet=proxy_wallet,
            error=str(exc),
        )
        return None


async def fetch_market_holder_distribution(
    client: httpx.AsyncClient,
    *,
    condition_id: str,
) -> tuple[list[HolderDistribution], int]:
    """Load all holder distributions for *condition_id*."""
    proxy_wallets = await fetch_market_proxy_wallets(client, condition_id=condition_id)
    results = await asyncio.gather(
        *[
            _fetch_holder_distribution_safe(
                client,
                condition_id=condition_id,
                proxy_wallet=proxy_wallet,
            )
            for proxy_wallet in proxy_wallets
        ]
    )
    holders = [holder for holder in results if holder is not None]
    if not holders:
        raise HTTPError(f"no holders loaded for market {condition_id!r}")

    max_days_left = compute_market_max_days_left(holders)
    log.info(
        "poly-house holder-distribution loaded",
        market=condition_id,
        holders=len(holders),
        max_days_left=max_days_left,
    )
    return holders, max_days_left
