"""poly-house activity API client."""

from __future__ import annotations

from typing import Any

import httpx

from holder_boost.config import get_settings
from holder_boost.http import HTTPError, get_json
from holder_boost.logging import get_logger
from holder_boost.models.markets import HolderDistribution

log = get_logger("poly_house.activity")


def _holder_distributions_from_payload(payload: Any) -> list[HolderDistribution]:
    if not isinstance(payload, dict):
        raise HTTPError(
            f"unexpected holder-distribution response: expected object, got {type(payload)!r}"
        )
    holders_raw = payload.get("holders")
    if holders_raw is None:
        return []
    if not isinstance(holders_raw, list):
        raise HTTPError(
            "unexpected holder-distribution response: holders is not a list"
        )
    return [HolderDistribution.model_validate(item) for item in holders_raw]


async def fetch_market_holder_distribution(
    client: httpx.AsyncClient,
    *,
    condition_id: str,
) -> list[HolderDistribution]:
    """Load holder distribution for *condition_id* from poly-house ``GET /activity/holder-distribution``."""
    base = get_settings().poly_house.base_url.rstrip("/")
    url = f"{base}/activity/holder-distribution"
    payload = await get_json(client, url, params={"condition_id": condition_id})
    holders = _holder_distributions_from_payload(payload)
    log.info(
        "poly-house holder-distribution loaded",
        market=condition_id,
        holders=len(holders),
    )
    return holders
