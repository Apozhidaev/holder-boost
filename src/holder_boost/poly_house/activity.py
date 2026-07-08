"""poly-house activity API client."""

from __future__ import annotations

from typing import Any

import httpx

from holder_boost.config import get_settings
from holder_boost.http import HTTPError, get_json
from holder_boost.logging import get_logger
from holder_boost.models.markets import HolderDistribution

log = get_logger("poly_house.activity")


def _holder_distribution_from_payload(payload: Any) -> tuple[list[HolderDistribution], int]:
    if not isinstance(payload, dict):
        raise HTTPError(
            f"unexpected holder-distribution response: expected object, got {type(payload)!r}"
        )
    holders_raw = payload.get("holders")
    if holders_raw is None:
        holders: list[HolderDistribution] = []
    elif not isinstance(holders_raw, list):
        raise HTTPError(
            "unexpected holder-distribution response: holders is not a list"
        )
    else:
        holders = [HolderDistribution.model_validate(item) for item in holders_raw]

    max_days_left_raw = payload.get("max_days_left", 0)
    try:
        max_days_left = int(max_days_left_raw)
    except (TypeError, ValueError) as exc:
        raise HTTPError(
            "unexpected holder-distribution response: "
            f"max_days_left is not an int: {max_days_left_raw!r}"
        ) from exc

    return holders, max_days_left


async def fetch_market_holder_distribution(
    client: httpx.AsyncClient,
    *,
    condition_id: str,
) -> tuple[list[HolderDistribution], int]:
    """Load holder distribution for *condition_id* from poly-house ``GET /activity/holder-distribution``."""
    base = get_settings().poly_house.base_url.rstrip("/")
    url = f"{base}/activity/holder-distribution"
    payload = await get_json(client, url, params={"condition_id": condition_id})
    holders, max_days_left = _holder_distribution_from_payload(payload)
    log.info(
        "poly-house holder-distribution loaded",
        market=condition_id,
        holders=len(holders),
        max_days_left=max_days_left,
    )
    return holders, max_days_left
