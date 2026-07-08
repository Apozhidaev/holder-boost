"""Build ``/predict`` request bodies from stored market holder data."""

from __future__ import annotations

from holder_boost.api.schemas import PredictHolder, PredictRequest
from holder_boost.models.markets import ResolvedMarket


def build_predict_request(market: ResolvedMarket, *, days_left: int) -> PredictRequest:
    """Map holder ``distribution[days_left]`` into a ``PredictRequest``."""
    holders: list[PredictHolder] = []
    for holder in market.holders:
        outcome_index = holder.distribution.get(days_left)
        if outcome_index not in (0, 1):
            continue
        holders.append(
            PredictHolder(
                proxy_wallet=holder.proxy_wallet,
                outcome_index=outcome_index,
            )
        )
    if not holders:
        raise ValueError(f"no holder positions at days_left={days_left}")
    return PredictRequest(days_left=days_left, holders=holders)
