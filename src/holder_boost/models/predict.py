"""Tag-level Yes win rate from resolved markets."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class Predict(BaseModel):
    """Historical Yes probability for markets under one Gamma tag."""

    model_config = ConfigDict(frozen=True)

    tag_id: int
    tag_slug: str
    tag_label: str
    outcome_probs: list[float]
    trade_price_probs: list[list[float]]
    trade_price_probs_raw: list[list[float | None]]
    yes_win_markets: int
    total_markets: int
    computed_at: datetime
