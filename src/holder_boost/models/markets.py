"""Resolved Polymarket markets from poly-house."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StopPriceDetails(BaseModel):
    """Stop-loss bucket for one rounded entry price (cents)."""

    model_config = ConfigDict(frozen=True)

    count: int
    min_price: int
    min_mean: int


class HolderDistribution(BaseModel):
    proxy_wallet: str
    distribution: dict[int, int]


class ResolvedMarket(BaseModel):
    """One resolved market."""

    model_config = ConfigDict(frozen=True)

    id: int
    slug: str
    event_id: int
    condition_id: str
    clob_token_ids: list[str]
    neg_risk: bool
    tags: list[int]
    resolved_outcome_index: int
    trade_cent_prices: list[list[int]]
    stop_prices: dict[int, StopPriceDetails] = Field(default_factory=dict)
    holders: list[HolderDistribution] = Field(default_factory=list)
    max_days: int = Field(default=0)

