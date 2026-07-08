"""Resolved Polymarket markets from poly-house."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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
    holders: list[HolderDistribution] = Field(default_factory=list)
    max_days_left: int = Field(default=0)
