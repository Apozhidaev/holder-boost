"""Open Polymarket markets from Gamma ``/markets/keyset``."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class OpenMarket(BaseModel):
    """One open market row for screening."""

    model_config = ConfigDict(frozen=True)

    id: str
    slug: str
    question: str
    event_slug: str
    event_title: str
    condition_id: str
    outcomes: list[str]
    outcome_prices: list[str]
    clob_token_ids: list[str]
    end_date: str | None
    group_item_title: str | None = None
    accepting_orders: bool
    neg_risk: bool
    tags: list[int]
    deadline: date

    def deadline_title(self) -> str:
        return self.deadline.strftime("%b %d, %Y")


class StopPriceDetails(BaseModel):
    """Stop-loss bucket for one rounded entry price (cents)."""

    model_config = ConfigDict(frozen=True)

    count: int
    min_price: int
    min_mean: int


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
