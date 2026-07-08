"""Order book domain models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class OrderLevel(BaseModel):
    model_config = ConfigDict(frozen=True)

    price: float
    size: float


class OrderBook(BaseModel):
    """Top-of-book aware order book for a single CLOB asset."""

    model_config = ConfigDict(frozen=True)

    asset_id: str
    bids: list[OrderLevel]
    asks: list[OrderLevel]

    @property
    def best_ask(self) -> OrderLevel | None:
        return self.asks[0] if self.asks else None

    @property
    def best_bid(self) -> OrderLevel | None:
        return self.bids[0] if self.bids else None
