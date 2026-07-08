"""Search screen result row after CLOB price filter."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from holder_boost.models.markets import OpenMarket
from holder_boost.models.orderbook import OrderBook


class SearchMatch(BaseModel):
    """One market that passed the configured best-ask corridor."""

    model_config = ConfigDict(frozen=True)

    market: OpenMarket
    outcome_index: int
    outcome_label: str
    ask: float
    bid: float | None
    spread: float | None

    @classmethod
    def try_from_orderbook(
        cls,
        market: OpenMarket,
        book: OrderBook,
        *,
        outcome_index: int,
        min_price: float,
        max_price: float,
        max_spread: float,
    ) -> SearchMatch | None:
        best_ask = book.best_ask
        if best_ask is None:
            return None
        ask = best_ask.price
        if ask < min_price or ask > max_price:
            return None
        best_bid = book.best_bid
        bid = best_bid.price if best_bid is not None else None
        if bid is None:
            return None
        spread = round(ask - bid, 4)
        if spread > max_spread:
            return None
        return cls(
            market=market,
            outcome_index=outcome_index,
            outcome_label=market.outcomes[outcome_index],
            ask=ask,
            bid=bid,
            spread=spread,
        )
