"""Compute tag-level Yes win probability from resolved markets."""

from __future__ import annotations

from datetime import datetime
from itertools import pairwise

from holder_boost.models.markets import ResolvedMarket
from holder_boost.models.predict import Predict
from holder_boost.models.tag import Tag

_MIN_CENT = 0
_MAX_CENT = 100
_CENT_PRICES = _MAX_CENT + 1  # 0..100


def _compute_trade_price_probs_raw(
    markets: list[ResolvedMarket], outcome_index: int
) -> list[float | None]:
    probs: list[float | None] = []
    for price in range(_CENT_PRICES):
        win_prices = 0
        total_prices = 0
        for m in markets:
            count = m.trade_cent_prices[outcome_index].count(price)
            if count == 0:
                continue
            total_prices += count
            if m.resolved_outcome_index == outcome_index:
                win_prices += count
        probs.append(win_prices / total_prices if total_prices > 0 else None)
    return probs


def _require_prob(probs: list[float | None], index: int) -> float:
    value = probs[index]
    if value is None:
        raise ValueError(f"missing prob at cent {index}")
    return value


def _linear_fill(
    probs: list[float | None],
    start: int,
    end: int,
    p_start: float,
    p_end: float,
) -> None:
    if end <= start:
        return
    for i in range(start + 1, end):
        if probs[i] is None:
            probs[i] = p_start + (p_end - p_start) * ((i - start) / (end - start))


def _impute_price_probs(raw: list[float | None]) -> list[float]:
    """Fill missing cent probs: 0¢ first, gaps between measured points, flat tail to 100¢."""
    probs: list[float | None] = list(raw)
    measured = [i for i, p in enumerate(raw) if p is not None]

    if probs[_MIN_CENT] is None:
        probs[_MIN_CENT] = 0.0

    if not measured:
        for i in range(_MIN_CENT + 1, _CENT_PRICES):
            probs[i] = 0.0
        return [_require_prob(probs, i) for i in range(_CENT_PRICES)]

    for prev, curr in pairwise(measured):
        _linear_fill(
            probs,
            prev,
            curr,
            _require_prob(probs, prev),
            _require_prob(probs, curr),
        )

    first = measured[0]
    if first > _MIN_CENT:
        _linear_fill(
            probs,
            _MIN_CENT,
            first,
            _require_prob(probs, _MIN_CENT),
            _require_prob(probs, first),
        )

    last = measured[-1]
    p_last = _require_prob(probs, last)
    for i in range(last + 1, _CENT_PRICES):
        if probs[i] is None:
            probs[i] = p_last

    return [_require_prob(probs, i) for i in range(_CENT_PRICES)]


def compute_predict(
    tag: Tag,
    markets: list[ResolvedMarket],
    *,
    computed_at: datetime,
) -> Predict | None:
    """Return :class:`Predict` or ``None`` when *markets* is empty."""
    total = len(markets)
    if total == 0:
        return None
    yes_wins = sum(1 for m in markets if m.resolved_outcome_index == 0)
    prob_yes = yes_wins / total
    trade_price_probs_raw = [
        _compute_trade_price_probs_raw(markets, 0),
        _compute_trade_price_probs_raw(markets, 1),
    ]
    trade_price_probs = [_impute_price_probs(raw) for raw in trade_price_probs_raw]
    return Predict(
        tag_id=tag.id,
        tag_slug=tag.slug,
        tag_label=tag.label,
        outcome_probs=[prob_yes, 1 - prob_yes],
        trade_price_probs=trade_price_probs,
        trade_price_probs_raw=trade_price_probs_raw,
        yes_win_markets=yes_wins,
        total_markets=total,
        computed_at=computed_at,
    )
