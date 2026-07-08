"""Weighted empirical outcome probability for open markets (search ``prop_raw``)."""

from __future__ import annotations

from holder_boost.models.markets import ResolvedMarket
from holder_boost.utils.math import wilson_ci


def _tag_idf_sum(tags: set[int], tag_markets_count: dict[int, int]) -> float:
    total = 0.0
    for tag_id in tags:
        count = tag_markets_count.get(tag_id, 0)
        if count > 0:
            total += 1.0 / count
    return total


def _market_weight(
    open_tags: set[int],
    hist_tags: set[int],
    tag_markets_count: dict[int, int],
) -> float:
    union_sum = _tag_idf_sum(open_tags | hist_tags, tag_markets_count)
    if union_sum == 0.0:
        return 0.0
    inter_sum = _tag_idf_sum(open_tags & hist_tags, tag_markets_count)
    return inter_sum / union_sum


def compute_prop_raw(
    *,
    open_tags: list[int],
    cent_price: int,
    outcome_index: int,
    resolved_markets: list[ResolvedMarket],
    tag_markets_count: dict[int, int],
    neg_risk: bool,
) -> tuple[float | None, float | None]:
    """Return weighted win rate at *cent_price* or ``None`` when no observations."""
    open_tag_set = set(open_tags)
    win_phat = 0.0
    total_phat = 0.0
    stop_loss_acc = 0
    stop_loss_total = 0.0
    stop_count_acc = 0

    for market in resolved_markets:
        if market.neg_risk != neg_risk:
            continue
        try:
            m_count = market.trade_cent_prices[outcome_index].count(cent_price)
        except IndexError:
            continue
        if m_count == 0:
            continue

        m_w = _market_weight(open_tag_set, set(market.tags), tag_markets_count)
        weighted = m_count * m_w
        total_phat += weighted
        if market.resolved_outcome_index == outcome_index:
            win_phat += weighted
            stop_loss_details = market.stop_prices.get(cent_price)
            if stop_loss_details is not None:
                stop_loss_acc += weighted * stop_loss_details.min_mean / 100
                stop_count_acc += weighted * stop_loss_details.count
                stop_loss_total += weighted

    prop = win_phat / total_phat if total_phat > 0.0 else None
    stop_loss_mean = stop_loss_acc / stop_loss_total if stop_loss_total > 0.0 else None
    stop_loss_count = (
        stop_count_acc / stop_loss_total if stop_loss_total > 0.0 else None
    )
    stop_loss_ci = (
        wilson_ci(stop_loss_mean, int(stop_loss_count) + 1)
        if stop_loss_mean is not None and stop_loss_count is not None
        else None
    )
    return prop, stop_loss_ci[0] if stop_loss_ci is not None else None
