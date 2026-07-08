"""Build training DataFrame from resolved markets."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from holder_boost.ml.features import (
    feature_columns,
    row_is_all_missing,
    wallet_values_for_row,
)
from holder_boost.models.markets import ResolvedMarket


def _frequent_wallets(markets: Sequence[ResolvedMarket], min_wallet_count: int) -> list[str]:
    counts: dict[str, int] = {}
    for market in markets:
        seen: set[str] = set()
        for holder in market.holders:
            if holder.proxy_wallet in seen:
                continue
            seen.add(holder.proxy_wallet)
            counts[holder.proxy_wallet] = counts.get(holder.proxy_wallet, 0) + 1
    return sorted(wallet for wallet, count in counts.items() if count >= min_wallet_count)


def _expand_rows(
    markets: Sequence[ResolvedMarket],
    frequent_wallets: Sequence[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for market in markets:
        if market.max_days_left < 0:
            continue
        holder_by_wallet = {holder.proxy_wallet: holder for holder in market.holders}
        for days_left in range(market.max_days_left + 1):
            wallet_values = wallet_values_for_row(
                frequent_wallets=frequent_wallets,
                holder_by_wallet=holder_by_wallet,
                day=days_left,
            )
            if row_is_all_missing(wallet_values):
                continue
            rows.append(
                {
                    "days_left": days_left,
                    "resolved_outcome_index": market.resolved_outcome_index,
                    **wallet_values,
                }
            )
    return rows


def build_training_frame(
    markets: Sequence[ResolvedMarket],
    *,
    min_wallet_count: int,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return ``(X, y, frequent_wallets)`` for CatBoost training."""
    frequent_wallets = _frequent_wallets(markets, min_wallet_count)
    if not frequent_wallets:
        raise ValueError(
            f"no frequent wallets after min_wallet_count={min_wallet_count} filter",
        )

    rows = _expand_rows(markets, frequent_wallets)
    if not rows:
        raise ValueError("no training rows after filtering all-(-1) wallet rows")

    columns = feature_columns(frequent_wallets)
    X = pd.DataFrame([{col: row[col] for col in columns} for row in rows])
    y = pd.Series(
        [row["resolved_outcome_index"] for row in rows],
        dtype=int,
        name="resolved_outcome_index",
    )
    return X, y, frequent_wallets
