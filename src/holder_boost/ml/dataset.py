"""Build training DataFrame from resolved markets."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from holder_boost.models.markets import HolderDistribution, ResolvedMarket


def _wallet_column(wallet: str) -> str:
    return f"proxy_wallet_{wallet}"


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


def _wallet_value(holder_by_wallet: dict[str, HolderDistribution], wallet: str, day: int) -> int:
    holder = holder_by_wallet.get(wallet)
    if holder is None:
        return -1
    return holder.distribution.get(day, -1)


def _expand_rows(
    markets: Sequence[ResolvedMarket],
    frequent_wallets: Sequence[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    wallet_columns = [_wallet_column(wallet) for wallet in frequent_wallets]

    for market in markets:
        holder_by_wallet = {holder.proxy_wallet: holder for holder in market.holders}
        for day in range(market.max_days + 1):
            wallet_values = {
                column: _wallet_value(holder_by_wallet, wallet, day)
                for column, wallet in zip(wallet_columns, frequent_wallets, strict=True)
            }
            if all(value == -1 for value in wallet_values.values()):
                continue
            rows.append(
                {
                    "day": day,
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

    wallet_columns = [_wallet_column(wallet) for wallet in frequent_wallets]
    feature_columns = ["day", *wallet_columns]
    X = pd.DataFrame([{col: row[col] for col in feature_columns} for row in rows])
    y = pd.Series(
        [row["resolved_outcome_index"] for row in rows],
        dtype=int,
        name="resolved_outcome_index",
    )
    return X, y, frequent_wallets
