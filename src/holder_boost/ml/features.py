"""Feature helpers for holder-distribution rows."""

from __future__ import annotations

from collections.abc import Sequence

from holder_boost.models.markets import HolderDistribution


def wallet_column(wallet: str) -> str:
    return f"proxy_wallet_{wallet}"


def wallet_value_from_distribution(
    holder_by_wallet: dict[str, HolderDistribution],
    wallet: str,
    day: int,
) -> int:
    holder = holder_by_wallet.get(wallet)
    if holder is None:
        return -1
    return holder.distribution.get(day, -1)


def wallet_values_for_row(
    *,
    frequent_wallets: Sequence[str],
    holder_by_wallet: dict[str, HolderDistribution] | None = None,
    holders: dict[str, int] | None = None,
    day: int | None = None,
) -> dict[str, int]:
    """Build wallet column values for one feature row."""
    if (holder_by_wallet is None) == (holders is None):
        raise ValueError("provide exactly one of holder_by_wallet or holders")

    out: dict[str, int] = {}
    for wallet in frequent_wallets:
        column = wallet_column(wallet)
        if holders is not None:
            out[column] = holders.get(wallet, -1)
        else:
            assert holder_by_wallet is not None and day is not None
            out[column] = wallet_value_from_distribution(holder_by_wallet, wallet, day)
    return out


def feature_columns(frequent_wallets: Sequence[str]) -> list[str]:
    return ["days_left", *[wallet_column(wallet) for wallet in frequent_wallets]]


def row_is_all_missing(wallet_values: dict[str, int]) -> bool:
    return all(value == -1 for value in wallet_values.values())
