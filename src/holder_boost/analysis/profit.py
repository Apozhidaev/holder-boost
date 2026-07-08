"""Spread profit math: iteration counts and compound return formulas."""

from __future__ import annotations

from datetime import UTC, datetime

_SECONDS_PER_YEAR = 365.25 * 24 * 3600


def get_annualized_profit_value(
    *,
    total_return: float,
    taken_at: datetime,
    far_deadline: datetime,
) -> float:
    """Simple annualized return: ``r / years`` from ``taken_at`` to ``far_deadline``.

    Linear normalization (no reinvestment / CAGR). ``years`` is UTC calendar time
    between the two instants, floored to **one second**.
    """
    t0 = taken_at.astimezone(UTC)
    t1 = far_deadline.astimezone(UTC)
    seconds = max((t1 - t0).total_seconds(), 1.0)
    years = seconds / _SECONDS_PER_YEAR
    return total_return / years


def get_daily_profit_value(
    *,
    total_return: float,
    taken_at: datetime,
    deadline: datetime,
) -> float:
    """Simple return per calendar day: ``r / days`` from ``taken_at`` to ``deadline``.

    ``days`` uses UTC-normalized calendar time, floored to **one second** (same
    spirit as :func:`get_annualized_profit_value`).
    """
    t0 = taken_at.astimezone(UTC)
    t1 = deadline.astimezone(UTC)
    seconds = max((t1 - t0).total_seconds(), 1.0)
    days = seconds / 86400.0
    return total_return / days
