"""US Eastern time: Polymarket series titles and resolution copy use ET."""

from __future__ import annotations

from datetime import date, datetime
import re
from zoneinfo import ZoneInfo

POLYMARKET_TZ = ZoneInfo("America/New_York")

_GROUP_TITLE_RE = re.compile(r"^([A-Za-z]+) (\d{1,2})(?:,\s*(\d{4}))?$")

_MONTH_FULL: dict[str, int] = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

def deadline_from_group_item_title(title: str | None) -> date | None:
    if title is None:
        return None
    m = _GROUP_TITLE_RE.fullmatch(title)
    if not m:
        return None
    month_name, day_s, year_s = m.group(1), m.group(2), m.group(3)
    month = _MONTH_FULL.get(month_name) or _MONTH_FULL.get(month_name.title())
    if month is None:
        return None
    day = int(day_s)
    if not 1 <= day <= 31:
        return None

    if year_s is not None:
        return date(int(year_s), month, day)

    today = datetime.now(POLYMARKET_TZ).date()
    return date(today.year, month, day)


def deadline_from_end_date_iso(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)

def deadline_from_none() -> date:
    today = datetime.now(POLYMARKET_TZ).date()
    return date(today.year + 1, today.month, today.day)


def series_calendar_date(dt: datetime) -> date:
    """Calendar date for series deadlines and slots (Eastern Time)."""
    return dt.astimezone(POLYMARKET_TZ).date()


def series_day_start_et(d: date) -> datetime:
    """First instant of ``d`` in US Eastern (period/slot start semantics)."""
    return datetime(d.year, d.month, d.day, tzinfo=POLYMARKET_TZ)


def series_day_end_et(d: date) -> datetime:
    """Last representable instant of ``d`` in US Eastern (resolution-day semantics)."""
    return datetime(d.year, d.month, d.day, 23, 59, 59, 999_999, tzinfo=POLYMARKET_TZ)
