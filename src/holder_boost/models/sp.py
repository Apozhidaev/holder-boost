"""Generic time series models."""

from datetime import datetime

from pydantic import BaseModel


class TimeSeriesPoint(BaseModel):
    """Single point in a time series (timestamp, value)."""

    timestamp: datetime
    value: float


class SeriesPoint(BaseModel):
    """Single point in a series (x, y)."""

    x: float
    y: float
