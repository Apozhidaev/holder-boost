"""Structlog event projected into a Mongo ``logs`` document."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class LogEntry(BaseModel):
    """One row in the ``logs`` collection.

    ``ts`` MUST be a tz-aware UTC ``datetime`` so motor serializes it as a
    BSON ``Date`` — this is what TTL indexes operate on. Extra structlog
    bind fields land in ``extra`` and are flattened into the top level on
    persistence.
    """

    model_config = ConfigDict(extra="allow")

    ts: datetime
    level: str
    msg: str
    source: str
