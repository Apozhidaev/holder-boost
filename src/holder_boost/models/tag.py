"""Domain models for Polymarket tags."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Tag(BaseModel):
    """One Polymarket tag."""

    model_config = ConfigDict(frozen=True)

    id: int
    slug: str
    label: str
    markets_count: int = Field(default=0)
