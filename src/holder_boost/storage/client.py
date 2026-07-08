"""Motor (async MongoDB) client + index bootstrap.

The CLI calls :func:`get_database` once at command start. Connection failures
surface immediately with a friendly message — Mongo is mandatory.
"""

from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING

from holder_boost.config import MongoSettings, get_settings

MARKETS_COLLECTION = "markets"
PREDICTS_COLLECTION = "predicts"
TAGS_COLLECTION = "tags"
LOGS_COLLECTION = "logs"


class MongoConnectionError(RuntimeError):
    """Raised when Mongo is unreachable at startup."""


_client: AsyncIOMotorClient | None = None


def _get_client(mongo: MongoSettings) -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(
            mongo.uri, tz_aware=True, uuidRepresentation="standard"
        )
    return _client


async def get_database() -> AsyncIOMotorDatabase:
    """Return the Mongo database, validating connectivity on first call."""
    settings = get_settings()
    client = _get_client(settings.mongo)
    try:
        await client.admin.command("ping")
    except Exception as exc:
        raise MongoConnectionError(
            f"failed to connect to MongoDB at {settings.mongo.uri}: {exc}"
        ) from exc
    return client[settings.mongo.db]


async def ensure_indexes(db: AsyncIOMotorDatabase) -> None:
    """Create indexes if they don't already exist (events, logs)."""
    settings = get_settings()
    ttl_seconds = settings.logging.mongo_ttl_days * 86400

    await db[MARKETS_COLLECTION].create_index(
        [("id", ASCENDING)],
        unique=True,
        name="markets_id_unique",
    )
    await db[MARKETS_COLLECTION].create_index(
        [("tags", ASCENDING), ("neg_risk", ASCENDING)],
        name="markets_tags_neg_risk",
    )
    await db[PREDICTS_COLLECTION].create_index(
        [("tag_id", ASCENDING)],
        unique=True,
        name="predicts_tag_id_unique",
    )
    await db[TAGS_COLLECTION].create_index(
        [("id", ASCENDING)],
        unique=True,
        name="tags_id_unique",
    )
    await db[LOGS_COLLECTION].create_index(
        [("ts", ASCENDING)],
        name="ts_ttl",
        expireAfterSeconds=ttl_seconds,
    )
    await db[LOGS_COLLECTION].create_index(
        [("source", ASCENDING), ("ts", DESCENDING)],
        name="source_ts",
    )
