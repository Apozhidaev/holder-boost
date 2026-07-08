"""Repository for the ``tags`` Mongo collection (resolved sync)."""

from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING

from holder_boost.models.tag import Tag
from holder_boost.storage.client import TAGS_COLLECTION


def _to_doc(tag: Tag) -> dict:
    return tag.model_dump(mode="python")


async def insert_tag_if_absent(
    db: AsyncIOMotorDatabase,
    tag: Tag,
) -> bool:
    """Insert ``tag`` when no document exists for ``id``; skip if present."""
    coll = db[TAGS_COLLECTION]
    result = await coll.update_one(
        {"id": tag.id},
        {"$setOnInsert": _to_doc(tag)},
        upsert=True,
    )
    return result.upserted_id is not None


async def fetch_all_tags(db: AsyncIOMotorDatabase) -> list[Tag]:
    """Return all tags in ``tags``, sorted by ``id`` ascending."""
    coll = db[TAGS_COLLECTION]
    cursor = coll.find({}).sort("id", ASCENDING)
    out: list[Tag] = []
    async for doc in cursor:
        doc = dict(doc)
        doc.pop("_id", None)
        out.append(Tag.model_validate(doc))
    return out
