"""Repository for the ``predicts`` Mongo collection."""

from __future__ import annotations

from collections.abc import Iterable

from motor.motor_asyncio import AsyncIOMotorDatabase

from holder_boost.models.predict import Predict
from holder_boost.storage.client import PREDICTS_COLLECTION


async def fetch_predicts_by_tag_ids(
    db: AsyncIOMotorDatabase,
    tag_ids: Iterable[int],
) -> dict[int, Predict]:
    """Return ``tag_id`` → :class:`Predict` for tags that have a stored row."""
    ids = list(tag_ids)
    if not ids:
        return {}
    coll = db[PREDICTS_COLLECTION]
    out: dict[int, Predict] = {}
    async for doc in coll.find({"tag_id": {"$in": ids}}):
        row = Predict.model_validate(doc)
        out[row.tag_id] = row
    return out


async def upsert_predict(db: AsyncIOMotorDatabase, row: Predict) -> None:
    """Replace or insert the document for ``tag_id`` (latest predict wins)."""
    coll = db[PREDICTS_COLLECTION]
    doc = row.model_dump(mode="python")
    await coll.replace_one({"tag_id": row.tag_id}, doc, upsert=True)
