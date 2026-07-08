"""Repository for the ``markets`` Mongo collection (resolved sync)."""

from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorDatabase

from holder_boost.models.markets import HolderDistribution, ResolvedMarket
from holder_boost.storage.client import MARKETS_COLLECTION


def _holders_to_doc(holders: list[HolderDistribution]) -> list[dict]:
    return [
        {
            "proxy_wallet": holder.proxy_wallet,
            "distribution": {str(cent): count for cent, count in holder.distribution.items()},
        }
        for holder in holders
    ]


def _holders_from_doc(raw: object) -> list[HolderDistribution]:
    if not isinstance(raw, list):
        return []
    out: list[HolderDistribution] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        distribution_raw = item.get("distribution")
        distribution: dict[int, int] = {}
        if isinstance(distribution_raw, dict):
            distribution = {int(cent): int(count) for cent, count in distribution_raw.items()}
        out.append(
            HolderDistribution(
                proxy_wallet=str(item["proxy_wallet"]),
                distribution=distribution,
            )
        )
    return out


def _to_doc(market: ResolvedMarket) -> dict:
    doc = market.model_dump(mode="python")
    doc["holders"] = _holders_to_doc(market.holders)
    return doc


def _from_doc(doc: dict) -> ResolvedMarket:
    payload = dict(doc)
    payload.pop("_id", None)
    payload.pop("trade_cent_prices", None)
    payload.pop("stop_prices", None)
    if "holders" in payload:
        payload["holders"] = _holders_from_doc(payload["holders"])
    return ResolvedMarket.model_validate(payload)


async def market_exists(db: AsyncIOMotorDatabase, market_id: int) -> bool:
    """Return whether a document exists for Gamma market ``id``."""
    coll = db[MARKETS_COLLECTION]
    doc = await coll.find_one({"id": market_id}, projection={"_id": 1})
    return doc is not None


async def insert_market_if_absent(
    db: AsyncIOMotorDatabase,
    market: ResolvedMarket,
) -> bool:
    """Insert ``market`` when no document exists for ``id``; skip if present."""
    coll = db[MARKETS_COLLECTION]
    result = await coll.update_one(
        {"id": market.id},
        {"$setOnInsert": _to_doc(market)},
        upsert=True,
    )
    return result.upserted_id is not None


async def fetch_markets_by_tag_id(
    db: AsyncIOMotorDatabase,
    tag_id: int,
    *,
    neg_risk: bool,
) -> list[ResolvedMarket]:
    """Return resolved markets whose ``tags`` array contains *tag_id* and ``neg_risk`` matches."""
    coll = db[MARKETS_COLLECTION]
    cursor = coll.find({"tags": tag_id, "neg_risk": neg_risk})
    out: list[ResolvedMarket] = []
    async for doc in cursor:
        out.append(_from_doc(doc))
    return out


async def fetch_all_markets(
    db: AsyncIOMotorDatabase,
    *,
    neg_risk: bool | None = None,
) -> list[ResolvedMarket]:
    """Return resolved markets; filter by ``neg_risk`` when provided."""
    coll = db[MARKETS_COLLECTION]
    query = {"neg_risk": neg_risk} if neg_risk is not None else {}
    cursor = coll.find(query)
    out: list[ResolvedMarket] = []
    async for doc in cursor:
        out.append(_from_doc(doc))
    return out
