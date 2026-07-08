"""``search`` command: screen open markets by CLOB best ask and write CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import httpx
import typer
from motor.motor_asyncio import AsyncIOMotorDatabase

from holder_boost.analysis.prop_raw import compute_prop_raw
from holder_boost.config import SearchSettings, get_settings
from holder_boost.logging import get_logger
from holder_boost.ml.inference import LoadedModel, load_artifacts, predict_outcome_prob
from holder_boost.models.markets import OpenMarket, ResolvedMarket
from holder_boost.models.search_match import SearchMatch
from holder_boost.polymarket.clob import get_orderbook
from holder_boost.polymarket.data import fetch_open_position_keys
from holder_boost.polymarket.gamma import fetch_open_markets
from holder_boost.storage.repos.markets import fetch_all_markets
from holder_boost.storage.repos.tags import fetch_all_tags

SEARCH_CSV_PATH = Path("csv/search.csv")

_CSV_COLUMNS = (
    "outcome",
    # "url",
    # "market",
    "event_title",
    "group_title",
    # "question",
    "ask",
    "spread",
    "prob",
    "prop_raw",
    "edge",
    "stop_loss",
    "neg_risk",
    "status",
    # "bid",
    "config",
)


def _passes_gamma_prefilter(market: OpenMarket, cfg: SearchSettings) -> bool:
    """Cheap Gamma pre-screen before CLOB ``/book``.

    Treats ``outcome_prices[outcome_index]`` as mid. Skips only when the band
    ``[mid - tol, mid + tol]`` (``tol = max(outcome_price_tolerance, max_spread)``)
    cannot overlap ``[min_price, max_price]``.
    """
    if not market.accepting_orders:
        return False
    try:
        mid = float(market.outcome_prices[cfg.outcome_index])
    except (IndexError, ValueError):
        return True
    tol = max(cfg.outcome_price_tolerance, cfg.max_spread)
    return mid + tol >= cfg.min_price and mid - tol <= cfg.max_price


def _price_cell(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 100:.2f}"


def _cent_price(price: float) -> int:
    return max(0, min(100, round(price * 100)))


def _model_prob(
    market: OpenMarket,
    *,
    price: float,
    loaded: LoadedModel,
) -> float:
    return predict_outcome_prob(
        tags=market.tags,
        price=_cent_price(price),
        loaded=loaded,
    )


def _position_status(
    market: OpenMarket,
    *,
    outcome_index: int,
    position_keys: set[tuple[str, int]],
) -> str:
    key = (market.condition_id, outcome_index)
    return "open" if key in position_keys else "—"


def _write_csv(
    path: Path,
    matches: list[SearchMatch],
    *,
    outcome_index: int,
    position_keys: set[tuple[str, int]],
    loaded: LoadedModel,
    resolved_markets: list[ResolvedMarket],
    tag_markets_count: dict[int, int],
) -> None:
    rows: list[tuple[float, SearchMatch, float, float | None, float | None]] = []
    for row in matches:
        market = row.market
        cent_price = _cent_price(row.ask)
        prob = _model_prob(market, price=row.ask, loaded=loaded)
        prop_raw, stop_loss = compute_prop_raw(
            open_tags=market.tags,
            cent_price=cent_price,
            outcome_index=outcome_index,
            resolved_markets=resolved_markets,
            tag_markets_count=tag_markets_count,
            neg_risk=market.neg_risk,
        )
        edge = (prop_raw or prob) - row.ask
        rows.append((edge, row, prob, prop_raw, stop_loss))
    rows.sort(key=lambda item: item[0], reverse=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for edge, row, prob, prop_raw, stop_loss in rows:
            market = row.market
            writer.writerow(
                {
                    "outcome": row.outcome_label,
                    # "url": f"https://polymarket.com/event/{market.event_slug}",
                    # "market": market.slug,
                    "event_title": market.event_title,
                    "group_title": market.group_item_title,
                    # "question": market.question,
                    "ask": _price_cell(row.ask),
                    "spread": _price_cell(row.spread),
                    "prob": _price_cell(prob),
                    "prop_raw": _price_cell(prop_raw),
                    "edge": _price_cell(edge),
                    "stop_loss": _price_cell(stop_loss),
                    "neg_risk": "yes" if market.neg_risk else "—",
                    "status": _position_status(
                        market,
                        outcome_index=outcome_index,
                        position_keys=position_keys,
                    ),
                    # "bid": _price_cell(row.bid),
                    "config": f"""
# {market.deadline_title()}
# {f"https://polymarket.com/event/{market.event_slug}/{market.slug}"}
[[stop_loss_bot.bots]]
condition_id = "{market.condition_id}"
                    """,
                }
            )


async def run_search(client: httpx.AsyncClient, db: AsyncIOMotorDatabase) -> None:
    """Load open markets, query CLOB sequentially, write ``csv/search.csv``."""
    log = get_logger("cli.search")
    settings = get_settings()
    cfg = settings.search
    poly = settings.polymarket

    try:
        loaded = load_artifacts(settings.train.model_dir, cfg.outcome_index)
    except FileNotFoundError as exc:
        log.error(
            "catboost model not found",
            error=str(exc),
            model_dir=settings.train.model_dir,
            outcome_index=cfg.outcome_index,
        )
        print(
            f"search: model not found for outcome_index={cfg.outcome_index} "
            f"in {settings.train.model_dir} — run `holder-boost train` first",
        )
        raise typer.Exit(code=1) from exc

    position_keys = await fetch_open_position_keys(client, user=poly.proxy_wallet)

    markets = await fetch_open_markets(
        client,
        tag_ids=cfg.tag_ids if cfg.tag_ids else None,
        deadline_in_days=cfg.deadline_in_days,
    )
    matches: list[SearchMatch] = []
    pre_skipped = 0
    checked = 0

    for market in markets:
        if not _passes_gamma_prefilter(market, cfg):
            pre_skipped += 1
            continue
        checked += 1
        book = await get_orderbook(client, market.clob_token_ids[cfg.outcome_index])
        match = SearchMatch.try_from_orderbook(
            market,
            book,
            outcome_index=cfg.outcome_index,
            min_price=cfg.min_price,
            max_price=cfg.max_price,
            max_spread=cfg.max_spread,
        )
        if match is not None:
            matches.append(match)

    tags = await fetch_all_tags(db)
    tag_markets_count = {tag.id: tag.markets_count for tag in tags}
    resolved_markets = await fetch_all_markets(db)

    _write_csv(
        SEARCH_CSV_PATH,
        matches,
        outcome_index=cfg.outcome_index,
        position_keys=position_keys,
        loaded=loaded,
        resolved_markets=resolved_markets,
        tag_markets_count=tag_markets_count,
    )
    log.info(
        "search complete",
        markets=len(markets),
        positions=len(position_keys),
        pre_skipped=pre_skipped,
        checked=checked,
        matched=len(matches),
        path=str(SEARCH_CSV_PATH),
    )
    print(
        f"search: markets={len(markets)} positions={len(position_keys)} "
        f"pre_skipped={pre_skipped} checked={checked} matched={len(matches)} "
        f"wrote {SEARCH_CSV_PATH}"
    )
