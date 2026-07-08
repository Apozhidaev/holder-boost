"""``search-holder`` command: holders with private usernames on open markets."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, TypedDict

import httpx

from holder_boost.config import get_settings
from holder_boost.logging import get_logger
from holder_boost.models.markets import OpenMarket
from holder_boost.polymarket.data import fetch_holders
from holder_boost.polymarket.gamma import fetch_open_markets

SEARCH_HOLDER_CSV_PATH = Path("csv/search_holder.csv")

_CSV_COLUMNS: tuple[str, ...] = (
    "url",
    "name",
    "proxyWallet",
    "amount",
    "outcomeIndex",
    "neg_risk",
)


class SearchHolderCsvRow(TypedDict):
    url: str
    name: str
    proxyWallet: str
    amount: str
    outcomeIndex: str
    neg_risk: str


def _market_url(market: OpenMarket) -> str:
    return f"https://polymarket.com/event/{market.event_slug}/{market.slug}"


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _is_private_username(holder: dict[str, Any]) -> bool:
    return holder.get("displayUsernamePublic") is True


def _matches_proxy_wallet(holder: dict[str, Any], proxy_wallet: str | None) -> bool:
    if not proxy_wallet:
        return True
    wallet = holder.get("proxyWallet")
    if wallet is None:
        return False
    return str(wallet).lower() == proxy_wallet.lower()


def _holder_row(
    url: str,
    holder: dict[str, Any],
    *,
    neg_risk: bool,
    proxy_wallet: str | None,
) -> SearchHolderCsvRow | None:
    if not _is_private_username(holder) or not _matches_proxy_wallet(holder, proxy_wallet):
        return None
    return {
        "url": url,
        "name": _cell(holder.get("name")),
        "proxyWallet": _cell(holder.get("proxyWallet")),
        "amount": _cell(holder.get("amount")),
        "outcomeIndex": _cell(holder.get("outcomeIndex")),
        "neg_risk": "yes" if neg_risk else "—",
    }


def _write_csv(path: Path, rows: list[SearchHolderCsvRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


async def run_search_holder(client: httpx.AsyncClient) -> None:
    """Load open markets, fetch holders with ``displayUsernamePublic == false``, write CSV."""
    log = get_logger("cli.search_holder")
    cfg = get_settings().search_holder

    markets = await fetch_open_markets(
        client,
        tag_ids=cfg.tag_ids if cfg.tag_ids else None,
        deadline_in_days=cfg.deadline_in_days,
    )

    csv_rows: list[SearchHolderCsvRow] = []
    holders_total = 0
    markets_total = len(markets)

    for idx, market in enumerate(markets, start=1):
        holders = await fetch_holders(client, market.condition_id)
        holders_total += len(holders)
        url = _market_url(market)
        matched = 0
        for holder in holders:
            row = _holder_row(
                url,
                holder,
                neg_risk=market.neg_risk,
                proxy_wallet=cfg.proxy_wallet,
            )
            if row is not None:
                csv_rows.append(row)
                matched += 1
        log.info(
            "holders loaded",
            market=idx,
            markets_total=markets_total,
            slug=market.slug,
            holders=len(holders),
            matched=matched,
        )

    _write_csv(SEARCH_HOLDER_CSV_PATH, csv_rows)
    log.info(
        "search-holder complete",
        markets=len(markets),
        holders=holders_total,
        matched_rows=len(csv_rows),
        proxy_wallet=cfg.proxy_wallet,
        path=str(SEARCH_HOLDER_CSV_PATH),
    )
    print(
        f"search-holder: markets={len(markets)} holders={holders_total} "
        f"matched_rows={len(csv_rows)} wrote {SEARCH_HOLDER_CSV_PATH}"
    )
