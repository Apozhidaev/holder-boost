"""Build training DataFrame from resolved markets."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from holder_boost.models.markets import ResolvedMarket


def _expand_rows(
    markets: Sequence[ResolvedMarket],
    *,
    outcome_index: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for market in markets:
        try:
            prices = market.trade_cent_prices[outcome_index]
        except IndexError:
            continue
        won = market.resolved_outcome_index == outcome_index
        for price in prices:
            rows.append(
                {
                    "tags": market.tags,
                    "price": price,
                    "won": int(won),
                }
            )
    return rows


def _frequent_tags(rows: list[dict[str, object]], min_tag_count: int) -> list[int]:
    counts: dict[int, int] = {}
    for row in rows:
        for tag_id in row["tags"]:  # type: ignore[union-attr]
            counts[tag_id] = counts.get(tag_id, 0) + 1
    return sorted(tag_id for tag_id, count in counts.items() if count >= min_tag_count)


def build_training_frame(
    markets: Sequence[ResolvedMarket],
    *,
    outcome_index: int,
    min_tag_count: int,
) -> tuple[pd.DataFrame, pd.Series, MultiLabelBinarizer, list[int]]:
    """Return ``(X, y, mlb, frequent_tags)`` for CatBoost training."""
    rows = _expand_rows(markets, outcome_index=outcome_index)
    if not rows:
        raise ValueError("no training rows: markets have no take-prices for outcome_index")

    frequent = _frequent_tags(rows, min_tag_count)
    if not frequent:
        raise ValueError(
            f"no frequent tags after min_tag_count={min_tag_count} filter",
        )

    tags_list = [row["tags"] for row in rows]
    mlb = MultiLabelBinarizer(classes=frequent)
    tag_matrix = mlb.fit_transform(tags_list)
    tag_columns = [f"tag_{t}" for t in mlb.classes_]
    tag_df = pd.DataFrame(tag_matrix, columns=tag_columns)

    df = pd.concat(
        [
            pd.DataFrame({"price": [row["price"] for row in rows]}),
            tag_df,
        ],
        axis=1,
    )
    y = pd.Series([row["won"] for row in rows], dtype=int, name="won")
    return df, y, mlb, frequent
