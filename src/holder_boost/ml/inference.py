"""CatBoost inference on holder-distribution rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

from holder_boost.ml.artifacts import load_metadata, load_model, load_wallets
from holder_boost.ml.features import feature_columns, row_is_all_missing, wallet_values_for_row


@dataclass(frozen=True)
class LoadedModel:
    model: CatBoostClassifier
    frequent_wallets: list[str]
    feature_names: list[str]


def load_model_artifacts(model_dir: Path) -> LoadedModel:
    model = load_model(model_dir)
    frequent_wallets = load_wallets(model_dir)
    metadata = load_metadata(model_dir)
    return LoadedModel(
        model=model,
        frequent_wallets=frequent_wallets,
        feature_names=metadata.feature_names,
    )


def build_predict_row(
    *,
    days_left: int,
    holders: dict[str, int],
    frequent_wallets: list[str],
) -> pd.DataFrame:
    wallet_values = wallet_values_for_row(
        frequent_wallets=frequent_wallets,
        holders=holders,
    )
    if row_is_all_missing(wallet_values):
        raise ValueError("all wallet features are -1")

    row = {"days_left": days_left, **wallet_values}
    columns = feature_columns(frequent_wallets)
    return pd.DataFrame([{col: row[col] for col in columns}])


def predict_probs(
    loaded: LoadedModel,
    *,
    days_left: int,
    holders: dict[str, int],
) -> list[float]:
    X = build_predict_row(
        days_left=days_left,
        holders=holders,
        frequent_wallets=loaded.frequent_wallets,
    )
    X = X[loaded.feature_names]
    probs = loaded.model.predict_proba(X)[0]
    return [float(probs[0]), float(probs[1])]
