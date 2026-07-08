"""Load trained CatBoost artifacts and predict outcome probability."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from holder_boost.ml.artifacts import load_metadata, load_mlb, load_model, outcome_model_dir


@dataclass
class LoadedModel:
    model: CatBoostClassifier
    mlb: MultiLabelBinarizer
    model_dir: Path
    outcome_index: int


_cache: dict[tuple[Path, int], LoadedModel] = {}


def load_artifacts(model_dir: str | Path, outcome_index: int) -> LoadedModel:
    """Load model, MLB and metadata for *outcome_index*; cache per process."""
    base = Path(model_dir).resolve()
    path = outcome_model_dir(base, outcome_index).resolve()
    key = (base, outcome_index)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    metadata = load_metadata(path)
    loaded = LoadedModel(
        model=load_model(path),
        mlb=load_mlb(path),
        model_dir=path,
        outcome_index=metadata.outcome_index,
    )
    _cache[key] = loaded
    return loaded


def predict_outcome_prob(
    *,
    tags: list[int],
    price: int,
    loaded: LoadedModel,
) -> float:
    """Return P(outcome wins) for the trained ``outcome_index``."""
    tag_vector = loaded.mlb.transform([tags])
    feature_names = [f"tag_{t}" for t in loaded.mlb.classes_]
    X_new = pd.DataFrame(tag_vector, columns=feature_names)
    X_new["price"] = price
    X_new = X_new[loaded.model.feature_names_]
    proba = loaded.model.predict_proba(X_new)[0, 1]
    return round(float(proba), 4)
