"""Save and load CatBoost model artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
from catboost import CatBoostClassifier
from sklearn.preprocessing import MultiLabelBinarizer

MODEL_FILENAME = "catboost.cbm"
MLB_FILENAME = "mlb.pkl"
METADATA_FILENAME = "metadata.json"
FEATURE_IMPORTANCE_FILENAME = "feature_importance.csv"

OUTCOME_INDICES = (0, 1)


def outcome_model_dir(base_dir: Path, outcome_index: int) -> Path:
    """Per-outcome artifact directory under ``base_dir``."""
    return base_dir / f"outcome_{outcome_index}"


@dataclass(frozen=True)
class ModelMetadata:
    outcome_index: int
    best_params: dict[str, Any]
    frequent_tags: list[int]
    feature_names: list[str]
    best_cv_auc: float | None
    n_rows: int
    n_features: int
    trained_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome_index": self.outcome_index,
            "best_params": self.best_params,
            "frequent_tags": self.frequent_tags,
            "feature_names": self.feature_names,
            "best_cv_auc": self.best_cv_auc,
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "trained_at": self.trained_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        return cls(
            outcome_index=int(data["outcome_index"]),
            best_params=dict(data["best_params"]),
            frequent_tags=[int(t) for t in data["frequent_tags"]],
            feature_names=list(data["feature_names"]),
            best_cv_auc=data.get("best_cv_auc"),
            n_rows=int(data["n_rows"]),
            n_features=int(data["n_features"]),
            trained_at=str(data["trained_at"]),
        )


def model_path(model_dir: Path) -> Path:
    return model_dir / MODEL_FILENAME


def mlb_path(model_dir: Path) -> Path:
    return model_dir / MLB_FILENAME


def metadata_path(model_dir: Path) -> Path:
    return model_dir / METADATA_FILENAME


def feature_importance_path(model_dir: Path) -> Path:
    return model_dir / FEATURE_IMPORTANCE_FILENAME


def save_artifacts(
    model_dir: Path,
    *,
    model: CatBoostClassifier,
    mlb: MultiLabelBinarizer,
    metadata: ModelMetadata,
    feature_importance_csv: str | None = None,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path(model_dir)))
    joblib.dump(mlb, mlb_path(model_dir))
    metadata_path(model_dir).write_text(
        json.dumps(metadata.to_dict(), indent=2),
        encoding="utf-8",
    )
    if feature_importance_csv is not None:
        feature_importance_path(model_dir).write_text(
            feature_importance_csv,
            encoding="utf-8",
        )


def load_model(model_dir: Path) -> CatBoostClassifier:
    path = model_path(model_dir)
    if not path.is_file():
        raise FileNotFoundError(f"CatBoost model not found: {path}")
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def load_mlb(model_dir: Path) -> MultiLabelBinarizer:
    path = mlb_path(model_dir)
    if not path.is_file():
        raise FileNotFoundError(f"MultiLabelBinarizer not found: {path}")
    return joblib.load(path)


def load_metadata(model_dir: Path) -> ModelMetadata:
    path = metadata_path(model_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Model metadata not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return ModelMetadata.from_dict(data)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()
