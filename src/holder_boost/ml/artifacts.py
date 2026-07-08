"""Save and load CatBoost model artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
from catboost import CatBoostClassifier

MODEL_FILENAME = "catboost.cbm"
WALLETS_FILENAME = "wallets.pkl"
METADATA_FILENAME = "metadata.json"
FEATURE_IMPORTANCE_FILENAME = "feature_importance.csv"


@dataclass(frozen=True)
class ModelMetadata:
    best_params: dict[str, Any]
    frequent_wallets: list[str]
    feature_names: list[str]
    best_cv_auc: float | None
    n_rows: int
    n_features: int
    trained_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_params": self.best_params,
            "frequent_wallets": self.frequent_wallets,
            "feature_names": self.feature_names,
            "best_cv_auc": self.best_cv_auc,
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "trained_at": self.trained_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        return cls(
            best_params=dict(data["best_params"]),
            frequent_wallets=[str(wallet) for wallet in data["frequent_wallets"]],
            feature_names=list(data["feature_names"]),
            best_cv_auc=data.get("best_cv_auc"),
            n_rows=int(data["n_rows"]),
            n_features=int(data["n_features"]),
            trained_at=str(data["trained_at"]),
        )


def model_path(model_dir: Path) -> Path:
    return model_dir / MODEL_FILENAME


def wallets_path(model_dir: Path) -> Path:
    return model_dir / WALLETS_FILENAME


def metadata_path(model_dir: Path) -> Path:
    return model_dir / METADATA_FILENAME


def feature_importance_path(model_dir: Path) -> Path:
    return model_dir / FEATURE_IMPORTANCE_FILENAME


def save_artifacts(
    model_dir: Path,
    *,
    model: CatBoostClassifier,
    frequent_wallets: list[str],
    metadata: ModelMetadata,
    feature_importance_csv: str | None = None,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path(model_dir)))
    joblib.dump(frequent_wallets, wallets_path(model_dir))
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


def load_wallets(model_dir: Path) -> list[str]:
    path = wallets_path(model_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Wallet list not found: {path}")
    return joblib.load(path)


def load_metadata(model_dir: Path) -> ModelMetadata:
    path = metadata_path(model_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Model metadata not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return ModelMetadata.from_dict(data)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()
