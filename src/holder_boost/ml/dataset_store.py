"""Save and load training frames as parquet + metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from holder_boost.ml.artifacts import utc_now_iso

TRAIN_PARQUET_FILENAME = "train.parquet"
METADATA_FILENAME = "metadata.json"
LABEL_COLUMN = "resolved_outcome_index"


@dataclass(frozen=True)
class DatasetMetadata:
    tag_id: int
    min_wallet_count: int
    frequent_wallets: list[str]
    feature_names: list[str]
    n_rows: int
    built_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag_id": self.tag_id,
            "min_wallet_count": self.min_wallet_count,
            "frequent_wallets": self.frequent_wallets,
            "feature_names": self.feature_names,
            "n_rows": self.n_rows,
            "built_at": self.built_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetMetadata:
        return cls(
            tag_id=int(data["tag_id"]),
            min_wallet_count=int(data["min_wallet_count"]),
            frequent_wallets=[str(wallet) for wallet in data["frequent_wallets"]],
            feature_names=list(data["feature_names"]),
            n_rows=int(data["n_rows"]),
            built_at=str(data["built_at"]),
        )


def dataset_dir_for_tag(base_dir: Path | str, tag_id: int) -> Path:
    return Path(base_dir) / str(tag_id)


def train_parquet_path(dataset_dir: Path) -> Path:
    return dataset_dir / TRAIN_PARQUET_FILENAME


def dataset_metadata_path(dataset_dir: Path) -> Path:
    return dataset_dir / METADATA_FILENAME


def save_training_frame(
    dataset_dir: Path,
    *,
    X: pd.DataFrame,
    y: pd.Series,
    frequent_wallets: list[str],
    tag_id: int,
    min_wallet_count: int,
) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    frame = X.copy()
    frame[LABEL_COLUMN] = y
    frame.to_parquet(train_parquet_path(dataset_dir), index=False)

    metadata = DatasetMetadata(
        tag_id=tag_id,
        min_wallet_count=min_wallet_count,
        frequent_wallets=frequent_wallets,
        feature_names=list(X.columns),
        n_rows=len(X),
        built_at=utc_now_iso(),
    )
    dataset_metadata_path(dataset_dir).write_text(
        json.dumps(metadata.to_dict(), indent=2),
        encoding="utf-8",
    )


def load_training_frame(
    dataset_dir: Path,
    *,
    expected_tag_id: int,
    expected_min_wallet_count: int,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    parquet_path = train_parquet_path(dataset_dir)
    metadata_path = dataset_metadata_path(dataset_dir)
    if not parquet_path.is_file():
        raise FileNotFoundError(
            f"training dataset not found: {parquet_path} (run `holder-boost dataset` first)"
        )
    if not metadata_path.is_file():
        raise FileNotFoundError(f"dataset metadata not found: {metadata_path}")

    metadata = DatasetMetadata.from_dict(
        json.loads(metadata_path.read_text(encoding="utf-8"))
    )
    if metadata.tag_id != expected_tag_id:
        raise ValueError(
            f"dataset tag_id={metadata.tag_id} does not match config tag_id={expected_tag_id}"
        )
    if metadata.min_wallet_count != expected_min_wallet_count:
        raise ValueError(
            "dataset min_wallet_count="
            f"{metadata.min_wallet_count} does not match config "
            f"min_wallet_count={expected_min_wallet_count}"
        )

    frame = pd.read_parquet(parquet_path)
    if LABEL_COLUMN not in frame.columns:
        raise ValueError(f"dataset missing label column {LABEL_COLUMN!r}")

    y = frame.pop(LABEL_COLUMN)
    X = frame
    if list(X.columns) != metadata.feature_names:
        raise ValueError("dataset feature columns do not match metadata.feature_names")

    return X, y, metadata.frequent_wallets
