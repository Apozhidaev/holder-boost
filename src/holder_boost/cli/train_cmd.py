"""``train`` command: CatBoost model from saved training dataset."""

from __future__ import annotations

from holder_boost.config import get_settings
from holder_boost.logging import get_logger
from holder_boost.ml.artifacts import model_dir_for_tag, save_artifacts
from holder_boost.ml.dataset_store import dataset_dir_for_tag, load_training_frame
from holder_boost.ml.trainer import train_model_from_frame


def run_train(*, skip_tuning: bool = False) -> None:
    """Load training parquet and train CatBoost; save model artifacts."""
    log = get_logger("cli.train")
    cfg = get_settings().train
    model_dir = model_dir_for_tag(cfg.model_dir, cfg.tag_id)
    dataset_dir = dataset_dir_for_tag(cfg.dataset_dir, cfg.tag_id)

    X, y, frequent_wallets = load_training_frame(
        dataset_dir,
        expected_tag_id=cfg.tag_id,
        expected_min_wallet_count=cfg.min_wallet_count,
    )
    log.info(
        "train data loaded",
        tag_id=cfg.tag_id,
        dataset_dir=str(dataset_dir),
        rows=len(X),
        features=len(X.columns),
        wallets=len(frequent_wallets),
    )

    result = train_model_from_frame(
        X,
        y,
        frequent_wallets,
        cfg,
        skip_tuning=skip_tuning,
    )
    save_artifacts(
        model_dir,
        model=result.model,
        frequent_wallets=result.frequent_wallets,
        metadata=result.metadata,
        feature_importance_csv=result.feature_importance_csv,
    )

    auc_msg = (
        f" auc={result.metadata.best_cv_auc:.5f}"
        if result.metadata.best_cv_auc is not None
        else ""
    )
    summary = (
        f"rows={result.metadata.n_rows} "
        f"features={result.metadata.n_features} "
        f"wallets={len(result.metadata.frequent_wallets)}"
        f"{auc_msg}"
    )
    log.info(
        "train complete",
        model_dir=str(model_dir),
        rows=result.metadata.n_rows,
        features=result.metadata.n_features,
        wallets=len(result.metadata.frequent_wallets),
        best_cv_auc=result.metadata.best_cv_auc,
    )

    print(f"train: tag_id={cfg.tag_id} saved={model_dir}")
    print(f"  {summary}")
