"""CatBoost training with optional Optuna hyperparameter search."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from holder_boost.config import TrainSettings
from holder_boost.ml.artifacts import ModelMetadata, utc_now_iso
from holder_boost.ml.dataset import build_training_frame
from holder_boost.models.markets import ResolvedMarket

warnings.filterwarnings("ignore", category=UserWarning)


DEFAULT_PARAMS: dict[str, Any] = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "bagging_temperature": 1.0,
    "border_count": 128,
    "grow_policy": "SymmetricTree",
    "min_data_in_leaf": 10,
    "bootstrap_type": "Bayesian",
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "verbose": False,
    "thread_count": -1,
    "early_stopping_rounds": 100,
}


@dataclass(frozen=True)
class TrainResult:
    model: CatBoostClassifier
    frequent_wallets: list[str]
    metadata: ModelMetadata
    feature_importance_csv: str


def _base_params(cfg: TrainSettings) -> dict[str, Any]:
    return {
        **DEFAULT_PARAMS,
        "random_seed": cfg.random_state,
    }


def _objective(
    trial: optuna.Trial,
    *,
    X: pd.DataFrame,
    y: pd.Series,
    cfg: TrainSettings,
) -> float:
    params = {
        "iterations": trial.suggest_int("iterations", 700, 2200),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.12, log=True),
        "depth": trial.suggest_int("depth", 5, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 25.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "border_count": trial.suggest_int("border_count", 64, 255),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide"]),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 3, 80),
        "bootstrap_type": "Bayesian",
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": cfg.random_state,
        "verbose": False,
        "thread_count": -1,
        "early_stopping_rounds": 100,
    }

    skf = StratifiedKFold(
        n_splits=cfg.n_folds,
        shuffle=True,
        random_state=cfg.random_state,
    )
    auc_scores: list[float] = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            verbose=False,
        )

        preds = model.predict_proba(X_val)[:, 1]
        auc_scores.append(roc_auc_score(y_val, preds))

    return float(np.mean(auc_scores))


def _tune_hyperparams(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: TrainSettings,
) -> tuple[dict[str, Any], float]:
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        lambda trial: _objective(trial, X=X, y=y, cfg=cfg),
        n_trials=cfg.n_trials,
        show_progress_bar=True,
    )
    return study.best_params, float(study.best_value)


def _feature_importance_csv(model: CatBoostClassifier, feature_names: list[str]) -> str:
    importance = model.feature_importances_
    rows = sorted(
        zip(feature_names, importance, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    lines = ["feature,importance"]
    lines.extend(f"{name},{value}" for name, value in rows)
    return "\n".join(lines) + "\n"


def train_model(
    markets: list[ResolvedMarket],
    cfg: TrainSettings,
    *,
    skip_tuning: bool = False,
) -> TrainResult:
    """Train CatBoost to predict ``resolved_outcome_index`` from holder distribution rows."""
    X, y, frequent_wallets = build_training_frame(
        markets,
        min_wallet_count=cfg.min_wallet_count,
    )

    best_cv_auc: float | None = None
    if skip_tuning:
        best_params = _base_params(cfg)
    else:
        tuned, best_cv_auc = _tune_hyperparams(X, y, cfg)
        best_params = {**_base_params(cfg), **tuned}

    final_params = {
        **best_params,
        "verbose": 200,
    }
    final_model = CatBoostClassifier(**final_params)
    final_model.fit(X, y)

    feature_names = list(X.columns)
    metadata = ModelMetadata(
        best_params=best_params,
        frequent_wallets=frequent_wallets,
        feature_names=feature_names,
        best_cv_auc=best_cv_auc,
        n_rows=len(X),
        n_features=len(feature_names),
        trained_at=utc_now_iso(),
    )
    return TrainResult(
        model=final_model,
        frequent_wallets=frequent_wallets,
        metadata=metadata,
        feature_importance_csv=_feature_importance_csv(final_model, feature_names),
    )
