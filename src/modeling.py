"""Modelling utilities for the Credit Risk Scoring project.

Bu modül `feature_engineering.py`'deki agg fonksiyonlarına benzer şekilde
saf, yan-etkisiz yardımcılardan oluşur. Hedef: notebook'lar yalnızca
orchestration yapsın, asıl mantık burada test edilebilir biçimde dursun.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ks_2samp
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    brier_score_loss,
    log_loss,
)
from sklearn.calibration import calibration_curve

import lightgbm as lgb


DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 64,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 100,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbosity": -1,
    "random_state": 42,
}


def _sanitize_feature_names(columns: pd.Index) -> list[str]:
    """Return LightGBM-safe and unique feature names.

    LightGBM rejects feature names containing JSON-special characters.
    We normalize names to ASCII-friendly tokens and deduplicate collisions.
    """
    sanitized: list[str] = []
    counts: dict[str, int] = {}

    for raw_col in columns:
        col = str(raw_col)
        # Replace characters that can break LightGBM's JSON serialization.
        safe = re.sub(r'[\{\}\[\]":,\\]', "_", col)
        safe = re.sub(r"\s+", "_", safe).strip("_")
        safe = re.sub(r"_+", "_", safe)
        if not safe:
            safe = "feature"

        idx = counts.get(safe, 0)
        counts[safe] = idx + 1
        sanitized.append(safe if idx == 0 else f"{safe}_{idx}")

    return sanitized


def prepare_features(
    df: pd.DataFrame,
    *,
    target: str = "TARGET",
    id_cols: tuple[str, ...] = ("SK_ID_CURR",),
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Split a modelling frame into (X, y, feature_names).

    Tüm `category`/`string` dtype'lı sütunlar `category`'e çevrilir; bu sayede
    LightGBM kategorik feature'ları otomatik algılar.
    """
    drop = list(id_cols) + [target]
    feature_cols = [c for c in df.columns if c not in drop]
    X = df[feature_cols].copy()
    X.columns = _sanitize_feature_names(X.columns)
    y = df[target].astype(np.int8)

    for col in X.columns:
        if pd.api.types.is_string_dtype(X[col]) and not isinstance(X[col].dtype, pd.CategoricalDtype):
            X[col] = X[col].astype("category")

    return X, y, list(X.columns)


def prepare_features_for_inference(
    df: pd.DataFrame,
    *,
    feature_names: list[str] | None = None,
    id_cols: tuple[str, ...] = ("SK_ID_CURR",),
    target: str = "TARGET",
) -> pd.DataFrame:
    """Build a feature matrix ready for `Booster.predict`.

    Mirrors the column transforms applied in :func:`prepare_features` (sanitize
    + categorical coercion) but tolerates inputs that lack the target column,
    and — when ``feature_names`` is provided — reorders / restricts columns to
    that exact list so the downstream booster sees them in training order.
    """
    drop = list(id_cols)
    if target in df.columns:
        drop.append(target)

    cols = [c for c in df.columns if c not in drop]
    X = df[cols].copy()
    X.columns = _sanitize_feature_names(X.columns)

    for col in X.columns:
        if pd.api.types.is_string_dtype(X[col]) and not isinstance(X[col].dtype, pd.CategoricalDtype):
            X[col] = X[col].astype("category")

    if feature_names is not None:
        missing = [f for f in feature_names if f not in X.columns]
        if missing:
            raise ValueError(
                f"Input is missing {len(missing)} expected features. "
                f"First missing: {missing[:5]}"
            )
        X = X[feature_names]

    return X


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 200,
    log_period: int = 100,
) -> lgb.Booster:
    """Train a LightGBM binary classifier with early stopping on the valid set."""
    params = {**DEFAULT_LGBM_PARAMS, **(params or {})}

    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    valid_set = lgb.Dataset(
        X_valid, label=y_valid, reference=train_set, free_raw_data=False
    )

    callbacks = [
        lgb.early_stopping(early_stopping_rounds),
        lgb.log_evaluation(log_period),
    ]

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )
    return booster


def evaluate_pd_model(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
) -> dict[str, float]:
    """Compute AUC, Gini, KS, log-loss and Brier score for a binary classifier."""
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    auc = roc_auc_score(y_true, y_pred_proba)
    gini = 2.0 * auc - 1.0

    ks = ks_2samp(
        y_pred_proba[y_true == 1],
        y_pred_proba[y_true == 0],
    ).statistic

    return {
        "auc": float(auc),
        "gini": float(gini),
        "ks": float(ks),
        "log_loss": float(log_loss(y_true, y_pred_proba)),
        "brier": float(brier_score_loss(y_true, y_pred_proba)),
    }


def plot_roc(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    label: str | None = None,
) -> plt.Axes:
    """Plot ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ax.plot(fpr, tpr, color="#3a86ff", lw=2, label=f"{label or 'model'} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#aaa", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve", fontsize=12, weight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    return ax


def plot_calibration(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    *,
    n_bins: int = 10,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot reliability (calibration) curve in n equal-width bins."""
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins, strategy="quantile")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ax.plot(prob_pred, prob_true, marker="o", color="#ef476f", lw=2, label="model")
    ax.plot([0, 1], [0, 1], ls="--", color="#aaa", lw=1, label="perfect")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical default rate")
    ax.set_title(f"Calibration curve (quantile-binned, n={n_bins})", fontsize=12, weight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    return ax


def get_feature_importance(
    booster: lgb.Booster,
    *,
    importance_type: str = "gain",
) -> pd.DataFrame:
    """Return feature importance as a sorted DataFrame."""
    return (
        pd.DataFrame(
            {
                "feature": booster.feature_name(),
                "importance": booster.feature_importance(importance_type=importance_type),
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def plot_feature_importance(
    booster: lgb.Booster,
    *,
    top: int = 30,
    importance_type: str = "gain",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Horizontal bar plot of the top-`top` features by gain importance."""
    imp = get_feature_importance(booster, importance_type=importance_type).head(top)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(imp))))

    sns.barplot(x="importance", y="feature", data=imp, color="#3a86ff", ax=ax)
    ax.set_title(
        f"LightGBM feature importance ({importance_type}, top {top})",
        fontsize=12, weight="bold",
    )
    ax.set_xlabel(f"importance ({importance_type})")
    ax.set_ylabel("")
    return ax


def select_features_by_importance(
    importance_df: pd.DataFrame,
    *,
    top_n: int | None = None,
    min_importance: float = 0.0,
    cumulative_share: float | None = None,
) -> list[str]:
    """Pick a feature subset from a sorted importance DataFrame.

    Selection modes (apply in this order, all optional):
        - ``min_importance``: drop features whose gain is at or below this value.
        - ``top_n``: keep only the top-N most important features.
        - ``cumulative_share``: keep the smallest prefix that reaches this
          fraction of total gain (e.g. ``0.95`` keeps features covering 95%).

    The input is expected to be the output of :func:`get_feature_importance`
    (a frame with ``feature`` and ``importance`` columns sorted descending).
    """
    if not {"feature", "importance"}.issubset(importance_df.columns):
        raise ValueError("importance_df must contain 'feature' and 'importance' columns")

    df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    df = df[df["importance"] > min_importance]

    if cumulative_share is not None:
        if not 0.0 < cumulative_share <= 1.0:
            raise ValueError("cumulative_share must be in (0, 1]")
        total = df["importance"].sum()
        if total > 0:
            cum = df["importance"].cumsum() / total
            cutoff = (cum >= cumulative_share).idxmax() + 1
            df = df.iloc[:cutoff]

    if top_n is not None:
        df = df.head(top_n)

    return df["feature"].tolist()


def save_booster(booster: lgb.Booster, path: Path | str) -> None:
    """Persist a LightGBM booster to its native text format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))
