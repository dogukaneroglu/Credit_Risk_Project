"""Feature engineering primitives for the Credit Risk Scoring project.

Bu modül ilişkisel tabloların (bureau, bureau_balance, previous_application
vb.) `SK_ID_CURR` bazında bellek dostu agregasyonlarını içerir. Tüm fonksiyonlar
saf — yani aynı girdiyle aynı çıktıyı üretir, side-effect yoktur.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .utils import reduce_mem_usage


def _one_hot_encode(
    df: pd.DataFrame,
    *,
    columns: Iterable[str] | None = None,
    nan_as_category: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode object/category columns and return (df, new_columns)."""
    original = list(df.columns)
    cat_cols = (
        list(columns)
        if columns is not None
        else [
            c
            for c in df.columns
            if isinstance(df[c].dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(df[c])
            or pd.api.types.is_string_dtype(df[c])
        ]
    )
    encoded = pd.get_dummies(df, columns=cat_cols, dummy_na=nan_as_category)
    new_cols = [c for c in encoded.columns if c not in original]
    for c in new_cols:
        encoded[c] = encoded[c].astype(np.int8)
    return encoded, new_cols


def _flatten_agg_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Convert MultiIndex columns produced by `groupby.agg` into flat names."""
    df = df.copy()
    df.columns = pd.Index(
        [f"{prefix}_{col[0]}_{col[1].upper()}" for col in df.columns]
    )
    return df


def aggregate_bureau_balance(bureau_balance: pd.DataFrame) -> pd.DataFrame:
    """Roll up `bureau_balance` to one row per `SK_ID_BUREAU`.

    Çıktı sütunları:
    - `BB_MONTHS_BALANCE_MIN/MAX/SIZE`
    - `BB_STATUS_<X>_MEAN` (one-hot ortalamaları, yani ay-payı)
    """
    bb = bureau_balance.copy()
    bb, status_cols = _one_hot_encode(bb, columns=["STATUS"])

    agg_dict: dict[str, list[str]] = {"MONTHS_BALANCE": ["min", "max", "size"]}
    for c in status_cols:
        agg_dict[c] = ["mean"]

    agg = bb.groupby("SK_ID_BUREAU").agg(agg_dict)
    agg = _flatten_agg_columns(agg, prefix="BB")
    return reduce_mem_usage(agg.reset_index(), verbose=False)


def aggregate_bureau(
    bureau: pd.DataFrame,
    *,
    bureau_balance_agg: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Roll up `bureau` (+ optional `bureau_balance` agg) to one row per `SK_ID_CURR`.

    Çıktı `SK_ID_CURR` indeksli geniş bir DataFrame'dir; sütunlar `BURO_` öneki
    (ve kategorik özet sütunları için `BURO_ACTIVE_` / `BURO_CLOSED_` öneki)
    taşır.
    """
    b = bureau.copy()

    if bureau_balance_agg is not None:
        b = b.merge(bureau_balance_agg, how="left", on="SK_ID_BUREAU")

    b, cat_cols = _one_hot_encode(
        b,
        columns=[
            c
            for c in b.columns
            if isinstance(b[c].dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(b[c])
            or pd.api.types.is_string_dtype(b[c])
        ],
    )

    num_aggregations: dict[str, list[str]] = {
        "DAYS_CREDIT": ["min", "max", "mean", "var"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "DAYS_CREDIT_UPDATE": ["mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["mean"],
        "AMT_CREDIT_SUM": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean"],
        "AMT_CREDIT_SUM_LIMIT": ["mean", "sum"],
        "AMT_ANNUITY": ["max", "mean"],
        "CNT_CREDIT_PROLONG": ["sum"],
        "DAYS_ENDDATE_FACT": ["min", "max", "mean"],
    }
    if bureau_balance_agg is not None:
        for c in bureau_balance_agg.columns:
            if c == "SK_ID_BUREAU":
                continue
            num_aggregations[c] = ["min", "max", "mean"]

    cat_aggregations = {c: ["mean"] for c in cat_cols}

    available_num = {k: v for k, v in num_aggregations.items() if k in b.columns}
    agg = b.groupby("SK_ID_CURR").agg({**available_num, **cat_aggregations})
    agg = _flatten_agg_columns(agg, prefix="BURO")

    active_mask_col = "CREDIT_ACTIVE_Active"
    closed_mask_col = "CREDIT_ACTIVE_Closed"
    if active_mask_col in b.columns:
        active = (
            b[b[active_mask_col] == 1]
            .groupby("SK_ID_CURR")
            .agg(available_num)
        )
        active = _flatten_agg_columns(active, prefix="BURO_ACTIVE")
        agg = agg.join(active, how="left")
    if closed_mask_col in b.columns:
        closed = (
            b[b[closed_mask_col] == 1]
            .groupby("SK_ID_CURR")
            .agg(available_num)
        )
        closed = _flatten_agg_columns(closed, prefix="BURO_CLOSED")
        agg = agg.join(closed, how="left")

    counts = b.groupby("SK_ID_CURR").size().rename("BURO_COUNT")
    agg = agg.join(counts, how="left")

    return reduce_mem_usage(agg.reset_index(), verbose=False)


def merge_features(
    base: pd.DataFrame,
    *features: pd.DataFrame,
    on: str = "SK_ID_CURR",
) -> pd.DataFrame:
    """LEFT JOIN multiple feature tables onto a base frame on `on`."""
    out = base
    for feat in features:
        out = out.merge(feat, how="left", on=on)
    return out
