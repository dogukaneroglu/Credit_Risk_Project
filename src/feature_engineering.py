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


def _generic_aggregate(
    df: pd.DataFrame,
    *,
    group_col: str,
    num_aggregations: dict[str, list[str]],
    prefix: str,
    encode_cats: bool = True,
    drop_cols: Iterable[str] | None = None,
    count_name: str | None = None,
) -> pd.DataFrame:
    """Generic groupby-agg over numeric (+ optionally one-hot) columns.

    `bureau` dışındaki tablolar için tekrar eden boilerplate'i tek noktada
    toplar: kategorik kolonlar one-hot edilir, numerik agregasyonlar uygulanır,
    sütun isimleri düzleştirilir, bellek küçültülür.
    """
    work = df.copy()
    if drop_cols:
        work = work.drop(columns=[c for c in drop_cols if c in work.columns])

    cat_cols: list[str] = []
    if encode_cats:
        work, cat_cols = _one_hot_encode(work)

    available_num = {k: v for k, v in num_aggregations.items() if k in work.columns}
    cat_aggs = {c: ["mean"] for c in cat_cols}

    agg = work.groupby(group_col).agg({**available_num, **cat_aggs})
    agg = _flatten_agg_columns(agg, prefix=prefix)

    if count_name:
        agg[count_name] = work.groupby(group_col).size()

    return reduce_mem_usage(agg.reset_index(), verbose=False)


def aggregate_previous_application(previous_application: pd.DataFrame) -> pd.DataFrame:
    """Roll up `previous_application` to one row per `SK_ID_CURR`.

    Eklenen türetilmiş feature: `APP_CREDIT_PERC = AMT_APPLICATION / AMT_CREDIT`.
    `NAME_CONTRACT_STATUS = Approved / Refused` filtreleriyle ayrı özetler de
    üretilir, çünkü onaylanan vs. reddedilen başvurular ileride ayrı sinyaller
    taşır.
    """
    prev = previous_application.copy()
    if "AMT_APPLICATION" in prev.columns and "AMT_CREDIT" in prev.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = prev["AMT_APPLICATION"] / prev["AMT_CREDIT"]
        prev["APP_CREDIT_PERC"] = ratio.replace([np.inf, -np.inf], np.nan)

    num_aggregations: dict[str, list[str]] = {
        "AMT_ANNUITY": ["min", "max", "mean"],
        "AMT_APPLICATION": ["min", "max", "mean"],
        "AMT_CREDIT": ["min", "max", "mean"],
        "APP_CREDIT_PERC": ["min", "max", "mean", "var"],
        "AMT_DOWN_PAYMENT": ["min", "max", "mean"],
        "AMT_GOODS_PRICE": ["min", "max", "mean"],
        "HOUR_APPR_PROCESS_START": ["min", "max", "mean"],
        "RATE_DOWN_PAYMENT": ["min", "max", "mean"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "CNT_PAYMENT": ["mean", "sum"],
    }

    agg = _generic_aggregate(
        prev,
        group_col="SK_ID_CURR",
        num_aggregations=num_aggregations,
        prefix="PREV",
        drop_cols=["SK_ID_PREV"],
        count_name="PREV_COUNT",
    )

    for status in ("Approved", "Refused"):
        col = f"NAME_CONTRACT_STATUS_{status}"
        prev_with_flag = prev.copy()
        if col not in pd.get_dummies(prev_with_flag, columns=["NAME_CONTRACT_STATUS"]).columns:
            continue
        subset = prev_with_flag[prev_with_flag["NAME_CONTRACT_STATUS"] == status]
        if subset.empty:
            continue
        sub_agg = _generic_aggregate(
            subset,
            group_col="SK_ID_CURR",
            num_aggregations=num_aggregations,
            prefix=f"PREV_{status.upper()}",
            drop_cols=["SK_ID_PREV"],
            encode_cats=False,
        )
        agg = agg.merge(sub_agg, how="left", on="SK_ID_CURR")

    return reduce_mem_usage(agg, verbose=False)


def aggregate_pos_cash(pos_cash: pd.DataFrame) -> pd.DataFrame:
    """Roll up `POS_CASH_balance` to one row per `SK_ID_CURR`."""
    num_aggregations: dict[str, list[str]] = {
        "MONTHS_BALANCE": ["max", "mean", "size"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
    }
    return _generic_aggregate(
        pos_cash,
        group_col="SK_ID_CURR",
        num_aggregations=num_aggregations,
        prefix="POS",
        drop_cols=["SK_ID_PREV"],
        count_name="POS_COUNT",
    )


def aggregate_installments_payments(installments_payments: pd.DataFrame) -> pd.DataFrame:
    """Roll up `installments_payments` to one row per `SK_ID_CURR`.

    Türetilmiş ödeme disiplin metrikleri:
    - `PAYMENT_PERC` : ödenen / planlanan
    - `PAYMENT_DIFF` : planlanan − ödenen (eksik tutar)
    - `DPD`          : geç ödeme gün sayısı (≥0)
    - `DBD`          : erken ödeme gün sayısı (≥0)
    """
    ins = installments_payments.copy()

    if "AMT_PAYMENT" in ins.columns and "AMT_INSTALMENT" in ins.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            ins["PAYMENT_PERC"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"]
        ins["PAYMENT_PERC"] = ins["PAYMENT_PERC"].replace([np.inf, -np.inf], np.nan)
        ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]

    if "DAYS_ENTRY_PAYMENT" in ins.columns and "DAYS_INSTALMENT" in ins.columns:
        diff = ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]
        ins["DPD"] = diff.clip(lower=0)
        ins["DBD"] = (-diff).clip(lower=0)

    num_aggregations: dict[str, list[str]] = {
        "NUM_INSTALMENT_VERSION": ["nunique"],
        "DPD": ["max", "mean", "sum"],
        "DBD": ["max", "mean", "sum"],
        "PAYMENT_PERC": ["max", "mean", "var"],
        "PAYMENT_DIFF": ["max", "mean", "var"],
        "AMT_INSTALMENT": ["max", "mean", "sum"],
        "AMT_PAYMENT": ["max", "mean", "sum"],
        "DAYS_ENTRY_PAYMENT": ["max", "mean", "sum"],
    }

    return _generic_aggregate(
        ins,
        group_col="SK_ID_CURR",
        num_aggregations=num_aggregations,
        prefix="INSTAL",
        drop_cols=["SK_ID_PREV"],
        encode_cats=False,
        count_name="INSTAL_COUNT",
    )


def add_application_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add classic domain-driven ratio features on the application table.

    Bu oranlar Home Credit/Kaggle community'sinin yıllar içinde valide ettiği
    güçlü manüel feature'lardır. `DAYS_EMPLOYED = 365243` (≈1000 yıl) bilinen
    anomalidir — burada `NaN`'a çevrilir.
    """
    out = df.copy()

    if "DAYS_EMPLOYED" in out.columns:
        out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].replace(365243, np.nan)

    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(out.columns):
        out["CREDIT_INCOME_PERCENT"] = out["AMT_CREDIT"] / out["AMT_INCOME_TOTAL"]
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(out.columns):
        out["ANNUITY_INCOME_PERCENT"] = out["AMT_ANNUITY"] / out["AMT_INCOME_TOTAL"]
    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(out.columns):
        out["CREDIT_TERM"] = out["AMT_ANNUITY"] / out["AMT_CREDIT"]
    if {"DAYS_EMPLOYED", "DAYS_BIRTH"}.issubset(out.columns):
        out["DAYS_EMPLOYED_PERCENT"] = out["DAYS_EMPLOYED"] / out["DAYS_BIRTH"]
    if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(out.columns):
        out["INCOME_PER_PERSON"] = out["AMT_INCOME_TOTAL"] / out["CNT_FAM_MEMBERS"]

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def aggregate_credit_card_balance(credit_card_balance: pd.DataFrame) -> pd.DataFrame:
    """Roll up `credit_card_balance` to one row per `SK_ID_CURR`.

    Tüm numerik sütunlar üzerinde `min/max/mean/sum/var` agregasyonu uygulanır;
    kategorik sütunlar (NAME_CONTRACT_STATUS) one-hot edilip ortalama alınır.
    """
    cc = credit_card_balance.copy()
    drop_cols = ["SK_ID_PREV"]
    cc = cc.drop(columns=[c for c in drop_cols if c in cc.columns])

    numeric_cols = [
        c for c in cc.columns
        if c != "SK_ID_CURR" and pd.api.types.is_numeric_dtype(cc[c])
    ]
    num_aggregations = {c: ["min", "max", "mean", "sum", "var"] for c in numeric_cols}

    return _generic_aggregate(
        cc,
        group_col="SK_ID_CURR",
        num_aggregations=num_aggregations,
        prefix="CC",
        count_name="CC_COUNT",
    )
