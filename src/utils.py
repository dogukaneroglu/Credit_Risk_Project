"""Shared utilities for the Credit Risk Scoring project.

Tüm notebook ve modüller bu yardımcıları kullanır. Kaizen prensibi gereği
fonksiyonlar küçük, tek sorumluluklu ve test edilebilir tutulmuştur.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def reduce_mem_usage(
    df: pd.DataFrame,
    *,
    category_threshold: float = 0.5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Downcast numeric dtypes and convert low-cardinality strings to category.

    Float sütunlar `float32`'ye, integer sütunlar mümkün olan en küçük signed
    integer tipine indirilir. NaN içeren integer sütunlar float olarak korunur
    çünkü pandas'ta klasik `int` tipleri NaN'i temsil edemez.

    Parameters
    ----------
    df : pd.DataFrame
        Bellek kullanımı küçültülecek DataFrame.
    category_threshold : float, default 0.5
        `unique / len` oranı bu eşiğin altında kalan object sütunları
        `category` tipine dönüştürülür.
    verbose : bool, default True
        Önce/sonra bellek kullanımını yazdırır.

    Returns
    -------
    pd.DataFrame
        Bellek kullanımı düşürülmüş yeni DataFrame.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    out = df.copy()

    for col in out.columns:
        col_dtype = out[col].dtype

        if pd.api.types.is_integer_dtype(col_dtype):
            col_min, col_max = out[col].min(), out[col].max()
            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                out[col] = out[col].astype(np.int8)
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                out[col] = out[col].astype(np.int16)
            elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                out[col] = out[col].astype(np.int32)
            else:
                out[col] = out[col].astype(np.int64)

        elif pd.api.types.is_float_dtype(col_dtype):
            out[col] = out[col].astype(np.float32)

        elif (
            pd.api.types.is_object_dtype(col_dtype)
            or pd.api.types.is_string_dtype(col_dtype)
        ):
            n_unique = out[col].nunique(dropna=False)
            if len(out) > 0 and n_unique / len(out) < category_threshold:
                out[col] = out[col].astype("category")

    end_mem = out.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem if start_mem else 0.0
        print(
            f"Memory usage: {start_mem:.2f} MB -> {end_mem:.2f} MB "
            f"({reduction:.1f}% reduction)"
        )
    return out


def load_csv(
    filename: str,
    *,
    raw_dir: Path | str = RAW_DIR,
    optimize_memory: bool = True,
    usecols: Iterable[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load a CSV from `data/raw/` and optionally downcast its dtypes."""
    path = Path(raw_dir) / filename
    if verbose:
        print(f"Loading: {path}")
    df = pd.read_csv(path, usecols=list(usecols) if usecols else None)
    if verbose:
        print(f"Shape: {df.shape}")
    if optimize_memory:
        df = reduce_mem_usage(df, verbose=verbose)
    return df


def missing_value_report(df: pd.DataFrame, top: int | None = 20) -> pd.DataFrame:
    """Return a per-column missing value summary sorted by ratio."""
    n = len(df)
    miss = df.isna().sum()
    report = (
        pd.DataFrame(
            {
                "missing": miss,
                "missing_ratio": (miss / n) if n else 0.0,
                "dtype": df.dtypes.astype(str),
            }
        )
        .sort_values("missing_ratio", ascending=False)
    )
    return report.head(top) if top else report


def split_feature_types(
    df: pd.DataFrame,
    *,
    exclude: Sequence[str] | None = None,
) -> dict[str, list[str]]:
    """Split columns into numeric / categorical / high-cardinality buckets.

    `category` ve `object` tipli sütunlar `categorical` olarak işaretlenir.
    Çok yüksek kardinaliteli (>50 unique) kategorikler ayrı bir kovaya alınır
    çünkü bunları doğrudan görselleştirmek anlamlı olmaz.
    """
    excluded = set(exclude or [])
    numeric: list[str] = []
    categorical: list[str] = []
    high_card: list[str] = []

    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        elif (
            isinstance(df[col].dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
        ):
            n_unique = df[col].nunique(dropna=False)
            if n_unique > 50:
                high_card.append(col)
            else:
                categorical.append(col)
        else:
            high_card.append(col)

    return {"numeric": numeric, "categorical": categorical, "high_cardinality": high_card}


def plot_target_distribution(
    df: pd.DataFrame,
    target: str = "TARGET",
    *,
    ax: plt.Axes | None = None,
    palette: Sequence[str] = ("#3a86ff", "#ef476f"),
) -> plt.Axes:
    """Bar plot of class counts with ratio annotations."""
    counts = df[target].value_counts().sort_index()
    ratios = df[target].value_counts(normalize=True).sort_index()

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    sns.barplot(x=counts.index.astype(str), y=counts.values, palette=list(palette), ax=ax)
    ax.set_title(f"{target} class distribution", fontsize=12, weight="bold")
    ax.set_xlabel(target)
    ax.set_ylabel("count")

    for i, (cnt, ratio) in enumerate(zip(counts.values, ratios.values)):
        ax.text(
            i,
            cnt,
            f"{cnt:,}\n({ratio:.1%})",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.margins(y=0.18)
    return ax


def plot_categorical_default_rate(
    df: pd.DataFrame,
    col: str,
    *,
    target: str = "TARGET",
    overall_rate: float | None = None,
    ax: plt.Axes | None = None,
    max_categories: int = 12,
) -> plt.Axes:
    """Plot per-category default rate (mean of TARGET) with sample count overlay."""
    work = df[[col, target]].copy()
    work[col] = work[col].astype("object").fillna("__missing__")

    counts = work.groupby(col, observed=True).size()
    rates = work.groupby(col, observed=True)[target].mean()

    summary = pd.DataFrame({"count": counts, "rate": rates})
    summary = summary.sort_values("rate", ascending=False).head(max_categories)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    sns.barplot(
        x=summary.index.astype(str),
        y=summary["rate"].values,
        color="#ef476f",
        ax=ax,
    )
    if overall_rate is not None:
        ax.axhline(overall_rate, ls="--", color="#222", alpha=0.6, label=f"overall = {overall_rate:.2%}")
        ax.legend(loc="upper right", fontsize=9)

    ax.set_title(f"Default rate by {col}", fontsize=11, weight="bold")
    ax.set_xlabel(col)
    ax.set_ylabel("P(TARGET=1)")
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    for i, (rate, n) in enumerate(zip(summary["rate"].values, summary["count"].values)):
        ax.text(i, rate, f"n={n:,}", ha="center", va="bottom", fontsize=8, color="#333")
    ax.margins(y=0.18)
    return ax


def plot_numeric_by_target(
    df: pd.DataFrame,
    col: str,
    *,
    target: str = "TARGET",
    ax: plt.Axes | None = None,
    clip_quantiles: tuple[float, float] = (0.01, 0.99),
) -> plt.Axes:
    """KDE comparison of a numeric feature between TARGET classes.

    Uç değerler grafiği bozmasın diye varsayılan olarak %1-%99 aralığına
    kırpılır.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    series = df[col].dropna()
    if clip_quantiles is not None and not series.empty:
        lo, hi = series.quantile(list(clip_quantiles))
        mask = (df[col] >= lo) & (df[col] <= hi)
    else:
        mask = df[col].notna()

    sns.kdeplot(
        data=df.loc[mask],
        x=col,
        hue=target,
        common_norm=False,
        fill=True,
        alpha=0.35,
        palette={0: "#3a86ff", 1: "#ef476f"},
        ax=ax,
    )
    ax.set_title(f"{col} | distribution by {target}", fontsize=11, weight="bold")
    ax.set_xlabel(col)
    ax.set_ylabel("density")
    return ax
