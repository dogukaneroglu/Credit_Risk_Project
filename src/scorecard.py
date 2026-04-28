"""Scorecard utilities for the Credit Risk Scoring project.

PD olasılıklarını yorumlanabilir kredi skorlarına dönüştürmek, decile/band
analizi yapmak, KS bazlı kesim eşiği bulmak ve lift / kalibrasyon grafiklerini
çizmek için saf, yan-etkisiz yardımcı fonksiyonlar.

Mimari prensip: notebook'lar yalnızca orchestration; mantık burada test edilebilir
biçimde durur (mirror of `src/modeling.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


EPS = 1e-12


# ---------------------------------------------------------------------------
# Scorecard parametre matematigi
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScorecardParams:
    """PDO bazli scorecard parametreleri.

    Convention (higher score = lower risk):
        score = offset + factor * log((1 - pd) / pd)

    where:
        factor = pdo / log(2)
        offset = base_score - factor * log(base_odds)
        base_odds = (1 - base_pd) / base_pd
    """

    pdo: float
    base_score: float
    base_pd: float

    @property
    def base_odds(self) -> float:
        return (1.0 - self.base_pd) / self.base_pd

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2.0)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * np.log(self.base_odds)

    def as_dict(self) -> dict[str, float]:
        return {
            "pdo": self.pdo,
            "base_score": self.base_score,
            "base_pd": self.base_pd,
            "base_odds": self.base_odds,
            "factor": self.factor,
            "offset": self.offset,
        }


def pd_to_score(
    pd_proba: np.ndarray | pd.Series,
    params: ScorecardParams,
    *,
    score_min: int = 300,
    score_max: int = 850,
) -> np.ndarray:
    """Convert PD probabilities to integer scores in [score_min, score_max]."""
    p = np.clip(np.asarray(pd_proba, dtype=float), EPS, 1 - EPS)
    log_good_odds = np.log((1.0 - p) / p)
    raw = params.offset + params.factor * log_good_odds
    return np.clip(np.round(raw), score_min, score_max).astype(int)


# ---------------------------------------------------------------------------
# Score band / decile analizi
# ---------------------------------------------------------------------------


def compute_band_analysis(
    df: pd.DataFrame,
    *,
    score_col: str = "score",
    pd_col: str = "pd_proba",
    target_col: str = "TARGET",
    n_bands: int = 10,
) -> pd.DataFrame:
    """Score bazli quantile band tablosu (decile by default).

    Bantlar dusuk skorlu (yuksek riskli) musteri ile baslar; her bant icin
    actual default rate ve mean predicted PD birlikte raporlanir.
    """
    work = df[[score_col, pd_col, target_col]].copy()
    work["band"] = pd.qcut(
        work[score_col], q=n_bands, labels=False, duplicates="drop"
    ) + 1

    grouped = work.groupby("band", as_index=False).agg(
        count=(target_col, "size"),
        defaults=(target_col, "sum"),
        actual_pd=(target_col, "mean"),
        predicted_pd=(pd_col, "mean"),
        score_min=(score_col, "min"),
        score_max=(score_col, "max"),
    )
    grouped["score_range"] = (
        grouped["score_min"].astype(str) + "–" + grouped["score_max"].astype(str)
    )
    return grouped[
        ["band", "score_range", "count", "defaults", "actual_pd", "predicted_pd"]
    ]


# ---------------------------------------------------------------------------
# Cutoff secimi
# ---------------------------------------------------------------------------


def ks_curve(
    y_true: np.ndarray | pd.Series,
    score: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Per-threshold KS curve: TPR, FPR ve KS = TPR - FPR."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)

    order = np.argsort(-s)  # high score = low risk; iterate high->low
    s_sorted = s[order]
    y_sorted = y[order]

    cum_pos = np.cumsum(y_sorted)
    cum_neg = np.cumsum(1 - y_sorted)
    total_pos = max(int(y_sorted.sum()), 1)
    total_neg = max(int((1 - y_sorted).sum()), 1)

    return pd.DataFrame(
        {
            "threshold": s_sorted,
            "tpr": cum_pos / total_pos,
            "fpr": cum_neg / total_neg,
            "ks": cum_pos / total_pos - cum_neg / total_neg,
        }
    )


def find_optimal_cutoff(
    y_true: np.ndarray | pd.Series,
    score: np.ndarray | pd.Series,
) -> dict[str, float]:
    """KS-maksimize eden threshold + o noktadaki TPR/FPR/KS degerleri."""
    curve = ks_curve(y_true, score)
    # KS, "yuksek skor = iyi musteri" varsayiminda |TPR - FPR|; biz cutoff'un
    # ALTINI red etmek istedigimiz icin reddedilen kotuluk oranini maksimize eden
    # threshold = en yuksek (1 - tpr_low) - (1 - fpr_low) = -ks. Bu yuzden mutlak
    # degerin max'ini aliyoruz; dogal yorumlama icin abs.
    idx = curve["ks"].abs().idxmax()
    row = curve.loc[idx]
    return {
        "threshold": float(row["threshold"]),
        "ks": float(abs(row["ks"])),
        "tpr": float(row["tpr"]),
        "fpr": float(row["fpr"]),
    }


def cutoff_metrics_table(
    y_true: np.ndarray | pd.Series,
    score: np.ndarray | pd.Series,
    cutoffs: list[int] | np.ndarray,
) -> pd.DataFrame:
    """Cesitli score-cutoff'larda business metrics tablosu.

    Convention: cutoff'un *altini* reddet, ustunu kabul et.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    n = len(y)
    n_pos = max(int(y.sum()), 1)

    rows: list[dict[str, float]] = []
    for c in cutoffs:
        approve = s >= c
        n_approve = int(approve.sum())
        if n_approve == 0:
            continue

        approve_default_rate = float(y[approve].mean())
        reject = ~approve
        n_reject = int(reject.sum())
        catch_rate = float(y[reject].sum() / n_pos)

        rows.append(
            {
                "cutoff": int(c),
                "approval_rate": n_approve / n,
                "rejection_rate": n_reject / n,
                "approve_default_rate": approve_default_rate,
                "catch_rate": catch_rate,
            }
        )
    return pd.DataFrame(rows)


def classification_metrics_at_cutoff(
    y_true: np.ndarray | pd.Series,
    score: np.ndarray | pd.Series,
    cutoff: float,
) -> dict[str, Any]:
    """Confusion matrix + precision/recall/F1 'reject below cutoff' icin.

    Pozitif sınıf = default. Modelin "reject" kararı default tahmini
    olarak kabul edilir.
    """
    y = np.asarray(y_true).astype(int)
    pred_default = (np.asarray(score, dtype=float) < cutoff).astype(int)

    cm = confusion_matrix(y, pred_default, labels=[0, 1])
    return {
        "cutoff": float(cutoff),
        "confusion_matrix": cm,
        "precision": float(precision_score(y, pred_default, zero_division=0)),
        "recall": float(recall_score(y, pred_default, zero_division=0)),
        "f1": float(f1_score(y, pred_default, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Lift / kalibrasyon
# ---------------------------------------------------------------------------


def compute_lift_table(
    df: pd.DataFrame,
    *,
    score_col: str = "score",
    target_col: str = "TARGET",
    n_bands: int = 10,
) -> pd.DataFrame:
    """Lift = (band'daki default oran) / (genel default oran)."""
    base_rate = float(df[target_col].mean())
    work = df[[score_col, target_col]].copy()
    work["band"] = pd.qcut(work[score_col], q=n_bands, labels=False, duplicates="drop") + 1
    table = work.groupby("band", as_index=False).agg(
        count=(target_col, "size"),
        defaults=(target_col, "sum"),
        default_rate=(target_col, "mean"),
    )
    table["lift"] = table["default_rate"] / base_rate
    table["cum_defaults"] = table["defaults"].cumsum()
    table["cum_capture_rate"] = table["cum_defaults"] / table["defaults"].sum()
    return table


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_score_distribution(
    df: pd.DataFrame,
    *,
    score_col: str = "score",
    target_col: str = "TARGET",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Default vs non-default score dagilimi (overlapping KDE)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    palette = {0: "#3a86ff", 1: "#ef476f"}
    for cls, color in palette.items():
        sub = df.loc[df[target_col] == cls, score_col]
        sns.kdeplot(
            sub,
            ax=ax,
            color=color,
            fill=True,
            alpha=0.35,
            linewidth=2,
            label=f"{'default' if cls == 1 else 'non-default'} (n={len(sub):,})",
        )

    ax.set_xlabel("Credit score")
    ax.set_ylabel("Density")
    ax.set_title("Score distribution by outcome", fontsize=12, weight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    return ax


def plot_band_analysis(
    band_df: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Decile-bar: actual vs predicted default rate."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    width = 0.4
    x = band_df["band"].astype(int).to_numpy()
    ax.bar(x - width / 2, band_df["actual_pd"], width=width, label="actual", color="#ef476f")
    ax.bar(x + width / 2, band_df["predicted_pd"], width=width, label="predicted", color="#3a86ff")

    ax.set_xticks(x)
    ax.set_xlabel("Score band (1 = riskiest, N = safest)")
    ax.set_ylabel("Default rate")
    ax.set_title("Decile band: actual vs predicted PD", fontsize=12, weight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return ax


def plot_ks_curve(
    y_true: np.ndarray | pd.Series,
    score: np.ndarray | pd.Series,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """KS curve: TPR ve FPR'in score percentile'ina karsi grafigi."""
    curve = ks_curve(y_true, score)
    pct = np.linspace(0, 1, len(curve))

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    ax.plot(pct, curve["tpr"], label="TPR (defaulters)", color="#ef476f", lw=2)
    ax.plot(pct, curve["fpr"], label="FPR (non-defaulters)", color="#3a86ff", lw=2)

    idx = curve["ks"].abs().idxmax()
    ks_val = float(abs(curve.loc[idx, "ks"]))
    ax.vlines(pct[idx], curve.loc[idx, "fpr"], curve.loc[idx, "tpr"],
              colors="black", linestyles="--", lw=1.2,
              label=f"KS = {ks_val:.4f}")

    ax.set_xlabel("Population fraction (sorted high→low score)")
    ax.set_ylabel("Cumulative rate")
    ax.set_title("KS curve", fontsize=12, weight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return ax


def plot_lift(
    lift_df: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Bar chart of lift per decile."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    sns.barplot(x="band", y="lift", data=lift_df, color="#3a86ff", ax=ax)
    ax.axhline(1.0, ls="--", color="#666", lw=1, label="baseline (lift=1)")
    ax.set_xlabel("Score band (1 = riskiest)")
    ax.set_ylabel("Lift")
    ax.set_title("Lift chart by score band", fontsize=12, weight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return ax
