"""Generate static images used by README.md.

Reads artifacts produced by `06_Scorecard.ipynb` and renders a single
summary figure (4-panel) plus an ROC + calibration figure from the
LightGBM PD model in `05_Modeling_LightGBM.ipynb`.

Usage:
    python scripts/generate_readme_assets.py

Outputs (assets/readme/):
    - scorecard_summary.png  — score distribution, decile bands, KS, lift
    - model_diagnostics.png  — ROC and calibration on the validation set
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import PROCESSED_DIR
from src.modeling import plot_roc, plot_calibration
from src.scorecard import (
    compute_band_analysis,
    compute_lift_table,
    plot_band_analysis,
    plot_ks_curve,
    plot_lift,
    plot_score_distribution,
)


def main() -> None:
    assets = PROJECT_ROOT / "assets" / "readme"
    assets.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    scores = pd.read_parquet(PROCESSED_DIR / "valid_scores.parquet")

    # 1) Scorecard summary (4 panels)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    plot_score_distribution(scores, ax=axes[0, 0])

    band_df = compute_band_analysis(scores, n_bands=10)
    plot_band_analysis(band_df, ax=axes[0, 1])

    plot_ks_curve(scores["TARGET"], scores["score"], ax=axes[1, 0])

    lift_df = compute_lift_table(scores, n_bands=10)
    plot_lift(lift_df, ax=axes[1, 1])

    fig.suptitle(
        "Scorecard summary on validation set (n = 61,503)",
        fontsize=14, weight="bold", y=1.00,
    )
    plt.tight_layout()
    out1 = assets / "scorecard_summary.png"
    fig.savefig(out1, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved : {out1}")

    # 2) Model diagnostics (ROC + calibration)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_roc(scores["TARGET"], scores["pd_proba"], ax=axes[0], label="LightGBM")
    plot_calibration(scores["TARGET"], scores["pd_proba"], n_bins=10, ax=axes[1])
    fig.suptitle("LightGBM PD model — validation diagnostics",
                 fontsize=14, weight="bold", y=1.02)
    plt.tight_layout()
    out2 = assets / "model_diagnostics.png"
    fig.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved : {out2}")


if __name__ == "__main__":
    main()
