"""Credit Risk Scoring — inference CLI.

Loads the trained LightGBM PD model and the PDO scorecard parameters produced
by ``05_Modeling_LightGBM.ipynb`` / ``06_Scorecard.ipynb``, then prints a
per-customer decision table::

    SK_ID_CURR | PD (%) | Score | Decision

Examples
--------
Demo (5 random customers from the validation set)::

    python predict.py

Score a custom file::

    python predict.py --input data/raw/new_customers.parquet --n-samples 10

Override the cutoff and persist the result::

    python predict.py --cutoff 600 --output decisions.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import PROCESSED_DIR
from src.modeling import prepare_features_for_inference
from src.scorecard import ScorecardParams, pd_to_score


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

MODEL_PATH = PROCESSED_DIR / "lgbm_pd_model.txt"
PARAMS_PATH = PROCESSED_DIR / "scorecard_params.json"
FEATURES_PATH = PROCESSED_DIR / "lgbm_selected_features.json"
DEMO_DATA_PATH = PROCESSED_DIR / "valid.parquet"

DEFAULT_N_SAMPLES = 5
DEFAULT_RANDOM_STATE = 42
FALLBACK_CUTOFF = 587


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


def load_artifacts() -> tuple[lgb.Booster, ScorecardParams, list[str], int]:
    """Load model, scorecard params, selected feature list, default cutoff."""
    for path, label in (
        (MODEL_PATH, "LightGBM model"),
        (PARAMS_PATH, "scorecard params"),
        (FEATURES_PATH, "feature list"),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found at {path}. "
                f"Run notebooks 05 and 06 first to generate the artifacts."
            )

    booster = lgb.Booster(model_file=str(MODEL_PATH))

    with open(PARAMS_PATH, encoding="utf-8") as f:
        params_dict = json.load(f)
    scorecard = ScorecardParams(
        pdo=float(params_dict["pdo"]),
        base_score=float(params_dict["base_score"]),
        base_pd=float(params_dict["base_pd"]),
    )
    cutoff = int(params_dict.get("chosen_cutoff", FALLBACK_CUTOFF))

    with open(FEATURES_PATH, encoding="utf-8") as f:
        feature_names = json.load(f)
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError(f"{FEATURES_PATH} did not contain a non-empty list of features.")

    return booster, scorecard, feature_names, cutoff


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(
        f"Unsupported input extension: {suffix or '<none>'}. "
        f"Expected .parquet or .csv."
    )


def load_input(
    input_path: Path | None,
    *,
    n_samples: int,
    random_state: int,
) -> tuple[pd.DataFrame, str]:
    """Return (sample_df, source_label) for either demo or user-supplied input."""
    if input_path is None:
        if not DEMO_DATA_PATH.exists():
            raise FileNotFoundError(
                f"Demo dataset not found at {DEMO_DATA_PATH}. "
                f"Run notebook 04 to produce the validation parquet."
            )
        df = pd.read_parquet(DEMO_DATA_PATH)
        if n_samples and n_samples < len(df):
            df = df.sample(n=n_samples, random_state=random_state)
        return df.reset_index(drop=True), f"demo: {DEMO_DATA_PATH.name}"

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    df = _read_table(input_path)
    if df.empty:
        raise ValueError(f"Input file {input_path} contained zero rows.")
    if n_samples and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=random_state)
    return df.reset_index(drop=True), str(input_path)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_customers(
    df: pd.DataFrame,
    *,
    booster: lgb.Booster,
    params: ScorecardParams,
    feature_names: list[str],
    cutoff: int,
) -> pd.DataFrame:
    """Run prediction + scoring + decision for every row of ``df``."""
    X = prepare_features_for_inference(df, feature_names=feature_names)

    pd_proba = booster.predict(X, num_iteration=booster.best_iteration)
    pd_proba = np.asarray(pd_proba, dtype=float)
    score = pd_to_score(pd_proba, params)
    decision = np.where(score >= cutoff, "APPROVE", "REJECT")

    if "SK_ID_CURR" in df.columns:
        ids = df["SK_ID_CURR"].astype(int).to_numpy()
    else:
        ids = np.arange(1, len(df) + 1, dtype=int)

    return pd.DataFrame(
        {
            "SK_ID_CURR": ids,
            "PD (%)": np.round(pd_proba * 100, 2),
            "Score": score.astype(int),
            "Decision": decision,
        }
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def render_table(df: pd.DataFrame) -> str:
    """Render an ASCII grid table with right-aligned numeric columns."""
    headers = list(df.columns)
    rows = [[_format_cell(v) for v in row] for row in df.itertuples(index=False, name=None)]

    widths = [
        max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
        for i, h in enumerate(headers)
    ]

    def line(left: str, mid: str, right: str, fill: str = "-") -> str:
        return left + mid.join(fill * (w + 2) for w in widths) + right

    def format_row(values: list[str], align_right: list[bool]) -> str:
        cells = [
            f" {v.rjust(w)} " if right else f" {v.ljust(w)} "
            for v, w, right in zip(values, widths, align_right)
        ]
        return "|" + "|".join(cells) + "|"

    is_numeric = [pd.api.types.is_numeric_dtype(df[c]) for c in headers]

    top = line("+", "+", "+")
    head = format_row(headers, [False] * len(headers))
    sep = line("+", "+", "+", fill="=")
    body = [format_row(row, is_numeric) for row in rows]
    bottom = line("+", "+", "+")

    return "\n".join([top, head, sep, *body, bottom])


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def save_output(result: pd.DataFrame, output_path: Path) -> None:
    suffix = output_path.suffix.lower()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".csv":
        result.to_csv(output_path, index=False)
    elif suffix == ".parquet":
        result.to_parquet(output_path, index=False)
    else:
        raise ValueError(
            f"Unsupported output extension: {suffix or '<none>'}. Use .csv or .parquet."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description="Score credit applicants with the trained LightGBM PD model + PDO scorecard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path, default=None,
        help="Path to a .parquet or .csv file with customer features. "
             "If omitted, the script samples from the validation set (demo mode).",
    )
    parser.add_argument(
        "--n-samples", "-n",
        type=int, default=DEFAULT_N_SAMPLES,
        help="Number of rows to score (sampled randomly if the input has more).",
    )
    parser.add_argument(
        "--cutoff", "-c",
        type=int, default=None,
        help="Score cutoff for APPROVE / REJECT. Defaults to the value persisted in scorecard_params.json.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path, default=None,
        help="Optional path to write the results (.csv or .parquet).",
    )
    parser.add_argument(
        "--seed",
        type=int, default=DEFAULT_RANDOM_STATE,
        help="Random seed used when sampling rows.",
    )
    return parser.parse_args(argv)


def _print_banner(source: str, cutoff: int, n_rows: int, scorecard: ScorecardParams) -> None:
    print("=" * 64)
    print("Credit Risk Scoring — Inference CLI")
    print("=" * 64)
    print(f"Source       : {source}")
    print(f"Rows scored  : {n_rows}")
    print(f"Cutoff       : {cutoff}  (>= cutoff -> APPROVE, < cutoff -> REJECT)")
    print(f"Scorecard    : PDO={int(scorecard.pdo)}  base_score={int(scorecard.base_score)}  "
          f"base_pd={scorecard.base_pd:.2%}")
    print("-" * 64)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        booster, scorecard, feature_names, default_cutoff = load_artifacts()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"[error] failed to load model artifacts: {exc}", file=sys.stderr)
        return 1

    cutoff = args.cutoff if args.cutoff is not None else default_cutoff

    try:
        df, source = load_input(args.input, n_samples=args.n_samples, random_state=args.seed)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] failed to load input: {exc}", file=sys.stderr)
        return 1

    _print_banner(source, cutoff, len(df), scorecard)

    try:
        result = score_customers(
            df,
            booster=booster,
            params=scorecard,
            feature_names=feature_names,
            cutoff=cutoff,
        )
    except (KeyError, ValueError) as exc:
        print(f"[error] scoring failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[error] unexpected scoring failure: {exc}", file=sys.stderr)
        return 1

    print(render_table(result))

    n_approve = int((result["Decision"] == "APPROVE").sum())
    n_reject = len(result) - n_approve
    print(f"\nSummary: {n_approve} APPROVE | {n_reject} REJECT  "
          f"(approval rate {n_approve / len(result):.1%})")

    if args.output is not None:
        try:
            save_output(result, args.output)
        except (OSError, ValueError) as exc:
            print(f"[warn] could not save output: {exc}", file=sys.stderr)
        else:
            print(f"Saved : {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
