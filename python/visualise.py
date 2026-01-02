"""
visualise.py (replacement, hardcoded diagnostics)

generates
- Predicted vs Actual (within-algo and cross-algo)
- Residuals vs Runtime

Hardcoded scenarios (edit below if desired):
- Within-algo: insertion (train=test)
- Cross-algo: quick -> merge

Model:
- Ridge regression (alpha=1.0)
- StandardScaler fitted on training data only (Pipeline)


Optional:
  python3 python/visualise.py --csv data/raw/perf_runs.csv --outdir figures --min-time-s 0.5 --max-time-s 5
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


# Hardcoded scenarios
WITHIN_ALGO = "insertion"
CROSS_TRAIN_ALGO = "quick"
CROSS_TEST_ALGO = "merge"

# Feature sets
FEATURES_TIME = ["time_ns"]
FEATURES_ALL = ["time_ns", "instructions", "cycles", "branches", "branch_misses", "cache_misses"]

NS_PER_S = 1_000_000_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a small set of diagnostic figures for my dissertation (hardcoded scenarios)."
    )
    p.add_argument("--csv", type=Path, required=True, help="Path to perf_runs.csv")
    p.add_argument("--outdir", type=Path, default=Path("figures"), help="Directory to write PNG outputs")
    p.add_argument("--min-time-s", type=float, default=0.5, help="Min runtime (seconds) for filtering (default 0.5)")
    p.add_argument("--max-time-s", type=float, default=5.0, help="Max runtime (seconds) for filtering (default 5.0)")
    return p.parse_args()


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Basic required columns
    required = {"algo", "time_ns", "energy_j"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    # Convert numeric columns where present
    numeric_cols = ["n", "time_ns", "energy_j", "instructions", "cycles", "branches", "branch_misses", "cache_misses"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["algo", "time_ns", "energy_j"]).copy()

    # Filter out censored/floor values
    before = len(df)
    df = df[df["energy_j"] > 0.0].copy()
    removed = before - len(df)
    if before:
        print(f"[INFO] energy_j>0 filter: removed {removed} rows ({removed/before:.2%})")

    return df


def filter_time_band(df: pd.DataFrame, min_s: float, max_s: float, label: str) -> pd.DataFrame:
    before = len(df)
    lo = int(min_s * NS_PER_S)
    hi = int(max_s * NS_PER_S)
    df2 = df[(df["time_ns"] >= lo) & (df["time_ns"] <= hi)].copy()
    print(f"[INFO] Time-band {label}: kept {len(df2)}/{before} (removed {before-len(df2)})")
    return df2


def make_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])


def fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]):
    # Validate features exist
    for c in feature_cols:
        if c not in train_df.columns or c not in test_df.columns:
            raise ValueError(f"Missing feature column '{c}' in train/test data")

    X_train = train_df[feature_cols]
    y_train = train_df["energy_j"].astype(float)

    X_test = test_df[feature_cols]
    y_test = test_df["energy_j"].astype(float)

    model = make_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "r2": r2_score(y_test, y_pred),
        "rmse": root_mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
    }
    return y_test.values, y_pred, test_df["time_ns"].values, metrics


def save_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, title: str, outpath: Path):
    plt.figure()
    plt.scatter(y_true, y_pred, s=12)

    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi], linewidth=1)  # identity line

    plt.xlabel("Actual energy (J)")
    plt.ylabel("Predicted energy (J)")
    plt.title(title)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def save_residuals_vs_runtime(time_ns: np.ndarray, residuals: np.ndarray, title: str, outpath: Path):
    plt.figure()
    plt.scatter(time_ns / 1e9, residuals, s=12)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Runtime (s)")
    plt.ylabel("Residual (J)")
    plt.title(title)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def run_scenario(df: pd.DataFrame, outdir: Path, name: str, train_algo: str, test_algo: str | None,
                 feature_cols: list[str], min_time_s: float, max_time_s: float):
    # Select data
    train_df = df[df["algo"] == train_algo].copy()
    if train_df.empty:
        print(f"[WARN] Scenario '{name}': no rows for train_algo='{train_algo}'")
        return

    train_df = filter_time_band(train_df, min_time_s, max_time_s, label=f"({name} train={train_algo})")
    if train_df.empty:
        print(f"[WARN] Scenario '{name}': no rows left for train after time band")
        return

    if test_algo is None:
        # within-algo: test on held-out within same algo? (here: test on same set for diagnostics)
        test_df = train_df.copy()
        test_label = train_algo
    else:
        test_df = df[df["algo"] == test_algo].copy()
        if test_df.empty:
            print(f"[WARN] Scenario '{name}': no rows for test_algo='{test_algo}'")
            return
        test_df = filter_time_band(test_df, min_time_s, max_time_s, label=f"({name} test={test_algo})")
        if test_df.empty:
            print(f"[WARN] Scenario '{name}': no rows left for test after time band")
            return
        test_label = test_algo

    # Fit + predict
    y_true, y_pred, time_ns, m = fit_predict(train_df, test_df, feature_cols)

    feat_tag = "time" if feature_cols == FEATURES_TIME else "all"
    tag = f"{name}_{train_algo}_to_{test_label}_{feat_tag}"

    print(f"\n=== {tag} ===")
    print(f"[INFO] n_train={m['n_train']}, n_test={m['n_test']}")
    print(f"[INFO] R^2={m['r2']:.4f}, RMSE={m['rmse']:.6f} J, MAE={m['mae']:.6f} J")

    # Save plots
    save_pred_vs_actual(
        y_true, y_pred,
        title=f"Predicted vs Actual ({tag})",
        outpath=outdir / f"pred_actual_{tag}.png",
    )
    residuals = y_true - y_pred
    save_residuals_vs_runtime(
        time_ns, residuals,
        title=f"Residuals vs Runtime ({tag})",
        outpath=outdir / f"residuals_{tag}.png",
    )


def main() -> None:
    args = parse_args()
    df = load_and_clean(args.csv)

    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Rows loaded after cleaning: {len(df)}")
    print("[INFO] Rows by algorithm (post-clean):")
    print(df["algo"].value_counts().to_string())

    # Within-algo diagnostics (insertion)
    run_scenario(
        df=df, outdir=args.outdir, name="within",
        train_algo=WITHIN_ALGO, test_algo=None,
        feature_cols=FEATURES_TIME,
        min_time_s=args.min_time_s, max_time_s=args.max_time_s
    )
    run_scenario(
        df=df, outdir=args.outdir, name="within",
        train_algo=WITHIN_ALGO, test_algo=None,
        feature_cols=FEATURES_ALL,
        min_time_s=args.min_time_s, max_time_s=args.max_time_s
    )

    # Cross-algo diagnostics (quick -> merge)
    run_scenario(
        df=df, outdir=args.outdir, name="cross",
        train_algo=CROSS_TRAIN_ALGO, test_algo=CROSS_TEST_ALGO,
        feature_cols=FEATURES_TIME,
        min_time_s=args.min_time_s, max_time_s=args.max_time_s
    )
    run_scenario(
        df=df, outdir=args.outdir, name="cross",
        train_algo=CROSS_TRAIN_ALGO, test_algo=CROSS_TEST_ALGO,
        feature_cols=FEATURES_ALL,
        min_time_s=args.min_time_s, max_time_s=args.max_time_s
    )

    print(f"\n[INFO] Wrote figures to: {args.outdir.resolve()}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
