"""
This script parses CLI arguments, calls helper
functions in analysis.py, and fits a linear regression model and a ridge regression model.

This script has swolen to the full data prep + analysis script (oops)

Usage example python3 python/model.py --algo quick --model ridge
"""

import argparse
from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

from paths import RAW_PERF_CSV, MODELS_DIR, ensure_dirs
from analysis import load_dataset, build_features, split_train_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a regression model to predict per-run energy (J).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=RAW_PERF_CSV,
        help=f"Path to perf_runs.csv (default: {RAW_PERF_CSV})",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        help="If set, only use rows where algo == this value (example 'insertion').",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to reserve for the test set (default: 0.2).",
    )
   # parser.add_argument(
   #     "--no-time",
   #     action="store_true",
   #     help="Exclude time_ns from the feature set (use counters only).",
   # )
    parser.add_argument(
        "--mode",
        choices=["time", "counters", "all"],
        default="all",
        help=(
        "Choose features to select:\n"
        "time     -> time_ns only (baseline predictor)\n"
        "counters -> use only perf counters\n"
        "all      -> time_ns + counters (default)"
             )
    )
    parser.add_argument(
        "--model",
        choices=["linear", "ridge"],
        default="linear",
        help=(
        "Regression model to use:\n"
        "  linear -> ordinary least squares (default)\n"
        "  ridge  -> linear model with L2 regularisation"
        )
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        help="Standardise predictors (z-score) using training data only (via Pipeline).",
    )
    parser.add_argument(
        "--test-algo",
        type=str,
        default=None,
        help=(
            "Optional: if set, train on --algo and evaluate on this other algo.\n"
            "If not set, use usual random train/test split within --algo (or all algos)."
        )
    )
    parser.add_argument(
        "--min-time-s",
        type=float,
        default=None,
        help="Optional: keep only runs with runtime >= this many seconds.",
    )
    parser.add_argument(
        "--max-time-s",
        type=float,
        default=None,
        help="Optional: keep only runs with runtime <= this many seconds.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    ensure_dirs()

    # lil conversion for below function
    NS_PER_S = 1_000_000_000

    # helper function for limiting df by time bands
    def filter_time_band(df, min_s, max_s, label=""):
        before = len(df)
        if min_s is not None:
            df = df[df["time_ns"] >= int(min_s * NS_PER_S)]
        if max_s is not None:
            df = df[df["time_ns"] <= int(max_s * NS_PER_S)]
        removed = before - len(df)
        if min_s is not None or max_s is not None:
            print(f"[INFO] Time-band{(' ' + label) if label else ''}: "
                  f"kept {len(df)}/{before} (removed {removed})")
        return df

    #choose regression type based on CLI flag
    def make_model(kind: str, scale: bool):
        if kind == "linear":
            reg = LinearRegression()
        elif kind == "ridge":
            # Might tune this later
            reg = Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown model kind: {kind}")

        if scale:
            return Pipeline([
                ("scaler", StandardScaler()),
                ("reg", reg),
            ])
        return reg

    # Load full dataset (no filter yet)
    df = load_dataset(args.csv, algo_filter=None)
    if df.empty:
        print("[ERROR] No data found at CSV path; nothing to do.")
        return

    # Exclude runs where RAPL energy quantised to 0 (below measurement resolution)
    before = len(df)
    df = df[df["energy_j"] > 0.0]
    removed = before - len(df)
    if removed > 0:
        print(f"[INFO] Filtered energy_j>0: removed {removed} rows ({removed/before:.2%})\n")


    # Decide training/testing split mode
    cross_algo = args.test_algo is not None

    if cross_algo:
        # Cross-algorithm mode: train on args.algo, test on args.test_algo
        if args.algo is None:
            print("[ERROR] --algo must be set when using --test-algo")
            return

        train_algo = args.algo
        test_algo = args.test_algo

        train_df = df[df["algo"] == train_algo]
        test_df = df[df["algo"] == test_algo]

        if train_df.empty:
            print(f"[ERROR] No rows found for train algo='{train_algo}'")
            return
        if test_df.empty:
            print(f"[ERROR] No rows found for test algo='{test_algo}'")
            return

        train_df = filter_time_band(train_df, args.min_time_s, args.max_time_s, label=f"(train={train_algo})")
        test_df  = filter_time_band(test_df,  args.min_time_s, args.max_time_s, label=f"(test={test_algo})")

        if train_df.empty or test_df.empty:
            print("[ERROR] No data left after time-band filtering for train or test.")
            return

        print(f"Using data from: {args.csv}")
        print(f"Train algorithm: {train_algo}")
        print(f"Test  algorithm: {test_algo}")
        print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    else:
        # Normal mode: optional filter on a single algo, then random split
        if args.algo is not None:
            df = df[df["algo"] == args.algo]
        algo_label = args.algo or "all_algos"

        df = filter_time_band(df, args.min_time_s, args.max_time_s, label=f"(algo={algo_label})")
        if df.empty:
            print("[ERROR] No data left after time-band filtering.")
            return

        print(f"Using data from: {args.csv}")
        print(f"Algorithm filter (train+test): {algo_label}")
        print(f"Rows after filtering/cleaning: {len(df)}")

        if df.empty:
            print("[ERROR] No data after filtering; nothing to train on.")
            return

    #Choose feature columns based on --mode
    if args.mode == "time":
        print("[INFO] Using time_ns only (baseline model)")
        feature_cols = ["time_ns"]

    elif args.mode == "counters":
        print("[INFO] Using counters only (no time_ns)")
        feature_cols = [
            "instructions",
            "cycles",
            "branches",
            "branch_misses",
            "cache_misses",
        ]

    else:  # "all"
        print("[INFO] Using time_ns + counters")
        feature_cols = [
            "time_ns",
            "instructions",
            "cycles",
            "branches",
            "branch_misses",
            "cache_misses",
        ]

    # Build X_train, X_test, y_train, y_test
    if cross_algo:
        # Train on one algorithm, test on another
        X_train = train_df[feature_cols]
        y_train = train_df["energy_j"]
        X_test = test_df[feature_cols]
        y_test = test_df["energy_j"]
    else:
        # Single dataset, random train/test split
        X = df[feature_cols]
        y = df["energy_j"]

        X_train, X_test, y_train, y_test = split_train_test(
            X, y, test_size=args.test_size, random_state=42
        )

    print(f"Using features: {feature_cols}")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Diagnostics: target spread (R^2 can be misleading if y_test variance is tiny)
    y_test_mean = float(np.mean(y_test))
    y_test_std = float(np.std(y_test))
    y_train_mean = float(np.mean(y_train))
    print(f"[INFO] y_train mean={y_train_mean:.6f} J")
    print(f"[INFO] y_test  mean={y_test_mean:.6f} J, std={y_test_std:.6f} J")

    # Oracle baseline for context: predicting mean(y_test) gives R^2 = 0 by definition
    oracle_pred = np.full(len(y_test), y_test_mean)
    oracle_rmse = root_mean_squared_error(y_test, oracle_pred)
    oracle_mae = mean_absolute_error(y_test, oracle_pred)
    print(f"[INFO] Oracle baseline (mean of y_test): RMSE={oracle_rmse:.6f} J, MAE={oracle_mae:.6f} J")

    # Train model
    model = make_model(args.model, args.scale)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = root_mean_squared_error(y_test, y_pred_test)

    mae_test = mean_absolute_error(y_test, y_pred_test)

    # Always predict the mean training energy
    baseline_pred = np.full(shape=len(y_test), fill_value=float(np.mean(y_train)))
    baseline_rmse = root_mean_squared_error(y_test, baseline_pred)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    # Does this model beat the baseline? (Answeing yes or no to RQ stuff)
    rmse_beats = rmse_test < baseline_rmse
    mae_beats = mae_test < baseline_mae

    print("\nFeasibility check vs baseline:")
    print(f"Beats baseline RMSE? {'YES' if rmse_beats else 'NO'}")
    print(f"Beats baseline MAE?  {'YES' if mae_beats else 'NO'}")
    print(f"Overall feasible?    {'YES' if (rmse_beats and mae_beats) else 'NO'}")

    if cross_algo:
        header = f"train={train_algo}, test={test_algo}"
    else:
        header = args.algo or "all_algos"

    scale_label = "on" if args.scale else "off"
    print(f"\n=== Model summary (algos={header}, mode={args.mode}, model={args.model}, scale={scale_label}) ===")
    print(f"Train R^2: {r2_train:.4f}")
    print(f"Test  R^2: {r2_test:.4f}")
    print(f"Test  RMSE (J): {rmse_test:.6f}")
    print(f"Test  MAE  (J): {mae_test:.6f}")

    print(f"\nBaseline (mean predictor):")
    print(f"Baseline RMSE (J): {baseline_rmse:.6f}")
    print(f"Baseline MAE  (J): {baseline_mae:.6f}")

    # Extract coefficients (handle pipeline vs direct estimator)
    if hasattr(model, "named_steps"):
        reg = model.named_steps["reg"]
        coefs = reg.coef_
        coef_note = "(standardised predictors: 1 SD change)"
    else:
        coefs = model.coef_
        coef_note = "(raw predictors: 1 unit change)"

    print(f"\nCoefficients {coef_note}:")
    for name, coef in zip(feature_cols, coefs):
        print(f"  {name:15s} -> {coef:.6e}")

if __name__ == "__main__":
    main()
