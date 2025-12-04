"""
Train a simple regression model to predict per-run energy (J)
from OS-level performance counters and timing information.

This script is small on purpose, it parses CLI arguments, calls helper
functions in analysis.py, and fits a linear regression model.

Usage examples (from project root /energy_framework):

    All algorithms combined
    python3 python/model.py

    Only insertion sort
    python3 python/model.py --algo insertion

    Without time_ns as a feature (counters only)
    python3 python/model.py --no-time
"""

import argparse
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

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
        help=( """
Choose features to select:
time     -> time_ns only (baseline predictor)
counters -> use only perf counters 
all      -> time_ns + counters (default)
            """
        ), 
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    # Load data
    df = load_dataset(args.csv, algo_filter=args.algo)
    algo_label = args.algo or "all_algos"

    print(f"Using data from: {args.csv}")
    print(f"Algorithm filter: {algo_label}")
    print(f"Rows after filtering/cleaning: {len(df)}")

    # Build features/target
    if args.mode == "time":
        print("[INFO] Using time_ns only (baseline model)")
        X, y, feature_cols = build_features(
            df,
            use_time=True,
            extra_features=[],
            target_col="energy_j",
        )
        # force features to just ["time_ns"] if present
        feature_cols = [c for c in feature_cols if c == "time_ns"]
        X = df[feature_cols]
        y = df["energy_j"]

    elif args.mode == "counters":
        print("[INFO] Using counters only (no time_ns)")
        X, y, feature_cols = build_features(
            df,
            use_time=False,
            extra_features=[],
            target_col="energy_j",
        )

    else:  # "all"
        print("[INFO] Using time_ns + counters")
        X, y, feature_cols = build_features(
            df,
            use_time=True,
            extra_features=[],
            target_col="energy_j",
        )

    print(f"Using features: {feature_cols}")
    print(f"Total samples: {len(X)}")

    # Split into train/test
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=args.test_size, random_state=42
    )

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Quick evaluation (just numbers, no plots for now)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = root_mean_squared_error(y_test, y_pred_test)

    print("\n=== Model summary (Linear Regression) ===")
    print(f"Train R^2: {r2_train:.4f}")
    print(f"Test  R^2: {r2_test:.4f}")
    print(f"Test  RMSE (J): {rmse_test:.6f}")

    print("\nCoefficients:")
    for name, coef in zip(feature_cols, model.coef_):
        print(f"  {name:15s} -> {coef:.6e}")
    print(f"Intercept: {model.intercept_:.6e}")

    # (Optional later) could save the model parameters here
    # to a JSON or pickle under MODELS_DIR


if __name__ == "__main__":
    main()
