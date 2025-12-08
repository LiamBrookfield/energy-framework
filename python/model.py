"""
This script parses CLI arguments, calls helper
functions in analysis.py, and fits a linear regression model and a ridge regression model.

Usage example python3 python/model.py --algo quick --model ridge
"""

import argparse
from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge
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
    "--test-algo",
    type=str,
    default=None,
    help=(
        "Optional: if set, train on --algo and evaluate on this other algo.\n"
        "If not set, use usual random train/test split within --algo (or all algos)."
    )
)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    # choose regression type based on CLI flag
    def make_model(kind: str):
        if kind == "linear":
            return LinearRegression()
        elif kind == "ridge":
            # Might tune this value later
            return Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown model kind: {kind}")

    # Load data and apply optional algorithm filter
    df = load_dataset(args.csv, algo_filter=None)

    if args.algo is not None:
        df = df[df["algo"] == args.algo]

    algo_label = args.algo or "all_algos"

    print(f"Using data from: {args.csv}")
    print(f"Algorithm filter: {algo_label}")
    print(f"Rows after filtering/cleaning: {len(df)}")

    if df.empty:
        print("[ERROR] No data after filtering; nothing to train on.")
        return

    # Build features and target according to --mode
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

    # Slice the dataframe into X (features) and y (target)
    X = df[feature_cols]
    y = df["energy_j"]

    print(f"Using features: {feature_cols}")
    print(f"Total samples: {len(X)}")

    # Train/test split
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=args.test_size, random_state=42
    )

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Train model
    model = make_model(args.model)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = root_mean_squared_error(y_test, y_pred_test)

    print(f"\n=== Model summary (algo={algo_label}, mode={args.mode}, model={args.model}) ===")
    print(f"Train R^2: {r2_train:.4f}")
    print(f"Test  R^2: {r2_test:.4f}")
    print(f"Test  RMSE (J): {rmse_test:.6f}")

    print("\nCoefficients:")
    for name, coef in zip(feature_cols, model.coef_):
        print(f"  {name:15s} -> {coef:.6e}")

if __name__ == "__main__":
    main()
