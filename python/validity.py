"""
Validity / robustness checks for perf_runs.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

FEATURES_ALL = ["time_ns", "instructions", "cycles", "branches", "branch_misses", "cache_misses"]
NS_PER_S = 1_000_000_000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--min-time-s", type=float, default=None)
    p.add_argument("--max-time-s", type=float, default=None)
    p.add_argument("--train", type=str, default=None, help="Train algorithm (for cross-algo checks)")
    p.add_argument("--test", type=str, default=None, help="Test algorithm (for cross-algo checks)")
    p.add_argument("--repeats", type=int, default=20, help="Repeated split count for stability checks")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot", action="store_true", help="Show quick diagnostic plots")
    return p.parse_args()


def filter_energy_positive(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["energy_j"] > 0.0].copy()
    removed = before - len(df)
    if before > 0:
        print(f"[INFO] energy_j>0 filter: removed {removed} rows ({removed/before:.2%})")
    return df


def filter_time_band(df: pd.DataFrame, min_s, max_s, label="") -> pd.DataFrame:
    before = len(df)
    if min_s is not None:
        df = df[df["time_ns"] >= int(min_s * NS_PER_S)]
    if max_s is not None:
        df = df[df["time_ns"] <= int(max_s * NS_PER_S)]
    df = df.copy()
    if min_s is not None or max_s is not None:
        print(f"[INFO] Time-band {label}: kept {len(df)}/{before} (removed {before-len(df)})")
    return df


def make_ridge(scale: bool) -> object:
    if scale:
        return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    return Ridge(alpha=1.0)


def summary_target(y: pd.Series, name: str):
    y = pd.Series(y).astype(float)
    print(f"[INFO] {name}: mean={y.mean():.6f} J, std={y.std(ddof=0):.6f} J, min={y.min():.6f}, max={y.max():.6f}")


def eval_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "r2": r2_score(y_test, pred),
        "rmse": root_mean_squared_error(y_test, pred),
        "mae": mean_absolute_error(y_test, pred),
        "pred": pred,
    }


def baseline_mean(y_train, y_test):
    mu = float(np.mean(y_train))
    pred = np.full(len(y_test), mu)
    return {
        "rmse": root_mean_squared_error(y_test, pred),
        "mae": mean_absolute_error(y_test, pred),
        "mu": mu,
    }


def robust_outlier_mask(residuals: np.ndarray, k: float = 6.0) -> np.ndarray:
    # Median Absolute Deviation
    r = np.asarray(residuals)
    med = np.median(r)
    mad = np.median(np.abs(r - med)) + 1e-12
    z = 0.6745 * (r - med) / mad
    return np.abs(z) <= k


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)

    # Basic integrity
    print(f"[INFO] Rows loaded: {len(df)}")
    print("[INFO] Rows by algorithm:")
    print(df["algo"].value_counts(dropna=False).to_string())

    # Ensure numeric columns are numeric
    for c in ["n", "time_ns", "energy_j"] + FEATURES_ALL[1:]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["algo", "n", "time_ns", "energy_j"])

    # Energy floor filter
    df = filter_energy_positive(df)

    # If cross-algo requested, run matched-duration checks + sensitivity
    if args.train and args.test:
        train_df = df[df["algo"] == args.train].copy()
        test_df = df[df["algo"] == args.test].copy()

        train_df = filter_time_band(train_df, args.min_time_s, args.max_time_s, label=f"(train={args.train})")
        test_df = filter_time_band(test_df, args.min_time_s, args.max_time_s, label=f"(test={args.test})")

        if train_df.empty or test_df.empty:
            print("[ERROR] No data left for train or test after filtering.")
            return

        X_train = train_df[FEATURES_ALL]
        y_train = train_df["energy_j"]
        X_test = test_df[FEATURES_ALL]
        y_test = test_df["energy_j"]

        summary_target(y_train, "y_train")
        summary_target(y_test, "y_test")

        # Compare scaled vs unscaled ridge (sensitivity)
        for scale in [True, False]:
            model = make_ridge(scale=scale)
            res = eval_model(model, X_train, y_train, X_test, y_test)
            bl = baseline_mean(y_train, y_test)

            label = "scale=on" if scale else "scale=off"
            print(f"\n=== Cross-algo ridge ({args.train}→{args.test}, {label}) ===")
            print(f"Test R^2: {res['r2']:.4f}")
            print(f"Test RMSE: {res['rmse']:.6f} J")
            print(f"Test MAE : {res['mae']:.6f} J")
            print(f"Baseline (train mean={bl['mu']:.6f}): RMSE={bl['rmse']:.6f}, MAE={bl['mae']:.6f}")

            # Outlier sensitivity on residuals (descriptive)
            residuals = (y_test.values - res["pred"])
            mask = robust_outlier_mask(residuals, k=6.0)
            removed = int((~mask).sum())
            if removed > 0 and removed < len(residuals):
                rmse_rob = root_mean_squared_error(y_test.values[mask], res["pred"][mask])
                mae_rob = mean_absolute_error(y_test.values[mask], res["pred"][mask])
                print(f"[INFO] Robust check (remove {removed} high-residual points): RMSE={rmse_rob:.6f}, MAE={mae_rob:.6f}")

            if args.plot:
                plt.figure()
                plt.scatter(y_test, res["pred"], s=12)
                plt.xlabel("Actual energy (J)")
                plt.ylabel("Predicted energy (J)")
                plt.title(f"Predicted vs Actual ({args.train}→{args.test}, {label})")
                plt.show()

                plt.figure()
                plt.scatter(X_test["time_ns"] / 1e9, residuals, s=12)
                plt.axhline(0, linewidth=1)
                plt.xlabel("Runtime (s)")
                plt.ylabel("Residual (J)")
                plt.title(f"Residuals vs Runtime ({args.train}→{args.test}, {label})")
                plt.show()

    else:
        # Within-algo split stability (repeat splits)
        # Choose one algo or do all by looping.
        for algo in sorted(df["algo"].unique()):
            sub = df[df["algo"] == algo].copy()
            sub = filter_time_band(sub, args.min_time_s, args.max_time_s, label=f"(algo={algo})")
            if len(sub) < 50:
                continue

            X = sub[FEATURES_ALL]
            y = sub["energy_j"]

            rmses = []
            maes = []
            r2s = []

            for i in range(args.repeats):
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.2, random_state=args.seed + i
                )
                model = make_ridge(scale=True)
                res = eval_model(model, X_tr, y_tr, X_te, y_te)
                r2s.append(res["r2"])
                rmses.append(res["rmse"])
                maes.append(res["mae"])

            print(f"\n=== Within-algo stability (ridge, scale=on, algo={algo}) ===")
            print(f"Repeats: {args.repeats}")
            print(f"R^2  mean±sd: {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
            print(f"RMSE mean±sd: {np.mean(rmses):.6f} ± {np.std(rmses):.6f}")
            print(f"MAE  mean±sd: {np.mean(maes):.6f} ± {np.std(maes):.6f}")

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
