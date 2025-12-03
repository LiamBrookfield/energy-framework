"""
This isnt the main model, this is just for data loading, feature prep and utilities.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


# Numeric columns from the perf_runs.csv
NUMERIC_CANDIDATES = [
    "n",
    "time_ns",
    "energy_j",
    "instructions",
    "cycles",
    "branches",
    "branch_misses",
    "cache_misses",
    "branch_miss_rate"
]


def load_dataset(csv_path: Path, algo_filter: Optional[str] = None) -> pd.DataFrame:
    """
    Load perf/RAPL data from CSV and trying to add option to filter by algorithm -
    will definitely help later on.
    Ensures key numeric-looking columns are converted to numeric.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    if algo_filter is not None:
        df = df[df["algo"] == algo_filter].copy()

    if "energy_j" not in df.columns:
        raise ValueError("CSV is missing required column 'energy_j'.")

    # Coerce numeric-looking columns
    for col in NUMERIC_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing target
    df = df.dropna(subset=["energy_j"])

    if df.empty:
        raise RuntimeError("No data available after filtering/cleaning.")

    return df


def build_features(
    df: pd.DataFrame,
    use_time: bool = True,
    extra_features: Optional[List[str]] = None,
    target_col: str = "energy_j",
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Construct feature matrix X and target vector y from the DataFrame.

    By default uses:
        time_ns (optional), instructions, cycles, branches,
        branch_misses, cache_misses

    extra_features: list of additional column names to include if present.
    """
    base_features: List[str] = [
        "instructions",
        "cycles",
        "branches",
        "branch_misses",
        "cache_misses",
    ]
    if use_time:
        base_features.insert(0, "time_ns")

    if extra_features:
        base_features.extend(extra_features)

    # Keep only those columns that actually exist
    feature_cols = [c for c in base_features if c in df.columns]

    missing = [c for c in base_features if c not in df.columns]
    if missing:
        # Not fatal, but best I know if something isn't there
        print(f"[INFO] Skipping missing feature columns: {missing}")

    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Drop rows where any feature or target is NaN
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]

    if X.empty:
        raise RuntimeError("No rows left after feature/target preparation.")

    return X, y, feature_cols


def split_train_test(
    X, y, test_size: float = 0.2, random_state: int = 42
):
    """
    Simple train/test split wrapper so model.py doesn't
    have to import sklearn directly.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
