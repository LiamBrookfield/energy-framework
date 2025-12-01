"""
Shared path utilities, hopefully this should be safe for any other script to call.
Purpose is to keep everything consistent about where the project root is and where
the data is (data/raw data/processed)

I'll probably call this from visualise.py - maybe the regression script too.
"""

from pathlib import Path

# Absolute path to the project root: .../energy-framework
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Canonical raw CSV path
RAW_PERF_CSV = RAW_DATA_DIR / "perf_runs.csv"

# Output directories
PLOTS_DIR = DATA_DIR / "plots"
MODELS_DIR = DATA_DIR / "models"


def ensure_dirs() -> None:
    """
    Create standard output directories if they do not exist.
    Safe to call from any script (please I hope).
    """
    for d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, PLOTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
