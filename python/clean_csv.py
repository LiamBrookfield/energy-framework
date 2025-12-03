"""
Just a quality of life sc



import argparse
from pathlib import Path

import pandas as pd

from paths import RAW_PERF_CSV


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove rows for a given algorithm from a perf_runs.csv file."
    )
    parser.add_argument(
        "--algo",
        required=True,
        help="Algorithm name to remove (e.g. 'insertion', 'merge').",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=RAW_PERF_CSV,
        help=f"Path to perf_runs.csv (default: {RAW_PERF_CSV})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show how many rows would be removed, but do not modify the file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path: Path = args.csv

    if not csv_path.exists():
        print(f"[ERROR] CSV not found at: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    if "algo" not in df.columns:
        print("[ERROR] CSV has no 'algo' column; nothing to do.")
        return

    before = len(df)
    to_remove = df[df["algo"] == args.algo]
    after = before - len(to_remove)

    print(f"[INFO] File: {csv_path}")
    print(f"[INFO] Rows before: {before}")
    print(f"[INFO] Rows matching algo='{args.algo}': {len(to_remove)}")
    print(f"[INFO] Rows after: {after}")

    if args.dry_run:
        print("[INFO] Dry run only; no changes written.")
        return

    if len(to_remove) == 0:
        print("[INFO] Nothing to remove; file unchanged.")
        return

    # Make a simple backup
    backup_path = csv_path.with_suffix(csv_path.suffix + ".bak")
    csv_path.replace(backup_path)
    print(f"[INFO] Backup written to: {backup_path}")

    # Write filtered file (keep header)
    df_filtered = df[df["algo"] != args.algo]
    df_filtered.to_csv(csv_path, index=False)
    print(f"[INFO] Updated file written to: {csv_path}")


if __name__ == "__main__":
    main()
