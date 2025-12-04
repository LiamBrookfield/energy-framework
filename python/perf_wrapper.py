"""
This is the user facing script, accepts arguments from the command line and calls runner.py
generating a new row for each repition.

Input: CLI User Input
Output: Writes rows by calling runner.run_one() (gives the csv output)
"""

import argparse
from pathlib import Path
import csv
from runner import run_one
from csv_utils import ensure_csv_header
from datetime import datetime
import uuid

FIELDNAMES = [
    "run_id", "timestamp", "algo", "n", "time_ns", "energy_j",
    "instructions", "cycles", "branches", "branch_misses", "cache_misses",
    "branch_miss_rate"
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="insertion")
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--sudo", action="store_true")
    parser.add_argument("--output", default="data/raw/perf_runs.csv")
    return parser.parse_args()

def main():
    args = parse_args()
    path = Path(args.output)
    ensure_csv_header(path, FIELDNAMES)
    run_id = str(uuid.uuid4())

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)

        for n in args.sizes:
            for rep in range(args.reps):
                print(f"[INFO] Running {args.algo} n={n} rep={rep+1}/{args.reps}")
                row = run_one(args.algo, n, args.sudo)
                row["run_id"] = run_id
                timestamp = datetime.now().isoformat()
                row["timestamp"] = timestamp

                if row:
                    writer.writerow(row)
                    f.flush()
                elif row is None:
                    print("Row is None - runner.py is not working")
                    return

if __name__ == "__main__":
    main()

