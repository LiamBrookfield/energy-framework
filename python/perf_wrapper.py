"""
This is the user facing script, accepts arguments from the command line and calls runner.py
generating a new row for each repition.

Input: CLI User Input
Output: Writes rows
"""

import argparse
from pathlib import Path
import csv
from runner import run_one
from csv_utils import ensure_csv_header

FIELDNAMES = [
    "algo", "n", "time_ns", "energy_j",
    "instructions", "cycles", "branches", "branch_misses", "cache_misses",
    "branch_miss_rate"
]

#not needed because of user_input.py
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

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)

        for n in args.sizes:
            for rep in range(args.reps):
                print(f"[INFO] Running {args.algo} n={n} rep={rep+1}/{args.reps}")
                row = run_one(args.algo, n, args.sudo)
                if row:
                    writer.writerow(row)
                    f.flush()

if __name__ == "__main__":
    main()
