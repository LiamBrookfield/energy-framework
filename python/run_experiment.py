

from user_input import get_user_inputs
from runner import run_one
from csv_utils import ensure_csv_header, append_row
from pathlib import Path

FIELDNAMES = [
    "algo", "n", "time_ns", "energy_j",
    "instructions", "cycles", "branches",
    "branch_misses", "cache_misses",
    "branch_miss_rate"
]

def main():
    algo, n, reps, use_sudo = user_prompt()

    output_path = Path("data/raw/user_runs.csv")
    ensure_output_header(output_path, FIELDNAMES)

    for i in range(reps):
        print(f"Running {algo}, n={n}, rep {i+1}/{reps}")
        row = run_one(algo, n, use_sudo)
        append_row(output_path, row)

if __name__ == "__main__":
    main()
