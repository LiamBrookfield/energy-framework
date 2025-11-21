#!/usr/bin/env python3
"""
perf_wrapper.py

Run the C benchmark under perf stat, parse:
  - benchmark CSV from stdout: algo,n,time_ns,energy_j
  - perf counters from stderr

and write combined rows to a CSV in data/raw/.
"""

import argparse
import csv
import subprocess
from pathlib import Path


# From the Msc Project/energy-framework path
PERF_PATH = "/usr/lib/linux-tools/6.8.0-88-generic/perf"
BENCHMARK_PATH = "./bin/benchmark"

# Perf events to collect
PERF_EVENTS = [
    "instructions",
    "cycles",
    "branches",
    "branch-misses",
    "cache-misses",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run benchmark under perf stat and record counters + RAPL/time."
    )

    parser.add_argument(
        "--algo",
        type=str,
        default="insertion",
        help="Algorithm name to pass to the benchmark (default: insertion)",
    )

    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        required=True,
        help="List of input sizes n to benchmark (e.g. --sizes 50000 500000 1000000)",
    )

    parser.add_argument(
        "--reps",
        type=int,
        default=3,
        help="Number of repetitions per size (default: 3)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/perf_runs.csv",
        help="Output CSV path (default: data/raw/perf_runs.csv)",
    )

    parser.add_argument(
        "--sudo",
        action="store_true",
        help="Run perf with sudo (use if perf requires elevated permissions).",
    )

    return parser.parse_args()


def ensure_output_header(path: Path, fieldnames: list[str]) -> None:
    """
    Ensure the CSV file exists and has a header row.
    If the file doesn't exist or is empty, write the header.
    """
    if path.exists() and path.stat().st_size > 0:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def run_one(algo: str, n: int, use_sudo: bool) -> dict | None:
    """
    Run a single benchmark under perf and return a dict with:
      - algo, n, time_ns, energy_j
      - instructions, cycles, branches, branch_misses, cache_misses
    Returns None if something went wrong.
    """
    # Build perf command
    cmd = []

    if use_sudo:
        cmd.append("sudo")

    cmd.extend(
        [
            PERF_PATH,
            "stat",
            "-x",
            ",",
            "-e",
            ",".join(PERF_EVENTS),
            "--",
            BENCHMARK_PATH,
            algo,
            str(n),
        ]
    )

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"[WARN] Command failed for algo={algo}, n={n}, rc={result.returncode}")
        print("[stderr]")
        print(result.stderr)
        return None

    # GEt the CSV format from stdout
    stdout_lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not stdout_lines:
        print(f"[WARN] No benchmark output on stdout for algo={algo}, n={n}")
        return None

    bench_line = stdout_lines[-1].strip()
    bench_fields = bench_line.split(",")

    if len(bench_fields) < 4:
        print(f"[WARN] Unexpected benchmark CSV format: {bench_line}")
        return None

    algo_out, n_str, time_ns_str, energy_j_str = bench_fields[:4]

    # Parse perf stats from stderr
    counters = parse_perf_stat(result.stderr)

    # Build a single row dict
    row: dict[str, str | int | float] = {
        "algo": algo_out,
        "n": int(n_str),
        "time_ns": int(time_ns_str),
        "energy_j": float(energy_j_str),
    }

    # Add raw counters
    for ev in PERF_EVENTS:
        key = ev.replace("-", "_")
        row[key] = counters.get(key, "")

    # Derived metric: branch miss rate
    branches = row.get("branches")
    branch_misses = row.get("branch_misses")

    try:
        if isinstance(branches, int) and branches > 0 and isinstance(
            branch_misses, int
        ):
            row["branch_miss_rate"] = branch_misses / branches
        else:
            row["branch_miss_rate"] = ""
    except Exception:
        row["branch_miss_rate"] = ""

    return row


def parse_perf_stat(stderr_text: str) -> dict[str, int]:
    """
    Parse perf stat -x , output from stderr.

    Example line:
        4373732784,,instructions,313230897,100.00,5.52,insn per cycle

    Mainly care about:
        value (field 0)
        event (field 2)
    """
    counters: dict[str, int] = {}

    for line in stderr_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue

        value_str, _, event = parts[:3]

        # Skip lines that aren't actual counts
        if value_str in ("<not supported>", "<not counted>"):
            continue

        # Some locales print in an insane way
        value_str_clean = value_str.replace(",", "")

        try:
            value = int(value_str_clean)
        except ValueError:
            continue

        event_key = event.replace("-", "_")
        counters[event_key] = value

    return counters


def main() -> None:
    args = parse_args()

    output_path = Path(args.output)

    # Define CSV columns
    fieldnames = [
        "algo",
        "n",
        "time_ns",
        "energy_j",
        "instructions",
        "cycles",
        "branches",
        "branch_misses",
        "cache_misses",
        "branch_miss_rate",
    ]

    ensure_output_header(output_path, fieldnames)

    with output_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        for n in args.sizes:
            for rep in range(args.reps):
                print(f"[INFO] Running algo={args.algo}, n={n}, rep={rep+1}/{args.reps}")
                row = run_one(args.algo, n, use_sudo=args.sudo)
                if row is None:
                    print("[WARN] Skipping row due to errors.")
                    continue
                writer.writerow(row)
                f.flush()


if __name__ == "__main__":
    main()
