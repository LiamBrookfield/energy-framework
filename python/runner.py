"""
Parses the command line arguments
loop over int: size (n) and
""" 

import subprocess
from perf_parser import parse_perf_stat
from bench_parser import parse_benchmark_output

PERF_PATH = "/usr/lib/linux-tools/6.8.0-88-generic/perf"
BENCHMARK_PATH = "./bin/benchmark"
PERF_EVENTS = ["instructions", "cycles", "branches", "branch-misses", "cache-misses"]

def run_one(algo: str, n: int, use_sudo: bool) -> dict | None:
    cmd = []
    if use_sudo:
        cmd.append("sudo")

    cmd.extend([
        PERF_PATH,
        "stat",
        "-x", ",",
        "-e", ",".join(PERF_EVENTS),
        "--",
        BENCHMARK_PATH,
        algo,
        str(n),
    ])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return None
        print("Runner is returning none, did you update Makefile?")

    bench = parse_benchmark_output(result.stdout)
    if bench is None:
        return None
        print("Runner is returning none, did you update Makefile?")

    algo_out, n_val, time_ns, energy_j = bench

    counters = parse_perf_stat(result.stderr)

    row = {
        "algo": algo_out,
        "n": n_val,
        "time_ns": time_ns,
        "energy_j": energy_j,
    }

    for ev in PERF_EVENTS:
        row[ev.replace("-", "_")] = counters.get(ev.replace("-", "_"), "")

    # derived metric
    try:
        bm = row["branch_misses"]
        b = row["branches"]
        if isinstance(b, int) and b > 0 and isinstance(bm, int):
            row["branch_miss_rate"] = bm / b
        else:
            row["branch_miss_rate"] = ""
    except:
        row["branch_miss_rate"] = ""

    return row
