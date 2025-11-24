"""
Parses the CSV from benchmark.c
Understand the stdout line
Input: string containing stdout (possibly multiple lines) and output a dict
with parsed fields, example:

{
 "algo": "insertion",
 "n": 50000,
 "time_ns": 312938120,
 "energy_j": 2.23929293,
}
"""

def parse_benchmark_output(stdout: str) -> tuple[str, int, int, float] | None:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return None

    fields = lines[-1].split(",")
    if len(fields) < 4:
        return None

    algo, n_str, time_ns_str, energy_j_str = fields[:4]
    return algo, int(n_str), int(time_ns_str), float(energy_j_str)
