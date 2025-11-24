"""
Takes the output from perf stat and parses
"""

def parse_perf_stat(stderr_text: str) -> dict[str, int]:
    counters = {}
    for line in stderr_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue

        value_str, _, event = parts[:3]
        value_str = value_str.replace(",", "")

        try:
            value = int(value_str)
        except ValueError:
            continue

        counters[event.replace("-", "_")] = value

    return counters
