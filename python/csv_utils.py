"""
All related I/O for CSVs, don't care about how perf or benchmark work, this module
just makes files and headers, and then appends rows to said files

Input: A path (str) for the file to append values to
       A list of fieldnames (column headers)
       dict that represents ONE row

Output: Writes str to disk, depending on path
        Returns none or bool (success/failure)
"""

from pathlib import Path
import csv

def ensure_csv_header(path: Path, fieldnames: list[str]):
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
