#!/usr/bin/env python3
"""
Temporary script: re-processes only identical=false rows in comparison.csv,
adding hash1, hash2, total_frames, different_frames, max_pixel_diff, mean_pixel_diff.

Usage:
    uv run python rerun_false.py [--csv comparison.csv]
"""
import csv
import sys
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from check_deterministic import sha256, extract_frames, compare_frames, compare_two

import tempfile
import shutil
from tqdm import tqdm


FIELDNAMES = [
    "example", "identical", "method", "run1", "run2",
    "hash1", "hash2", "total_frames", "different_frames",
    "max_pixel_diff", "mean_pixel_diff",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=SCRIPT_DIR / "comparison.csv")
    args = parser.parse_args()
    csv_path = Path(args.csv)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    false_rows = [r for r in rows if r["identical"] == "false" and r["method"] != "missing"]
    print(f"Reprocessing {len(false_rows)} non-identical rows...")

    index = {r["example"]: r for r in rows}

    for row in tqdm(false_rows, desc="Comparing", unit="example"):
        v1, v2 = Path(row["run1"]), Path(row["run2"])
        if not v1.exists() or not v2.exists():
            continue

        h1 = sha256(v1)
        h2 = sha256(v2)

        tmp = Path(tempfile.mkdtemp())
        try:
            frames1 = extract_frames(v1, tmp / "v1")
            frames2 = extract_frames(v2, tmp / "v2")
            stats = compare_frames(frames1, frames2)
        finally:
            shutil.rmtree(tmp)

        row["hash1"] = h1
        row["hash2"] = h2
        row["total_frames"] = stats.get("total_frames", "")
        row["different_frames"] = stats.get("different_frames", "")
        row["max_pixel_diff"] = stats.get("max_pixel_diff", "")
        row["mean_pixel_diff"] = stats.get("mean_pixel_diff", "")

    # Ensure all rows have the new fields (fill blanks for identical=true rows)
    for row in rows:
        for field in ["hash1", "hash2", "total_frames", "different_frames", "max_pixel_diff", "mean_pixel_diff"]:
            row.setdefault(field, "")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {csv_path}")


if __name__ == "__main__":
    main()
