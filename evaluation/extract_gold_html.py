#!/usr/bin/env python3
"""Extract gold HTML files from metadata.jsonl into a flat directory.

Each entry with a 'full_html' field is written to {output_dir}/{id}.html.
Existing files are skipped (idempotent).

Usage:
    uv run python evaluation/extract_gold_html.py \
        --metadata _hf_staging/data/test/metadata.jsonl \
        --output-dir evaluation/gold_html
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Extract gold HTML from metadata.jsonl")
    parser.add_argument("--metadata", required=True, help="Path to metadata.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory for gold HTML files")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    with open(args.metadata) as f:
        for line in f:
            entry = json.loads(line)
            name = entry.get("id")
            html = entry.get("full_html", "")
            if not name or not html:
                continue
            dest = out / f"{name}.html"
            if dest.exists():
                skipped += 1
                continue
            dest.write_text(html, encoding="utf-8")
            written += 1

    print(f"Written {written} files, skipped {skipped} existing -> {out}")


if __name__ == "__main__":
    main()
