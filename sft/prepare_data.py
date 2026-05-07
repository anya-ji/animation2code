"""
Data preparation for Qwen3-VL-8B SFT.

Reads train/val splits from the dataset and converts them to LLaMA-Factory's
ShareGPT video format.

Usage:
    python sft/prepare_data.py --data-dir /path/to/dataset --output-dir sft/data

Output:
    sft/data/train.json
    sft/data/val.json
    sft/data/dataset_info.json
"""

import argparse
import json
from pathlib import Path

# ── Prompts ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert web developer specializing in CSS animations and JavaScript.\n"
    "Your task is to analyze an animation video and generate the HTML/CSS/JavaScript code that recreates it.\n"
    "Focus on accuracy - match the timing, easing, colors, and movement as closely as possible."
)

USER_PROMPT = (
    "Watch this animation video carefully and generate the complete code to recreate it.\n\n"
    "Provide your response as a single, complete HTML document only.\n"
    "Include all CSS inside <style> tags and any JavaScript inside <script> tags.\n"
    "Keep the code concise — avoid generating hundreds of repetitive CSS rules.\n\n"
    "Be precise about:\n"
    "- Animation duration and timing functions\n"
    "- Colors and gradients\n"
    "- Transform properties (rotate, scale, translate)\n"
    "- Keyframe percentages\n"
    "- Any JavaScript timing or interactions"
)

# Approximate tokens = chars / 4 (conservative estimate for HTML-heavy text)
CHARS_PER_TOKEN = 4
CUTOFF_LEN = 16384


def estimate_tokens(num_chars: int) -> int:
    return num_chars // CHARS_PER_TOKEN


def process_split(split: str, data_root: Path, out_dir: Path) -> tuple[list[dict], int, int]:
    """Convert one split's metadata.jsonl to LLaMA-Factory ShareGPT format."""
    metadata_path = data_root / split / "metadata.jsonl"
    records = []
    over_limit = 0

    with open(metadata_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            full_html = item.get("full_html", "")
            video_filename = Path(item["file_name"]).name
            # Store absolute path so LLaMA-Factory can find the video
            video_path = str((data_root / split / "videos" / video_filename).resolve())

            total_chars = len(SYSTEM_PROMPT) + len(USER_PROMPT) + len(full_html)
            if estimate_tokens(total_chars) > CUTOFF_LEN:
                over_limit += 1

            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"<video>{USER_PROMPT}"},
                    {"role": "assistant", "content": full_html},
                ],
                "videos": [video_path],
            }
            records.append(record)

    return records, len(records), over_limit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="Dataset root containing train/ and val/ splits")
    parser.add_argument("--output-dir", default="sft/data",
                        help="Output directory for train.json, val.json, dataset_info.json")
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        print(f"\nProcessing {split}...")
        records, total, over_limit = process_split(split, data_root, out_dir)

        out_path = out_dir / f"{split}.json"
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        print(f"  Total examples : {total}")
        print(f"  Est. > {CUTOFF_LEN} tokens: {over_limit} ({100*over_limit/total:.1f}%)")
        print(f"  Written to     : {out_path}")

    # Write dataset_info.json for LLaMA-Factory
    dataset_info = {
        "anim2code_train": {"file_name": "train.json", "formatting": "sharegpt",
                            "columns": {"messages": "messages", "videos": "videos"}},
        "anim2code_val":   {"file_name": "val.json",   "formatting": "sharegpt",
                            "columns": {"messages": "messages", "videos": "videos"}},
    }
    with open(out_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\nDataset info written to {out_dir / 'dataset_info.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
