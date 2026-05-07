#!/usr/bin/env python3
"""
Iterative Refinement — entry point.

Reads config.yaml, resolves examples, instantiates models, and runs the
pipeline for each example. Outputs go to output/<run_id>/.

Usage:
    python run.py                                         # all examples in dataset
    python run.py --ids codepen-ABKVZX-AbRrvV            # specific examples
    python run.py --start 0 --end 10                     # index slice
    python run.py --config my_config.yaml                # custom config
    python run.py --dataset-dir /path/to/dataset         # override dataset path
    python run.py --gemini-fps 10                        # override fps for all gemini stages
    python run.py --test-set path/to/metadata.jsonl      # restrict to test set IDs only
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────

EXPERIMENT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = EXPERIMENT_DIR.parent.parent
V3_NODE_MODULES = REPO_ROOT / "data_processing" / "video_processing" / "v3" / "node_modules"
LOCAL_NODE_MODULES = EXPERIMENT_DIR / "node_modules"


# ── Setup ─────────────────────────────────────────────────────────────────

def ensure_node_modules():
    """Symlink v3's node_modules here so recorder.js can find puppeteer-capture."""
    if LOCAL_NODE_MODULES.exists():
        return
    if not V3_NODE_MODULES.exists():
        print(
            f"ERROR: node_modules not found at {V3_NODE_MODULES}\n"
            "Run `npm install` inside data_processing/video_processing/v3/ first.",
            file=sys.stderr,
        )
        sys.exit(1)
    os.symlink(V3_NODE_MODULES, LOCAL_NODE_MODULES)
    print(f"Created node_modules symlink → {V3_NODE_MODULES}")


def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Dataset discovery ─────────────────────────────────────────────────────

def find_examples(dataset_dir: Path) -> list[tuple[str, Path]]:
    """
    Return sorted list of (example_id, gt_video_path) for all examples
    in dataset_dir that have an animation.mp4.
    """
    results = []
    for d in sorted(dataset_dir.iterdir()):
        if d.is_dir() and d.name.startswith("codepen-"):
            mp4 = d / "animation.mp4"
            if mp4.exists():
                results.append((d.name, mp4))
    return results


# ── Model instantiation ───────────────────────────────────────────────────

def build_models(cfg: dict):
    """Instantiate generator, critic, and editor models from config."""
    from modules.models import get_model
    from modules.pipeline import Models

    def make(role: str):
        role_cfg = cfg[role]
        model_name = role_cfg["model"]
        model_cfg = role_cfg[model_name]
        return get_model(model_name, model_cfg), model_name

    generator, gen_name = make("generator")
    critic, crt_name = make("critic")
    editor, edt_name = make("editor")

    return Models(
        generator=generator,
        critic=critic,
        editor=editor,
        generator_name=gen_name,
        critic_name=crt_name,
        editor_name=edt_name,
    )


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Iterative refinement pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--dataset-dir", help="Override dataset.local_dir from config")
    parser.add_argument("--ids", nargs="+", help="Specific example IDs to process")
    parser.add_argument("--start", type=int, default=0, help="Start index (ignored if --ids)")
    parser.add_argument("--end", type=int, default=None, help="End index (ignored if --ids)")
    parser.add_argument("--output-dir", help="Override output directory (skips timestamp)")
    parser.add_argument("--gemini-fps", type=float, default=None,
                        help="Override fps for all gemini stage configs")
    parser.add_argument("--include-target-video", action="store_true", default=None,
                        help="Override editor.include_target_video to True")
    parser.add_argument("--test-set", metavar="METADATA_JSONL",
                        help="Path to metadata.jsonl; restricts examples to those IDs only")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = EXPERIMENT_DIR / config_path
    cfg = load_config(config_path)

    # Apply --gemini-fps override to all stages that use gemini
    if args.gemini_fps is not None:
        for stage in ("generator", "critic", "editor"):
            if "gemini" in cfg.get(stage, {}):
                cfg[stage]["gemini"]["fps"] = args.gemini_fps

    # Apply --include-target-video override
    if args.include_target_video:
        cfg["editor"]["include_target_video"] = True

    # ── Resolve dataset dir ───────────────────────────────────────────────
    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
    else:
        dataset_dir = REPO_ROOT / cfg["dataset"]["local_dir"]
    if not dataset_dir.exists():
        print(f"ERROR: dataset dir not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Resolve output dir ────────────────────────────────────────────────
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        run_dir = EXPERIMENT_DIR / "output" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Setup ─────────────────────────────────────────────────────────────
    ensure_node_modules()

    # Save the effective config (post CLI-overrides) so each run is self-documenting
    (run_dir / "config.yaml").write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")

    # ── Logging ───────────────────────────────────────────────────────────
    from modules.logger import RunLogger
    logger = RunLogger(run_dir, run_id)
    logger.info("=== iterative refinement run %s ===", run_id)
    logger.info("Config:   %s", config_path)
    logger.info("Dataset:  %s", dataset_dir)
    logger.info("Output:   %s", run_dir)

    # ── Examples ──────────────────────────────────────────────────────────
    all_examples = find_examples(dataset_dir)

    # Filter to test set IDs if --test-set is given
    if args.test_set:
        test_ids = set()
        with open(args.test_set, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    test_ids.add(json.loads(line)["id"])
        all_examples = [(eid, vp) for eid, vp in all_examples if eid in test_ids]
        logger.info("Test set filter: %d examples", len(all_examples))

    if args.ids:
        id_set = set(args.ids)
        examples = [(eid, vp) for eid, vp in all_examples if eid in id_set]
    else:
        examples = all_examples[args.start : args.end]

    logger.info("Examples: %d", len(examples))
    def stage_info(role):
        m = cfg[role]["model"]
        fps = cfg[role].get(m, {}).get("fps", "?")
        return f"{m}(fps={fps})"

    logger.info(
        "Stages:   generator=%s  critic=%s  editor=%s",
        stage_info("generator"), stage_info("critic"), stage_info("editor"),
    )
    logger.info("Iterations: %d", cfg.get("max_iterations", 3))
    logger.info("Target video in editor: %s", cfg.get("editor", {}).get("include_target_video", False))

    if not examples:
        logger.warning("No examples found — exiting")
        return

    # ── Instantiate models ────────────────────────────────────────────────
    logger.info("Instantiating models...")
    from modules.pipeline import run_example
    models = build_models(cfg)
    logger.info("Models ready")

    # ── Run pipeline ──────────────────────────────────────────────────────
    results = []
    start_time = time.time()

    pbar = tqdm(examples, desc="examples", unit="ex")
    for idx, (example_id, gt_video) in enumerate(pbar):
        pbar.set_postfix_str(example_id[-12:])
        logger.info("[%d/%d] START %s", idx + 1, len(examples), example_id)
        trace = run_example(example_id, gt_video, cfg, models, run_dir, logger)
        status = trace["status"]
        results.append({
            "example_id": example_id,
            "status": status,
            "error": trace.get("error"),
            "iterations_completed": len(trace.get("iterations", [])),
        })
        ok = sum(1 for r in results if r["status"] == "success")
        pbar.set_postfix(ok=ok, err=len(results) - ok)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = round(time.time() - start_time, 1)
    successes = [r for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] != "success"]

    summary = {
        "run_id": run_id,
        "config": cfg,
        "stats": {
            "total": len(results),
            "success": len(successes),
            "error": len(errors),
            "elapsed_s": elapsed,
        },
        "results": results,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("=== Done ===")
    logger.info("  Total:    %d", len(results))
    logger.info("  Success:  %d", len(successes))
    logger.info("  Errors:   %d", len(errors))
    logger.info("  Time:     %.1fs", elapsed)
    logger.info("  Output:   %s", run_dir)

    if errors:
        logger.warning("Failed examples:")
        for r in errors:
            logger.warning("  %s: %s", r["example_id"], r["error"])


if __name__ == "__main__":
    main()
