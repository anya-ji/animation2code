#!/usr/bin/env python3
"""
Run experimental pipeline variants.

Usage:
    python run_experiment.py baseline --config configs/config.yaml --start 0 --end 5
    python run_experiment.py describe_first --config configs/config.yaml --ids codepen-XXXX
    python run_experiment.py all --config configs/config.yaml --start 0 --end 5
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from modules.logger import RunLogger
from modules.models import get_model
from modules.pipeline import Models, run_example
from modules.pipeline_experiments import (
    run_caption_bridge,
    run_describe_first,
    run_dual_critic,
)

EXPERIMENTS = {
    "baseline": "Baseline iterative refinement",
    "describe_first": "Describe-first (ChartIR-based)",
    "dual_critic": "Dual critic (METAL-based)",
    "caption_bridge": "Caption bridge (PhyT2V-based)",
    "all": "Run all four variants",
}


def discover_examples(dataset_dir: Path, args) -> list[str]:
    all_ids = sorted(
        d.name for d in dataset_dir.iterdir()
        if d.is_dir() and (d / "animation.mp4").exists()
    )
    if args.ids:
        return [i for i in args.ids if i in set(all_ids)]
    start = args.start or 0
    end = args.end or len(all_ids)
    return all_ids[start:end]


def run_single_experiment(experiment, example_ids, dataset_dir, cfg, models, args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if getattr(args, "run_dir", None):
        run_dir = Path(args.run_dir)
        run_name = run_dir.name
    else:
        run_name = f"{experiment}_{timestamp}"
        run_dir = ROOT / "output" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_record:
        cfg["recorder"]["skip"] = True

    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)
    with open(run_dir / "experiment_info.json", "w") as f:
        json.dump({"experiment": experiment, "description": EXPERIMENTS[experiment],
                    "example_ids": example_ids}, f, indent=2)

    logger = RunLogger(run_dir, run_name)
    logger.info("=" * 60)
    logger.info("Experiment: %s — %s", experiment, EXPERIMENTS[experiment])
    logger.info("Examples: %d", len(example_ids))
    logger.info("Output: %s", run_dir)
    logger.info("=" * 60)

    results = []
    pbar = tqdm(example_ids, desc=f"{experiment}", unit="ex")
    for eid in pbar:
        gt_video = dataset_dir / eid / "animation.mp4"
        if not gt_video.exists():
            logger.warning("Skipping %s: no animation.mp4", eid)
            continue

        pbar.set_postfix(example=eid[:30])
        logger.info("[%s] starting", eid)
        t0 = time.time()

        try:
            if experiment == "baseline":
                trace = run_example(eid, gt_video, cfg, models, run_dir, logger)
            elif experiment == "describe_first":
                trace = run_describe_first(eid, gt_video, cfg, models, run_dir, logger)
            elif experiment == "dual_critic":
                trace = run_dual_critic(eid, gt_video, cfg, models, run_dir, logger)
            elif experiment == "caption_bridge":
                trace = run_caption_bridge(eid, gt_video, cfg, models, run_dir, logger)

            elapsed = round(time.time() - t0, 1)
            status = trace.get("status", "unknown")
            results.append({"example": eid, "status": status, "elapsed_s": elapsed})
            pbar.set_postfix(example=eid[:20], status=status, time=f"{elapsed}s")
            logger.info("[%s] %s in %.1fs", eid, status, elapsed)

        except Exception as exc:
            elapsed = round(time.time() - t0, 1)
            results.append({"example": eid, "status": "error", "elapsed_s": elapsed, "error": str(exc)})
            logger.exception(f"[{eid}] fatal", exc)
            pbar.set_postfix(example=eid[:20], status="ERROR")

    summary = {
        "experiment": experiment,
        "total": len(results),
        "success": sum(1 for r in results if r["status"] == "success"),
        "error": sum(1 for r in results if r["status"] != "success"),
        "total_time_s": sum(r["elapsed_s"] for r in results),
        "results": results,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("Done: %d success, %d error, %.1f min total",
                summary["success"], summary["error"], summary["total_time_s"] / 60)
    logger.info("Output: %s", run_dir)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Run iterative refinement experiments")
    parser.add_argument("experiment", choices=list(EXPERIMENTS.keys()))
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--dataset-dir", default=str(REPO_ROOT / "dataset"))
    parser.add_argument("--skip-record", action="store_true",
                        help="Skip recording (generate code only, record locally later)")
    parser.add_argument("--run-dir", default=None,
                        help="Override output directory (skips timestamp). Use to resume / share iter_0.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.max_iterations:
        cfg["max_iterations"] = args.max_iterations

    dataset_dir = Path(args.dataset_dir)
    example_ids = discover_examples(dataset_dir, args)
    if not example_ids:
        print("No examples found.")
        return

    print(f"Found {len(example_ids)} examples")
    print(f"Experiment: {args.experiment} — {EXPERIMENTS[args.experiment]}")

    gen_name = cfg["generator"]["model"]
    crt_name = cfg["critic"]["model"]
    edt_name = cfg["editor"]["model"]

    print(f"Loading models: gen={gen_name}, critic={crt_name}, editor={edt_name}")
    models = Models(
        generator=get_model(gen_name, cfg["generator"][gen_name]),
        critic=get_model(crt_name, cfg["critic"][crt_name]),
        editor=get_model(edt_name, cfg["editor"][edt_name]),
        generator_name=gen_name,
        critic_name=crt_name,
        editor_name=edt_name,
    )
    print("Models loaded.")

    if args.experiment == "all":
        for exp in ["baseline", "describe_first", "dual_critic", "caption_bridge"]:
            print(f"\n{'='*60}\n{exp}: {EXPERIMENTS[exp]}\n{'='*60}")
            run_single_experiment(exp, example_ids, dataset_dir, cfg, models, args)
    else:
        run_single_experiment(args.experiment, example_ids, dataset_dir, cfg, models, args)


if __name__ == "__main__":
    main()
