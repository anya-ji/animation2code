#!/usr/bin/env python3
"""Run baseline inference for a model config, writing <run_dir>/<id>/full.html."""
import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

BASELINES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASELINES_DIR))


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("baseline")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def main():
    parser = argparse.ArgumentParser(description="Run baseline model inference")
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    parser.add_argument("--start", type=int, default=0, help="Start index into example list")
    parser.add_argument("--end", type=int, default=None, help="End index into example list")
    parser.add_argument("--ids", nargs="+", help="Specific example IDs (overrides --start/--end)")
    parser.add_argument(
        "--output-dir",
        help="Resume: use this existing run dir instead of creating a new timestamped one",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = str(config.get("output", {}).get("suffix", "")).strip()
    run_name = f"{timestamp}_{suffix}" if suffix else timestamp

    _base_dir_cfg = config.get("output", {}).get("base_dir", "")
    _log_root = Path(_base_dir_cfg).parts[0] if _base_dir_cfg else model_name
    log_dir = BASELINES_DIR / _log_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}.log"
    logger = setup_logging(log_path)
    logger.info("Config:  %s", config_path)
    logger.info("Model:   %s", model_name)

    if args.output_dir:
        run_dir = Path(args.output_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        base_dir = BASELINES_DIR / config["output"]["base_dir"]
        run_dir = base_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, run_dir / "_config.yaml")

    logger.info("Output:  %s", run_dir)

    from shared.dataloader import load_examples
    examples, get_video_path = load_examples(config["data"])

    if args.ids:
        id_set = set(args.ids)
        examples = [e for e in examples if e in id_set]
    else:
        examples = examples[args.start : args.end]

    if args.output_dir:
        remaining = [
            e for e in examples
            if not (run_dir / e / "full.html").exists()
            and not (run_dir / e / "_error.txt").exists()
        ]
        logger.info(
            "Selected: %d  Already done: %d  Remaining: %d",
            len(examples), len(examples) - len(remaining), len(remaining),
        )
        examples = remaining

    logger.info("Examples to process: %d", len(examples))

    if not examples:
        logger.info("Nothing to process.")
        return

    if model_name == "gemini":
        from models.gemini import GeminiModel
        model = GeminiModel(config, logger)
    elif model_name == "qwen3vl":
        from models.qwen3vl import Qwen3VLModel
        model = Qwen3VLModel(config, logger)
    elif model_name == "gpt":
        from models.gpt import GptModel
        model = GptModel(config, logger)
    elif model_name == "openrouter":
        from models.openrouter import OpenRouterModel
        model = OpenRouterModel(config, logger)
    else:
        logger.error("Unknown model: %s", model_name)
        sys.exit(1)

    model.run(examples, get_video_path, run_dir)
    logger.info("Done. Output: %s", run_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        try:
            logging.getLogger("baseline").exception("Fatal error")
        except Exception:
            pass
        traceback.print_exc()
        sys.exit(1)
