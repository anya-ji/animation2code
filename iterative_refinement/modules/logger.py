"""Logging for the iterative refinement pipeline.

RunLogger writes to the console and to run.log in the run directory. ExampleTrace collects each
example's history in trace.json, so a partial trace survives a crash.
"""

import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path


class RunLogger:
    """Thin wrapper around logging.Logger that writes to file + stdout."""

    def __init__(self, run_dir: Path, run_id: str):
        log_path = run_dir / "run.log"
        self.log_path = log_path

        self.logger = logging.getLogger(f"iterref.{run_id}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        self.logger.addHandler(ch)

    def info(self, msg: str, *args):
        self.logger.info(msg, *args)

    def warning(self, msg: str, *args):
        self.logger.warning(msg, *args)

    def error(self, msg: str, *args):
        self.logger.error(msg, *args)

    def debug(self, msg: str, *args):
        self.logger.debug(msg, *args)

    def exception(self, msg: str, exc: Exception):
        """Log an exception with full traceback to the file."""
        tb = traceback.format_exc()
        self.logger.error("%s\n%s", msg, tb)


class ExampleTrace:
    """Record one example's pipeline run in trace.json. Call save() after each stage."""

    def __init__(self, example_id: str, example_dir: Path):
        self.example_id = example_id
        self.example_dir = example_dir
        self.trace_path = example_dir / "trace.json"

        self.data: dict = {
            "example_id": example_id,
            "status": "running",
            "error": None,
            "started_at": datetime.utcnow().isoformat() + "Z",
            "finished_at": None,
            "iterations": [],
        }

    def add_iter_generate(
        self,
        iter_idx: int,
        model: str,
        html_chars: int,
        raw_output_file: str,
        duration_s: float,
    ):
        """Record the initial generation step (iter_0)."""
        self.data["iterations"].append({
            "iter": iter_idx,
            "stage": "generate",
            "model": model,
            "html_chars": html_chars,
            "raw_output_file": raw_output_file,
            "record_duration_s": duration_s,
        })
        self.save()

    def add_iter_refine(
        self,
        iter_idx: int,
        critic_model: str,
        editor_model: str,
        feedback_chars: int,
        html_chars: int,
        raw_critic_file: str,
        raw_editor_file: str,
        duration_s: float,
    ):
        """Record a refinement step (iter_1 onwards)."""
        self.data["iterations"].append({
            "iter": iter_idx,
            "stage": "refine",
            "critic_model": critic_model,
            "editor_model": editor_model,
            "feedback_chars": feedback_chars,
            "html_chars": html_chars,
            "raw_critic_file": raw_critic_file,
            "raw_editor_file": raw_editor_file,
            "record_duration_s": duration_s,
        })
        self.save()

    def mark_success(self):
        self.data["status"] = "success"
        self.data["finished_at"] = datetime.utcnow().isoformat() + "Z"
        self.save()

    def mark_error(self, error_msg: str, stage: str = ""):
        self.data["status"] = "error"
        self.data["error"] = f"{stage}: {error_msg}" if stage else error_msg
        self.data["finished_at"] = datetime.utcnow().isoformat() + "Z"
        self.save()

    def save(self):
        self.trace_path.write_text(
            json.dumps(self.data, indent=2), encoding="utf-8"
        )
