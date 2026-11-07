"""Core iterative refinement loop.

For each example:
  iter_0: The generator watches the gold video and writes HTML, which is recorded.
  iter_N: The critic compares the gold video with the last video, the editor applies its feedback,
          and the new HTML is recorded.

Every step's output is saved. If a step fails after its retries, the error goes in trace.json and
the pipeline moves on to the next example.
"""

import re
import time
from dataclasses import dataclass
from pathlib import Path

from modules.logger import ExampleTrace, RunLogger
from modules.models.base import BaseModel
from modules.recorder import Recorder


GENERATOR_SYSTEM = (
    "You are an expert web developer specializing in CSS animations and JavaScript. "
    "Your task is to analyze an animation video and generate the HTML/CSS/JavaScript "
    "code that recreates it. Focus on accuracy — match the timing, easing, colors, "
    "and movement as closely as possible."
)

GENERATOR_USER = (
    "Watch this animation video carefully and generate the complete code to recreate it.\n\n"
    "Provide your response as a single, complete HTML document only.\n"
    "Include all CSS inside <style> tags and any JavaScript inside <script> tags.\n\n"
    "Be precise about:\n"
    "- Animation duration and timing functions\n"
    "- Colors and gradients\n"
    "- Transform properties (rotate, scale, translate)\n"
    "- Keyframe percentages\n"
    "- Any JavaScript timing or interactions"
)

CRITIC_SYSTEM = (
    "You are a visual expert who compares web animations. "
    "Your job is to identify differences between a target animation and a current "
    "attempt, then describe exactly what code changes would make them match."
)

CRITIC_USER = (
    "You are given two animation videos:\n"
    "  1. TARGET — the animation we want to recreate.\n"
    "  2. CURRENT — the latest generated attempt.\n\n"
    "Analyze them carefully and provide:\n"
    "  (a) A concise numbered list of specific visual differences "
    "(colors, shapes, sizes, positions, motion paths, timing, easing, etc.).\n"
    "  (b) For each difference, a concrete suggestion for what to change in the code.\n\n"
    "Be specific and actionable. Prioritize the most impactful fixes first.\n"
    "Do NOT include any code — only describe what needs to change and why."
)

EDITOR_SYSTEM = (
    "You are an expert web developer. You will receive an HTML/CSS/JavaScript "
    "animation document and specific visual feedback on how to improve it. "
    "Your job is to produce a revised, complete HTML document that addresses all feedback."
)

EDITOR_USER_TEMPLATE = (
    "{target_video_note}"
    "Here is the current HTML code:\n\n"
    "```html\n{current_code}\n```\n\n"
    "Here is the visual feedback on what needs to change:\n\n"
    "{feedback}\n\n"
    "Generate an improved version of the complete HTML document that addresses "
    "all the feedback above. Output only a single, complete HTML document — "
    "no explanation, no markdown outside the document."
)


def parse_html(raw: str) -> tuple[str, str]:
    """Extract the HTML document from a model response.

    Args:
        raw: Raw model output.

    Returns:
        (html, error), where error is an empty string on success.
    """
    fenced = re.search(r"```(?:\w+)?\s*(.*?)```", raw, re.DOTALL)
    cleaned = fenced.group(1).strip() if fenced else raw.strip()

    if not fenced:
        partial = re.search(r"```(?:\w+)?\s*(.*)", raw, re.DOTALL)
        if partial:
            cleaned = partial.group(1).strip()

    if "<html" in cleaned.lower():
        return cleaned, ""

    return raw.strip(), "Response did not contain an <html> tag"


def with_retries(fn, retries: int, backoff_s: float, logger: RunLogger, stage: str):
    """Call fn up to 1 + retries times, waiting backoff_s * 2^attempt between tries.

    Args:
        fn: Function to call.
        retries: Number of retries after the first attempt.
        backoff_s: Initial wait in seconds.
        logger: RunLogger.
        stage: Stage name for log messages.

    Raises the last exception if every attempt fails.
    """
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt < retries:
                wait = backoff_s * (2 ** attempt)
                logger.warning(
                    "[%s] attempt %d/%d failed: %s — retrying in %.0fs",
                    stage, attempt + 1, retries + 1, exc, wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "[%s] all %d attempts failed: %s", stage, retries + 1, exc
                )
                raise


@dataclass
class Models:
    generator: BaseModel
    critic: BaseModel
    editor: BaseModel
    generator_name: str
    critic_name: str
    editor_name: str


def run_example(
    example_id: str,
    gt_video_path: Path,
    cfg: dict,
    models: Models,
    run_dir: Path,
    logger: RunLogger,
) -> dict:
    """Run the baseline refinement loop for one example.

    Args:
        example_id: Example name.
        gt_video_path: Path to the gold video.
        cfg: Parsed config.
        models: Generator, critic and editor models.
        run_dir: Output directory for this run.
        logger: RunLogger.

    Returns:
        The trace dict, also written to trace.json.
    """
    example_dir = run_dir / example_id
    example_dir.mkdir(parents=True, exist_ok=True)

    trace = ExampleTrace(example_id, example_dir)
    recorder = Recorder(cfg["recorder"])

    max_iters = int(cfg.get("max_iterations", 3))
    retries = int(cfg.get("retries", 2))
    backoff = float(cfg.get("retry_backoff_s", 5))
    include_target_video = cfg.get("editor", {}).get("include_target_video", False)

    try:
        iter_dir = example_dir / "iter_0"
        iter_dir.mkdir(exist_ok=True)

        logger.info("[%s] iter_0: generating initial code", example_id)
        t0 = time.time()

        raw_gen = with_retries(
            lambda: models.generator.generate_code(
                video_path=gt_video_path,
                system_prompt=GENERATOR_SYSTEM,
                user_prompt=GENERATOR_USER,
            ),
            retries, backoff, logger, f"{example_id}/iter_0/generate",
        )

        (iter_dir / "raw_output.txt").write_text(raw_gen, encoding="utf-8")
        current_html, parse_err = parse_html(raw_gen)
        if parse_err:
            logger.warning("[%s] iter_0 parse warning: %s", example_id, parse_err)
            (iter_dir / "_parse_error.txt").write_text(parse_err, encoding="utf-8")
        (iter_dir / "full.html").write_text(current_html, encoding="utf-8")

        logger.info("[%s] iter_0: recording", example_id)
        mp4_path = iter_dir / "animation.mp4"
        with_retries(
            lambda: recorder.record(iter_dir / "full.html", mp4_path),
            retries, backoff, logger, f"{example_id}/iter_0/record",
        )

        current_video = mp4_path
        elapsed = round(time.time() - t0, 1)
        trace.add_iter_generate(
            iter_idx=0,
            model=models.generator_name,
            html_chars=len(current_html),
            raw_output_file="iter_0/raw_output.txt",
            duration_s=elapsed,
        )
        logger.info("[%s] iter_0 done in %.1fs", example_id, elapsed)

        for i in range(1, max_iters + 1):
            iter_dir = example_dir / f"iter_{i}"
            iter_dir.mkdir(exist_ok=True)

            logger.info("[%s] iter_%d: critiquing", example_id, i)
            t0 = time.time()

            raw_feedback = with_retries(
                lambda: models.critic.critique(
                    target_video=gt_video_path,
                    generated_video=current_video,
                    system_prompt=CRITIC_SYSTEM,
                    user_prompt=CRITIC_USER,
                ),
                retries, backoff, logger, f"{example_id}/iter_{i}/critique",
            )

            (iter_dir / "raw_critic.txt").write_text(raw_feedback, encoding="utf-8")
            feedback = raw_feedback.strip()
            (iter_dir / "feedback.txt").write_text(feedback, encoding="utf-8")

            logger.info("[%s] iter_%d: editing code", example_id, i)

            user_prompt = EDITOR_USER_TEMPLATE.format(
                target_video_note=(
                    "The first video above is the TARGET animation you must match.\n\n"
                    if include_target_video else ""
                ),
                current_code=current_html,
                feedback=feedback,
            )
            target_video_arg = gt_video_path if include_target_video else None

            raw_edit = with_retries(
                lambda: models.editor.edit_code(
                    current_code=current_html,
                    feedback=feedback,
                    system_prompt=EDITOR_SYSTEM,
                    user_prompt=user_prompt,
                    target_video=target_video_arg,
                ),
                retries, backoff, logger, f"{example_id}/iter_{i}/edit",
            )

            (iter_dir / "raw_editor.txt").write_text(raw_edit, encoding="utf-8")
            new_html, parse_err = parse_html(raw_edit)
            if parse_err:
                logger.warning("[%s] iter_%d edit parse warning: %s", example_id, i, parse_err)
                (iter_dir / "_parse_error.txt").write_text(parse_err, encoding="utf-8")
            (iter_dir / "full.html").write_text(new_html, encoding="utf-8")
            current_html = new_html

            logger.info("[%s] iter_%d: recording", example_id, i)
            mp4_path = iter_dir / "animation.mp4"
            with_retries(
                lambda: recorder.record(iter_dir / "full.html", mp4_path),
                retries, backoff, logger, f"{example_id}/iter_{i}/record",
            )
            current_video = mp4_path

            elapsed = round(time.time() - t0, 1)
            trace.add_iter_refine(
                iter_idx=i,
                critic_model=models.critic_name,
                editor_model=models.editor_name,
                feedback_chars=len(feedback),
                html_chars=len(new_html),
                raw_critic_file=f"iter_{i}/raw_critic.txt",
                raw_editor_file=f"iter_{i}/raw_editor.txt",
                duration_s=elapsed,
            )
            logger.info("[%s] iter_%d done in %.1fs", example_id, i, elapsed)

    except Exception as exc:
        stage = f"iter_{len(trace.data['iterations'])}"
        trace.mark_error(str(exc), stage=stage)
        logger.exception(f"[{example_id}] fatal error at {stage}", exc)
        return trace.data

    trace.mark_success()
    logger.info("[%s] complete (%d iterations)", example_id, max_iters)
    return trace.data
