"""Refinement variants based on prior work, using the same models, recorder and trace as pipeline.py.

  describe_first (ChartIR): Describe the animation in words before generating code.
  dual_critic (METAL): Run a visual critic, then a code critic.
  caption_bridge (PhyT2V): Caption both videos and compare the captions as text.
"""

import time
from pathlib import Path

from modules.logger import ExampleTrace, RunLogger
from modules.pipeline import (
    EDITOR_SYSTEM,
    EDITOR_USER_TEMPLATE,
    GENERATOR_SYSTEM,
    GENERATOR_USER,
    Models,
    parse_html,
    with_retries,
)
from modules.recorder import Recorder


def _run_iter_0(example_id, gt_video_path, models, recorder, example_dir,
                retries, backoff, logger, trace):
    """Run iter_0 (initial generation). Returns (current_html, current_video)."""
    iter_dir = example_dir / "iter_0"
    iter_dir.mkdir(exist_ok=True)

    full_html_path = iter_dir / "full.html"
    mp4_path = iter_dir / "animation.mp4"
    if full_html_path.exists() and mp4_path.exists():
        logger.info("[%s] iter_0: REUSING existing shared iter_0 from %s", example_id, iter_dir)
        current_html = full_html_path.read_text(encoding="utf-8")
        trace.add_iter_generate(
            iter_idx=0, model=f"{models.generator_name}(shared)",
            html_chars=len(current_html), raw_output_file="iter_0/raw_output.txt",
            duration_s=0.0,
        )
        return current_html, mp4_path

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
    (iter_dir / "full.html").write_text(current_html, encoding="utf-8")

    logger.info("[%s] iter_0: recording", example_id)
    mp4_path = iter_dir / "animation.mp4"
    with_retries(
        lambda: recorder.record(iter_dir / "full.html", mp4_path),
        retries, backoff, logger, f"{example_id}/iter_0/record",
    )

    elapsed = round(time.time() - t0, 1)
    trace.add_iter_generate(
        iter_idx=0, model=models.generator_name,
        html_chars=len(current_html), raw_output_file="iter_0/raw_output.txt",
        duration_s=elapsed,
    )
    logger.info("[%s] iter_0 done in %.1fs", example_id, elapsed)
    return current_html, mp4_path


DESCRIBE_SYSTEM = (
    "You are an expert at analyzing web animations. Describe animations in "
    "precise, structured detail that a developer could use to write the code."
)

DESCRIBE_USER = (
    "Watch this animation video carefully and provide a structured description:\n\n"
    "1. OBJECTS: List every visual element (shapes, text, images) with their "
    "colors, sizes, and initial positions.\n"
    "2. MOTION: For each moving element, describe the trajectory, direction, "
    "and distance of movement.\n"
    "3. TIMING: Describe duration, delays, easing functions (linear, ease-in, "
    "ease-out, bounce, etc.), and whether animations loop.\n"
    "4. SEQUENCE: What happens first, second, third? Are things simultaneous "
    "or sequential?\n"
    "5. EFFECTS: Any gradients, shadows, opacity changes, scale changes, rotations?\n\n"
    "Be extremely precise about numbers (pixels, seconds, degrees)."
)

DESCRIBE_THEN_CODE_USER = (
    "Here is a detailed description of an animation:\n\n"
    "{description}\n\n"
    "Generate a single, complete HTML document that recreates this animation exactly.\n"
    "Include all CSS in <style> tags and JavaScript in <script> tags.\n"
    "Match the timing, easing, colors, and movement as precisely as possible."
)

DESCRIBE_DIFF_USER = (
    "You are given two animation videos:\n"
    "  1. TARGET — the animation we want to recreate.\n"
    "  2. CURRENT — the latest generated attempt.\n\n"
    "And the original structured description of the target:\n"
    "{description}\n\n"
    "Provide a HOLISTIC difference description covering ALL dimensions simultaneously:\n"
    "  - Objects: missing, extra, wrong color/size?\n"
    "  - Motion: wrong direction, speed, trajectory?\n"
    "  - Timing: wrong duration, easing, sequence?\n"
    "  - Effects: missing shadows, opacity, transforms?\n\n"
    "For each difference, suggest a specific code change. Do NOT include code."
)


def run_describe_first(
    example_id: str, gt_video_path: Path, cfg: dict,
    models: Models, run_dir: Path, logger: RunLogger,
) -> dict:
    """Describe the animation first, then generate code and critique against that description.

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
        logger.info("[%s] describe-first: generating description", example_id)
        t0 = time.time()
        iter_dir = example_dir / "iter_0"
        iter_dir.mkdir(exist_ok=True)

        full_html_path = iter_dir / "full.html"
        shared_mp4 = iter_dir / "animation.mp4"
        if full_html_path.exists() and shared_mp4.exists():
            logger.info("[%s] iter_0: REUSING existing shared iter_0 from %s", example_id, iter_dir)
            current_html = full_html_path.read_text(encoding="utf-8")
            current_video = shared_mp4
            trace.add_iter_generate(
                iter_idx=0, model=f"{models.generator_name}(shared)",
                html_chars=len(current_html), raw_output_file="iter_0/raw_output.txt",
                duration_s=0.0,
            )
        else:
            description = with_retries(
                lambda: models.generator.generate_code(
                    video_path=gt_video_path,
                    system_prompt=DESCRIBE_SYSTEM,
                    user_prompt=DESCRIBE_USER,
                ),
                retries, backoff, logger, f"{example_id}/describe",
            )
            (iter_dir / "description.txt").write_text(description, encoding="utf-8")
            logger.info("[%s] description: %d chars", example_id, len(description))

            logger.info("[%s] iter_0: generating code from description", example_id)
            code_prompt = DESCRIBE_THEN_CODE_USER.format(description=description)
            raw_gen = with_retries(
                lambda: models.generator.generate_code(
                    video_path=gt_video_path,
                    system_prompt=GENERATOR_SYSTEM,
                    user_prompt=code_prompt,
                ),
                retries, backoff, logger, f"{example_id}/iter_0/generate",
            )

            (iter_dir / "raw_output.txt").write_text(raw_gen, encoding="utf-8")
            current_html, _ = parse_html(raw_gen)
            (iter_dir / "full.html").write_text(current_html, encoding="utf-8")

            mp4_path = iter_dir / "animation.mp4"
            with_retries(
                lambda: recorder.record(iter_dir / "full.html", mp4_path),
                retries, backoff, logger, f"{example_id}/iter_0/record",
            )
            current_video = mp4_path

            elapsed = round(time.time() - t0, 1)
            trace.add_iter_generate(
                iter_idx=0, model=f"{models.generator_name}(describe_first)",
                html_chars=len(current_html), raw_output_file="iter_0/raw_output.txt",
                duration_s=elapsed,
            )
            logger.info("[%s] iter_0 done in %.1fs", example_id, elapsed)

        for i in range(1, max_iters + 1):
            iter_dir = example_dir / f"iter_{i}"
            iter_dir.mkdir(exist_ok=True)
            t0 = time.time()

            from modules.pipeline import CRITIC_SYSTEM
            diff_prompt = DESCRIBE_DIFF_USER.format(description=description)
            raw_feedback = with_retries(
                lambda: models.critic.critique(
                    target_video=gt_video_path, generated_video=current_video,
                    system_prompt=CRITIC_SYSTEM, user_prompt=diff_prompt,
                ),
                retries, backoff, logger, f"{example_id}/iter_{i}/critique",
            )
            feedback = raw_feedback.strip()
            (iter_dir / "feedback.txt").write_text(feedback, encoding="utf-8")

            user_prompt = EDITOR_USER_TEMPLATE.format(
                target_video_note=f"Original animation description:\n{description}\n\n",
                current_code=current_html, feedback=feedback,
            )
            raw_edit = with_retries(
                lambda: models.editor.edit_code(
                    current_code=current_html, feedback=feedback,
                    system_prompt=EDITOR_SYSTEM, user_prompt=user_prompt,
                    target_video=gt_video_path if include_target_video else None,
                ),
                retries, backoff, logger, f"{example_id}/iter_{i}/edit",
            )
            new_html, _ = parse_html(raw_edit)
            (iter_dir / "full.html").write_text(new_html, encoding="utf-8")
            (iter_dir / "raw_editor.txt").write_text(raw_edit, encoding="utf-8")
            current_html = new_html

            mp4_path = iter_dir / "animation.mp4"
            with_retries(
                lambda: recorder.record(iter_dir / "full.html", mp4_path),
                retries, backoff, logger, f"{example_id}/iter_{i}/record",
            )
            current_video = mp4_path

            elapsed = round(time.time() - t0, 1)
            trace.add_iter_refine(
                iter_idx=i, critic_model=models.critic_name,
                editor_model=models.editor_name,
                feedback_chars=len(feedback), html_chars=len(new_html),
                raw_critic_file=f"iter_{i}/feedback.txt",
                raw_editor_file=f"iter_{i}/raw_editor.txt",
                duration_s=elapsed,
            )
            logger.info("[%s] iter_%d done in %.1fs", example_id, i, elapsed)

    except Exception as exc:
        trace.mark_error(str(exc), stage=f"iter_{len(trace.data['iterations'])}")
        return trace.data

    trace.mark_success()
    return trace.data


VISUAL_CRITIC_SYSTEM = (
    "You are a visual expert. Compare two animation videos and describe ONLY "
    "the visual differences you see. Focus on what looks different, not code."
)

VISUAL_CRITIC_USER = (
    "Compare these two animations:\n"
    "  1. TARGET — what we want.\n"
    "  2. CURRENT — what we have.\n\n"
    "List ONLY visual differences: colors, shapes, sizes, positions, "
    "motion paths, speed, timing. Be specific (e.g., 'circle is blue "
    "instead of red', 'rotation is clockwise instead of counter-clockwise')."
)

CODE_CRITIC_SYSTEM = (
    "You are a code review expert for CSS animations and JavaScript. "
    "Analyze code structure and suggest improvements."
)

CODE_CRITIC_USER = (
    "Here is HTML/CSS/JS code for an animation that doesn't quite match the target.\n\n"
    "Visual differences identified:\n{visual_feedback}\n\n"
    "Current code:\n```html\n{current_code}\n```\n\n"
    "For each visual difference, identify the specific CSS property, keyframe, "
    "or JS line that causes it, and suggest the exact fix. Be precise about "
    "values (colors, durations, transforms)."
)

DUAL_EDITOR_USER = (
    "Here is the current HTML code:\n\n"
    "```html\n{current_code}\n```\n\n"
    "Visual feedback:\n{visual_feedback}\n\n"
    "Code-level fixes:\n{code_feedback}\n\n"
    "Generate an improved version that addresses all feedback. "
    "Output only a single, complete HTML document."
)


def run_dual_critic(
    example_id: str, gt_video_path: Path, cfg: dict,
    models: Models, run_dir: Path, logger: RunLogger,
) -> dict:
    """Run two critics before each edit (METAL).

    The visual critic compares the videos, then the code critic turns that into code fixes. The editor
    receives both.

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

    try:
        current_html, current_video = _run_iter_0(
            example_id, gt_video_path, models, recorder, example_dir,
            retries, backoff, logger, trace,
        )

        for i in range(1, max_iters + 1):
            iter_dir = example_dir / f"iter_{i}"
            iter_dir.mkdir(exist_ok=True)
            t0 = time.time()

            logger.info("[%s] iter_%d: visual critique", example_id, i)
            raw_visual = with_retries(
                lambda: models.critic.critique(
                    target_video=gt_video_path, generated_video=current_video,
                    system_prompt=VISUAL_CRITIC_SYSTEM,
                    user_prompt=VISUAL_CRITIC_USER,
                ),
                retries, backoff, logger, f"{example_id}/iter_{i}/visual_critique",
            )
            visual_feedback = raw_visual.strip()
            (iter_dir / "visual_feedback.txt").write_text(visual_feedback, encoding="utf-8")

            logger.info("[%s] iter_%d: code critique", example_id, i)
            code_critic_prompt = CODE_CRITIC_USER.format(
                visual_feedback=visual_feedback, current_code=current_html,
            )
            raw_code = with_retries(
                lambda: models.editor.edit_code(
                    current_code=current_html, feedback=visual_feedback,
                    system_prompt=CODE_CRITIC_SYSTEM,
                    user_prompt=code_critic_prompt,
                ),
                retries, backoff, logger, f"{example_id}/iter_{i}/code_critique",
            )
            code_feedback = raw_code.strip()
            (iter_dir / "code_feedback.txt").write_text(code_feedback, encoding="utf-8")
            (iter_dir / "feedback.txt").write_text(
                f"=== VISUAL ===\n{visual_feedback}\n\n=== CODE ===\n{code_feedback}",
                encoding="utf-8",
            )

            logger.info("[%s] iter_%d: editing with dual feedback", example_id, i)
            editor_prompt = DUAL_EDITOR_USER.format(
                current_code=current_html,
                visual_feedback=visual_feedback,
                code_feedback=code_feedback,
            )
            raw_edit = with_retries(
                lambda: models.editor.edit_code(
                    current_code=current_html,
                    feedback=f"{visual_feedback}\n\n{code_feedback}",
                    system_prompt=EDITOR_SYSTEM,
                    user_prompt=editor_prompt,
                ),
                retries, backoff, logger, f"{example_id}/iter_{i}/edit",
            )
            new_html, _ = parse_html(raw_edit)
            (iter_dir / "full.html").write_text(new_html, encoding="utf-8")
            (iter_dir / "raw_editor.txt").write_text(raw_edit, encoding="utf-8")
            current_html = new_html

            mp4_path = iter_dir / "animation.mp4"
            with_retries(
                lambda: recorder.record(iter_dir / "full.html", mp4_path),
                retries, backoff, logger, f"{example_id}/iter_{i}/record",
            )
            current_video = mp4_path

            elapsed = round(time.time() - t0, 1)
            trace.add_iter_refine(
                iter_idx=i, critic_model=f"{models.critic_name}(dual)",
                editor_model=models.editor_name,
                feedback_chars=len(visual_feedback) + len(code_feedback),
                html_chars=len(new_html),
                raw_critic_file=f"iter_{i}/feedback.txt",
                raw_editor_file=f"iter_{i}/raw_editor.txt",
                duration_s=elapsed,
            )
            logger.info("[%s] iter_%d done in %.1fs", example_id, i, elapsed)

    except Exception as exc:
        trace.mark_error(str(exc), stage=f"iter_{len(trace.data['iterations'])}")
        return trace.data

    trace.mark_success()
    return trace.data


CAPTION_SYSTEM = (
    "You are an expert at describing web animations in precise detail. "
    "Describe exactly what you see happening in the animation."
)

CAPTION_USER = (
    "Watch this animation video and describe what happens frame by frame:\n"
    "- What objects are present and what they look like\n"
    "- How they move, rotate, scale, or change\n"
    "- The timing and sequence of events\n"
    "- Any color changes, fading, or effects\n\n"
    "Be precise about directions (left/right/up/down), speeds (fast/slow), "
    "and timing (first 2 seconds, then...)."
)

CAPTION_COMPARE_USER = (
    "Here are descriptions of two animations:\n\n"
    "TARGET animation (what we want):\n{target_caption}\n\n"
    "CURRENT animation (what we generated):\n{generated_caption}\n\n"
    "Compare these descriptions and list the specific mismatches:\n"
    "1. What is different about the objects?\n"
    "2. What is different about the motion/movement?\n"
    "3. What is different about the timing?\n"
    "4. What is different about the visual effects?\n\n"
    "For each mismatch, suggest a specific code change to fix it."
)


def run_caption_bridge(
    example_id: str, gt_video_path: Path, cfg: dict,
    models: Models, run_dir: Path, logger: RunLogger,
) -> dict:
    """Caption each video and compare the captions as text (PhyT2V).

    This avoids asking the critic to compare two videos directly.

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
        # Caption the target video once (reused every iteration)
        logger.info("[%s] captioning target video", example_id)
        target_caption = with_retries(
            lambda: models.generator.generate_code(
                video_path=gt_video_path,
                system_prompt=CAPTION_SYSTEM,
                user_prompt=CAPTION_USER,
            ),
            retries, backoff, logger, f"{example_id}/caption_target",
        )
        (example_dir / "target_caption.txt").write_text(target_caption, encoding="utf-8")

        current_html, current_video = _run_iter_0(
            example_id, gt_video_path, models, recorder, example_dir,
            retries, backoff, logger, trace,
        )

        for i in range(1, max_iters + 1):
            iter_dir = example_dir / f"iter_{i}"
            iter_dir.mkdir(exist_ok=True)
            t0 = time.time()

            logger.info("[%s] iter_%d: captioning generated video", example_id, i)
            gen_caption = with_retries(
                lambda: models.critic.critique(
                    target_video=current_video,
                    generated_video=current_video,
                    system_prompt=CAPTION_SYSTEM,
                    user_prompt=CAPTION_USER,
                ),
                retries, backoff, logger, f"{example_id}/iter_{i}/caption_gen",
            )
            (iter_dir / "generated_caption.txt").write_text(gen_caption, encoding="utf-8")

            logger.info("[%s] iter_%d: comparing captions", example_id, i)
            from modules.pipeline import CRITIC_SYSTEM
            compare_prompt = CAPTION_COMPARE_USER.format(
                target_caption=target_caption, generated_caption=gen_caption,
            )
            raw_feedback = with_retries(
                lambda: models.editor.edit_code(
                    current_code="", feedback="",
                    system_prompt=CRITIC_SYSTEM,
                    user_prompt=compare_prompt,
                ),
                retries, backoff, logger, f"{example_id}/iter_{i}/compare",
            )
            feedback = raw_feedback.strip()
            (iter_dir / "feedback.txt").write_text(feedback, encoding="utf-8")
            (iter_dir / "raw_critic.txt").write_text(raw_feedback, encoding="utf-8")

            logger.info("[%s] iter_%d: editing code", example_id, i)
            user_prompt = EDITOR_USER_TEMPLATE.format(
                target_video_note=(f"Target animation description:\n{target_caption}\n\n"
                                   if not include_target_video else
                                   "The first video above is the TARGET.\n\n"),
                current_code=current_html, feedback=feedback,
            )
            raw_edit = with_retries(
                lambda: models.editor.edit_code(
                    current_code=current_html, feedback=feedback,
                    system_prompt=EDITOR_SYSTEM, user_prompt=user_prompt,
                    target_video=gt_video_path if include_target_video else None,
                ),
                retries, backoff, logger, f"{example_id}/iter_{i}/edit",
            )
            new_html, _ = parse_html(raw_edit)
            (iter_dir / "full.html").write_text(new_html, encoding="utf-8")
            (iter_dir / "raw_editor.txt").write_text(raw_edit, encoding="utf-8")
            current_html = new_html

            mp4_path = iter_dir / "animation.mp4"
            with_retries(
                lambda: recorder.record(iter_dir / "full.html", mp4_path),
                retries, backoff, logger, f"{example_id}/iter_{i}/record",
            )
            current_video = mp4_path

            elapsed = round(time.time() - t0, 1)
            trace.add_iter_refine(
                iter_idx=i, critic_model=f"{models.critic_name}(caption_bridge)",
                editor_model=models.editor_name,
                feedback_chars=len(feedback), html_chars=len(new_html),
                raw_critic_file=f"iter_{i}/raw_critic.txt",
                raw_editor_file=f"iter_{i}/raw_editor.txt",
                duration_s=elapsed,
            )
            logger.info("[%s] iter_%d done in %.1fs", example_id, i, elapsed)

    except Exception as exc:
        trace.mark_error(str(exc), stage=f"iter_{len(trace.data['iterations'])}")
        return trace.data

    trace.mark_success()
    return trace.data
