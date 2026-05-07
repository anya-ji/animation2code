"""
Experimental pipeline variants for iterative refinement.

Our experiments:
  1. metric_guided  — Feed DINO scores to critic so it knows what's weak
  2. best_of_n      — Generate N variants per iteration, keep best by DINO score
  3. reward_gated   — Only accept edits if DINO combined improves, else revert
  4. rl_explore     — RL-lite: metric-guided + best-of-N + reward gate

Literature-inspired:
  5. describe_first — (ChartIR) Describe animation in NL before generating code
  6. dual_critic    — (METAL) Separate visual critic + code critic
  7. caption_bridge — (PhyT2V) Caption both videos, compare as text

All variants reuse the same Models, Recorder, and trace infrastructure.
"""

import time
from pathlib import Path

from modules.logger import ExampleTrace, RunLogger
from modules.models.base import BaseModel
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

# ── DINO scoring helper ──────────────────────────────────────────────────

def compute_dino_score(gt_video: Path, gen_video: Path, device: str = "cpu") -> dict:
    """Compute DINO appearance + motion + combined between two videos.

    Returns dict with keys: appearance, motion, combined.
    Loads the extractor lazily on first call.
    """
    if not hasattr(compute_dino_score, "_extractor"):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".." / "evaluation"))
        from dino_eval import DINOExtractor, evaluate_example
        compute_dino_score._extractor = DINOExtractor(device=device)
        compute_dino_score._evaluate = evaluate_example

    scores = compute_dino_score._evaluate(
        compute_dino_score._extractor,
        str(gt_video),
        str(gen_video),
        top_k=3,
        stride=2,  # every other frame for speed
    )
    app = scores["appearance"]["mean"] or 0.0
    mot = scores["motion"]
    return {
        "appearance": round(app, 4),
        "motion": round(mot, 4),
        "combined": round(0.6 * app + 0.4 * mot, 4),
    }


# ── Shared: initial generation (iter_0) ──────────────────────────────────

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


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 1: Metric-Guided Critic
# ═══════════════════════════════════════════════════════════════════════════

METRIC_GUIDED_CRITIC_USER = (
    "You are given two animation videos:\n"
    "  1. TARGET — the animation we want to recreate.\n"
    "  2. CURRENT — the latest generated attempt.\n\n"
    "Automated evaluation scores (0-1 scale, higher = better):\n"
    "  - Appearance similarity: {appearance:.3f}\n"
    "  - Motion accuracy: {motion:.3f}\n"
    "  - Combined score: {combined:.3f}\n\n"
    "{focus_guidance}\n\n"
    "Analyze the videos and provide:\n"
    "  (a) A concise numbered list of specific visual differences.\n"
    "  (b) For each difference, a concrete suggestion for what to change in the code.\n\n"
    "Be specific and actionable. Do NOT include any code."
)


def run_metric_guided(
    example_id: str, gt_video_path: Path, cfg: dict,
    models: Models, run_dir: Path, logger: RunLogger,
    dino_device: str = "cpu",
) -> dict:
    """Experiment 1: Feed DINO scores to critic to guide its feedback."""
    example_dir = run_dir / example_id
    example_dir.mkdir(parents=True, exist_ok=True)
    trace = ExampleTrace(example_id, example_dir)
    recorder = Recorder(cfg["recorder"])

    max_iters = int(cfg.get("max_iterations", 3))
    retries = int(cfg.get("retries", 2))
    backoff = float(cfg.get("retry_backoff_s", 5))
    include_target_video = cfg.get("editor", {}).get("include_target_video", False)

    try:
        current_html, current_video = _run_iter_0(
            example_id, gt_video_path, models, recorder, example_dir,
            retries, backoff, logger, trace,
        )

        for i in range(1, max_iters + 1):
            iter_dir = example_dir / f"iter_{i}"
            iter_dir.mkdir(exist_ok=True)
            t0 = time.time()

            # Score current iteration with DINO
            logger.info("[%s] iter_%d: computing DINO scores", example_id, i)
            dino = compute_dino_score(gt_video_path, current_video, device=dino_device)
            logger.info("[%s] iter_%d: DINO app=%.3f mot=%.3f comb=%.3f",
                        example_id, i, dino["appearance"], dino["motion"], dino["combined"])

            # Build focus guidance based on weaker dimension
            if dino["appearance"] < dino["motion"]:
                focus = ("Appearance is the weaker dimension. Focus your feedback on "
                         "visual differences: colors, shapes, sizes, layout, gradients.")
            else:
                focus = ("Motion is the weaker dimension. Focus your feedback on "
                         "timing, easing, speed, direction, and animation trajectories.")

            critic_prompt = METRIC_GUIDED_CRITIC_USER.format(
                appearance=dino["appearance"], motion=dino["motion"],
                combined=dino["combined"], focus_guidance=focus,
            )

            logger.info("[%s] iter_%d: critiquing (metric-guided)", example_id, i)
            from modules.pipeline import CRITIC_SYSTEM
            raw_feedback = with_retries(
                lambda: models.critic.critique(
                    target_video=gt_video_path, generated_video=current_video,
                    system_prompt=CRITIC_SYSTEM, user_prompt=critic_prompt,
                ),
                retries, backoff, logger, f"{example_id}/iter_{i}/critique",
            )
            feedback = raw_feedback.strip()
            (iter_dir / "raw_critic.txt").write_text(raw_feedback, encoding="utf-8")
            (iter_dir / "feedback.txt").write_text(feedback, encoding="utf-8")
            (iter_dir / "dino_scores.txt").write_text(str(dino), encoding="utf-8")

            # Edit
            logger.info("[%s] iter_%d: editing code", example_id, i)
            user_prompt = EDITOR_USER_TEMPLATE.format(
                target_video_note=("The first video above is the TARGET.\n\n"
                                   if include_target_video else ""),
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
            (iter_dir / "raw_editor.txt").write_text(raw_edit, encoding="utf-8")
            new_html, _ = parse_html(raw_edit)
            (iter_dir / "full.html").write_text(new_html, encoding="utf-8")
            current_html = new_html

            # Record
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
                raw_critic_file=f"iter_{i}/raw_critic.txt",
                raw_editor_file=f"iter_{i}/raw_editor.txt",
                duration_s=elapsed,
            )
            logger.info("[%s] iter_%d done in %.1fs (DINO comb=%.3f)",
                        example_id, i, elapsed, dino["combined"])

    except Exception as exc:
        trace.mark_error(str(exc), stage=f"iter_{len(trace.data['iterations'])}")
        return trace.data

    trace.mark_success()
    return trace.data


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 2: Best-of-N (RL-lite)
# ═══════════════════════════════════════════════════════════════════════════

def run_best_of_n(
    example_id: str, gt_video_path: Path, cfg: dict,
    models: Models, run_dir: Path, logger: RunLogger,
    n_samples: int = 3, dino_device: str = "cpu",
) -> dict:
    """Experiment 2: Generate N code variants per iteration, keep the best by DINO."""
    example_dir = run_dir / example_id
    example_dir.mkdir(parents=True, exist_ok=True)
    trace = ExampleTrace(example_id, example_dir)
    recorder = Recorder(cfg["recorder"])

    max_iters = int(cfg.get("max_iterations", 3))
    retries = int(cfg.get("retries", 2))
    backoff = float(cfg.get("retry_backoff_s", 5))

    try:
        # iter_0: generate N variants, pick best
        logger.info("[%s] iter_0: generating %d variants (best-of-N)", example_id, n_samples)
        t0 = time.time()
        iter_dir = example_dir / "iter_0"
        iter_dir.mkdir(exist_ok=True)

        best_score = -1
        best_html = ""
        best_video = None

        for k in range(n_samples):
            sample_dir = iter_dir / f"sample_{k}"
            sample_dir.mkdir(exist_ok=True)

            raw_gen = with_retries(
                lambda: models.generator.generate_code(
                    video_path=gt_video_path,
                    system_prompt=GENERATOR_SYSTEM,
                    user_prompt=GENERATOR_USER,
                ),
                retries, backoff, logger, f"{example_id}/iter_0/sample_{k}",
            )
            html, _ = parse_html(raw_gen)
            (sample_dir / "full.html").write_text(html, encoding="utf-8")
            (sample_dir / "raw_output.txt").write_text(raw_gen, encoding="utf-8")

            mp4 = sample_dir / "animation.mp4"
            with_retries(
                lambda: recorder.record(sample_dir / "full.html", mp4),
                retries, backoff, logger, f"{example_id}/iter_0/record_{k}",
            )

            dino = compute_dino_score(gt_video_path, mp4, device=dino_device)
            (sample_dir / "dino_scores.txt").write_text(str(dino), encoding="utf-8")
            logger.info("[%s] iter_0 sample_%d: DINO comb=%.3f", example_id, k, dino["combined"])

            if dino["combined"] > best_score:
                best_score = dino["combined"]
                best_html = html
                best_video = mp4

        # Copy best to iter_0 root
        (iter_dir / "full.html").write_text(best_html, encoding="utf-8")
        import shutil
        shutil.copy2(best_video, iter_dir / "animation.mp4")
        current_html = best_html
        current_video = iter_dir / "animation.mp4"

        elapsed = round(time.time() - t0, 1)
        trace.add_iter_generate(
            iter_idx=0, model=f"{models.generator_name}(best_of_{n_samples})",
            html_chars=len(current_html), raw_output_file="iter_0/raw_output.txt",
            duration_s=elapsed,
        )
        logger.info("[%s] iter_0 best-of-%d done in %.1fs (best DINO=%.3f)",
                    example_id, n_samples, elapsed, best_score)

        # Refinement iterations: same best-of-N but with critic/editor
        for i in range(1, max_iters + 1):
            iter_dir = example_dir / f"iter_{i}"
            iter_dir.mkdir(exist_ok=True)
            t0 = time.time()

            # Critique once
            from modules.pipeline import CRITIC_SYSTEM, CRITIC_USER
            raw_feedback = with_retries(
                lambda: models.critic.critique(
                    target_video=gt_video_path, generated_video=current_video,
                    system_prompt=CRITIC_SYSTEM, user_prompt=CRITIC_USER,
                ),
                retries, backoff, logger, f"{example_id}/iter_{i}/critique",
            )
            feedback = raw_feedback.strip()
            (iter_dir / "feedback.txt").write_text(feedback, encoding="utf-8")

            # Generate N edits, pick best
            best_score = -1
            best_html_new = ""
            best_video_new = None

            for k in range(n_samples):
                sample_dir = iter_dir / f"sample_{k}"
                sample_dir.mkdir(exist_ok=True)

                user_prompt = EDITOR_USER_TEMPLATE.format(
                    target_video_note="", current_code=current_html, feedback=feedback,
                )
                raw_edit = with_retries(
                    lambda: models.editor.edit_code(
                        current_code=current_html, feedback=feedback,
                        system_prompt=EDITOR_SYSTEM, user_prompt=user_prompt,
                    ),
                    retries, backoff, logger, f"{example_id}/iter_{i}/edit_{k}",
                )
                html, _ = parse_html(raw_edit)
                (sample_dir / "full.html").write_text(html, encoding="utf-8")

                mp4 = sample_dir / "animation.mp4"
                with_retries(
                    lambda: recorder.record(sample_dir / "full.html", mp4),
                    retries, backoff, logger, f"{example_id}/iter_{i}/record_{k}",
                )

                dino = compute_dino_score(gt_video_path, mp4, device=dino_device)
                (sample_dir / "dino_scores.txt").write_text(str(dino), encoding="utf-8")
                logger.info("[%s] iter_%d sample_%d: DINO comb=%.3f",
                            example_id, i, k, dino["combined"])

                if dino["combined"] > best_score:
                    best_score = dino["combined"]
                    best_html_new = html
                    best_video_new = mp4

            (iter_dir / "full.html").write_text(best_html_new, encoding="utf-8")
            shutil.copy2(best_video_new, iter_dir / "animation.mp4")
            current_html = best_html_new
            current_video = iter_dir / "animation.mp4"

            elapsed = round(time.time() - t0, 1)
            trace.add_iter_refine(
                iter_idx=i, critic_model=models.critic_name,
                editor_model=f"{models.editor_name}(best_of_{n_samples})",
                feedback_chars=len(feedback), html_chars=len(best_html_new),
                raw_critic_file=f"iter_{i}/feedback.txt",
                raw_editor_file=f"iter_{i}/sample_0/full.html",
                duration_s=elapsed,
            )
            logger.info("[%s] iter_%d best-of-%d done (DINO=%.3f)",
                        example_id, i, n_samples, best_score)

    except Exception as exc:
        trace.mark_error(str(exc), stage=f"iter_{len(trace.data['iterations'])}")
        return trace.data

    trace.mark_success()
    return trace.data


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 3: Reward-Gated Refinement
# ═══════════════════════════════════════════════════════════════════════════

def run_reward_gated(
    example_id: str, gt_video_path: Path, cfg: dict,
    models: Models, run_dir: Path, logger: RunLogger,
    dino_device: str = "cpu",
) -> dict:
    """Experiment 3: Only accept edits if DINO combined improves. Revert otherwise."""
    example_dir = run_dir / example_id
    example_dir.mkdir(parents=True, exist_ok=True)
    trace = ExampleTrace(example_id, example_dir)
    recorder = Recorder(cfg["recorder"])

    max_iters = int(cfg.get("max_iterations", 3))
    retries = int(cfg.get("retries", 2))
    backoff = float(cfg.get("retry_backoff_s", 5))
    include_target_video = cfg.get("editor", {}).get("include_target_video", False)

    try:
        current_html, current_video = _run_iter_0(
            example_id, gt_video_path, models, recorder, example_dir,
            retries, backoff, logger, trace,
        )

        # Score iter_0
        best_dino = compute_dino_score(gt_video_path, current_video, device=dino_device)
        best_score = best_dino["combined"]
        logger.info("[%s] iter_0 DINO baseline: %.3f", example_id, best_score)

        reverts = 0
        for i in range(1, max_iters + 1):
            iter_dir = example_dir / f"iter_{i}"
            iter_dir.mkdir(exist_ok=True)
            t0 = time.time()

            # Critique
            from modules.pipeline import CRITIC_SYSTEM, CRITIC_USER
            raw_feedback = with_retries(
                lambda: models.critic.critique(
                    target_video=gt_video_path, generated_video=current_video,
                    system_prompt=CRITIC_SYSTEM, user_prompt=CRITIC_USER,
                ),
                retries, backoff, logger, f"{example_id}/iter_{i}/critique",
            )
            feedback = raw_feedback.strip()
            (iter_dir / "feedback.txt").write_text(feedback, encoding="utf-8")

            # Edit
            user_prompt = EDITOR_USER_TEMPLATE.format(
                target_video_note=("The first video above is the TARGET.\n\n"
                                   if include_target_video else ""),
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

            # Record
            mp4_path = iter_dir / "animation.mp4"
            with_retries(
                lambda: recorder.record(iter_dir / "full.html", mp4_path),
                retries, backoff, logger, f"{example_id}/iter_{i}/record",
            )

            # Score and decide: accept or revert
            new_dino = compute_dino_score(gt_video_path, mp4_path, device=dino_device)
            new_score = new_dino["combined"]

            if new_score >= best_score:
                logger.info("[%s] iter_%d: ACCEPT (%.3f -> %.3f)",
                            example_id, i, best_score, new_score)
                current_html = new_html
                current_video = mp4_path
                best_score = new_score
                (iter_dir / "_accepted.txt").write_text(
                    f"accepted: {best_score:.4f} -> {new_score:.4f}", encoding="utf-8")
            else:
                logger.info("[%s] iter_%d: REVERT (%.3f -> %.3f, worse)",
                            example_id, i, best_score, new_score)
                reverts += 1
                (iter_dir / "_reverted.txt").write_text(
                    f"reverted: {best_score:.4f} -> {new_score:.4f}", encoding="utf-8")

            (iter_dir / "dino_scores.txt").write_text(str(new_dino), encoding="utf-8")

            elapsed = round(time.time() - t0, 1)
            trace.add_iter_refine(
                iter_idx=i, critic_model=models.critic_name,
                editor_model=models.editor_name,
                feedback_chars=len(feedback), html_chars=len(new_html),
                raw_critic_file=f"iter_{i}/feedback.txt",
                raw_editor_file=f"iter_{i}/raw_editor.txt",
                duration_s=elapsed,
            )

        logger.info("[%s] reward-gated complete: final=%.3f, reverts=%d/%d",
                    example_id, best_score, reverts, max_iters)

    except Exception as exc:
        trace.mark_error(str(exc), stage=f"iter_{len(trace.data['iterations'])}")
        return trace.data

    trace.mark_success()
    return trace.data


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 4: RL Exploration (Best-of-N + Reward Gate + Metric-Guided)
# ═══════════════════════════════════════════════════════════════════════════

def run_rl_explore(
    example_id: str, gt_video_path: Path, cfg: dict,
    models: Models, run_dir: Path, logger: RunLogger,
    n_samples: int = 3, dino_device: str = "cpu",
) -> dict:
    """Experiment 4: Full RL-lite pipeline.

    Combines: metric-guided critic + best-of-N sampling + reward gating.
    The DINO score is the reward signal throughout. No text feedback is
    strictly necessary, but we still use it to guide the editor. The key
    difference from baseline is that the reward (DINO) drives all decisions:
    what to focus on, which sample to keep, whether to accept.
    """
    example_dir = run_dir / example_id
    example_dir.mkdir(parents=True, exist_ok=True)
    trace = ExampleTrace(example_id, example_dir)
    recorder = Recorder(cfg["recorder"])

    max_iters = int(cfg.get("max_iterations", 3))
    retries = int(cfg.get("retries", 2))
    backoff = float(cfg.get("retry_backoff_s", 5))

    try:
        # iter_0: best-of-N initial generation
        logger.info("[%s] iter_0: RL explore — %d samples", example_id, n_samples)
        t0 = time.time()
        iter_dir = example_dir / "iter_0"
        iter_dir.mkdir(exist_ok=True)

        best_score = -1
        best_html = ""
        best_video = None
        import shutil

        for k in range(n_samples):
            sample_dir = iter_dir / f"sample_{k}"
            sample_dir.mkdir(exist_ok=True)
            raw_gen = with_retries(
                lambda: models.generator.generate_code(
                    video_path=gt_video_path,
                    system_prompt=GENERATOR_SYSTEM,
                    user_prompt=GENERATOR_USER,
                ),
                retries, backoff, logger, f"{example_id}/iter_0/sample_{k}",
            )
            html, _ = parse_html(raw_gen)
            (sample_dir / "full.html").write_text(html, encoding="utf-8")
            mp4 = sample_dir / "animation.mp4"
            with_retries(
                lambda: recorder.record(sample_dir / "full.html", mp4),
                retries, backoff, logger, f"{example_id}/iter_0/record_{k}",
            )
            dino = compute_dino_score(gt_video_path, mp4, device=dino_device)
            (sample_dir / "dino_scores.txt").write_text(str(dino), encoding="utf-8")
            logger.info("[%s] iter_0 sample_%d: %.3f", example_id, k, dino["combined"])
            if dino["combined"] > best_score:
                best_score = dino["combined"]
                best_html = html
                best_video = mp4

        (iter_dir / "full.html").write_text(best_html, encoding="utf-8")
        shutil.copy2(best_video, iter_dir / "animation.mp4")
        current_html = best_html
        current_video = iter_dir / "animation.mp4"

        elapsed = round(time.time() - t0, 1)
        trace.add_iter_generate(
            iter_idx=0, model=f"{models.generator_name}(rl_explore_{n_samples})",
            html_chars=len(current_html), raw_output_file="iter_0/full.html",
            duration_s=elapsed,
        )
        logger.info("[%s] iter_0 done: best=%.3f (%.1fs)", example_id, best_score, elapsed)

        # Refinement: metric-guided critic + best-of-N edits + reward gate
        for i in range(1, max_iters + 1):
            iter_dir = example_dir / f"iter_{i}"
            iter_dir.mkdir(exist_ok=True)
            t0 = time.time()

            # Score current state
            current_dino = compute_dino_score(gt_video_path, current_video, device=dino_device)

            # Metric-guided critique
            if current_dino["appearance"] < current_dino["motion"]:
                focus = "Appearance is weaker. Focus on visual differences: colors, shapes, layout."
            else:
                focus = "Motion is weaker. Focus on timing, easing, speed, trajectories."

            critic_prompt = METRIC_GUIDED_CRITIC_USER.format(
                appearance=current_dino["appearance"],
                motion=current_dino["motion"],
                combined=current_dino["combined"],
                focus_guidance=focus,
            )
            from modules.pipeline import CRITIC_SYSTEM
            raw_feedback = with_retries(
                lambda: models.critic.critique(
                    target_video=gt_video_path, generated_video=current_video,
                    system_prompt=CRITIC_SYSTEM, user_prompt=critic_prompt,
                ),
                retries, backoff, logger, f"{example_id}/iter_{i}/critique",
            )
            feedback = raw_feedback.strip()
            (iter_dir / "feedback.txt").write_text(feedback, encoding="utf-8")

            # Best-of-N edits
            iter_best_score = -1
            iter_best_html = ""
            iter_best_video = None

            for k in range(n_samples):
                sample_dir = iter_dir / f"sample_{k}"
                sample_dir.mkdir(exist_ok=True)
                user_prompt = EDITOR_USER_TEMPLATE.format(
                    target_video_note="", current_code=current_html, feedback=feedback,
                )
                raw_edit = with_retries(
                    lambda: models.editor.edit_code(
                        current_code=current_html, feedback=feedback,
                        system_prompt=EDITOR_SYSTEM, user_prompt=user_prompt,
                    ),
                    retries, backoff, logger, f"{example_id}/iter_{i}/edit_{k}",
                )
                html, _ = parse_html(raw_edit)
                (sample_dir / "full.html").write_text(html, encoding="utf-8")
                mp4 = sample_dir / "animation.mp4"
                with_retries(
                    lambda: recorder.record(sample_dir / "full.html", mp4),
                    retries, backoff, logger, f"{example_id}/iter_{i}/record_{k}",
                )
                dino = compute_dino_score(gt_video_path, mp4, device=dino_device)
                (sample_dir / "dino_scores.txt").write_text(str(dino), encoding="utf-8")
                logger.info("[%s] iter_%d sample_%d: %.3f", example_id, i, k, dino["combined"])
                if dino["combined"] > iter_best_score:
                    iter_best_score = dino["combined"]
                    iter_best_html = html
                    iter_best_video = mp4

            # Reward gate: only accept if improved
            if iter_best_score >= best_score:
                logger.info("[%s] iter_%d: ACCEPT best sample (%.3f -> %.3f)",
                            example_id, i, best_score, iter_best_score)
                (iter_dir / "full.html").write_text(iter_best_html, encoding="utf-8")
                shutil.copy2(iter_best_video, iter_dir / "animation.mp4")
                current_html = iter_best_html
                current_video = iter_dir / "animation.mp4"
                best_score = iter_best_score
            else:
                logger.info("[%s] iter_%d: REVERT (best sample %.3f < current %.3f)",
                            example_id, i, iter_best_score, best_score)
                (iter_dir / "_reverted.txt").write_text(
                    f"reverted: best_sample={iter_best_score:.4f} < current={best_score:.4f}",
                    encoding="utf-8")

            elapsed = round(time.time() - t0, 1)
            trace.add_iter_refine(
                iter_idx=i, critic_model=models.critic_name,
                editor_model=f"{models.editor_name}(rl_explore_{n_samples})",
                feedback_chars=len(feedback), html_chars=len(iter_best_html),
                raw_critic_file=f"iter_{i}/feedback.txt",
                raw_editor_file=f"iter_{i}/sample_0/full.html",
                duration_s=elapsed,
            )

        logger.info("[%s] rl_explore complete: final=%.3f", example_id, best_score)

    except Exception as exc:
        trace.mark_error(str(exc), stage=f"iter_{len(trace.data['iterations'])}")
        return trace.data

    trace.mark_success()
    return trace.data


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 5: Describe-First (inspired by ChartIR)
# ═══════════════════════════════════════════════════════════════════════════

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
    dino_device: str = "cpu",
) -> dict:
    """Experiment 5: Describe animation in NL first, then generate code from description.

    Inspired by ChartIR's 'description instruction' approach.
    Decouples visual understanding from code translation.
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
        # Step 1: Describe the animation
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

            # Step 2: Generate code from description (+ video)
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

        # Refinement: use holistic difference descriptions (ChartIR style)
        for i in range(1, max_iters + 1):
            iter_dir = example_dir / f"iter_{i}"
            iter_dir.mkdir(exist_ok=True)
            t0 = time.time()

            # Holistic difference critique (referencing the original description)
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

            # Edit with description context
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


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 6: Dual Critic (inspired by METAL)
# ═══════════════════════════════════════════════════════════════════════════

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
    dino_device: str = "cpu",
) -> dict:
    """Experiment 6: Separate visual critic + code critic (METAL approach).

    Two-pass critique:
    1. Visual critic: compares videos, describes visual differences
    2. Code critic: reads the visual feedback + code, suggests specific code fixes

    The editor gets both feedbacks.
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

            # Pass 1: Visual critic (looks at videos)
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

            # Pass 2: Code critic (reads visual feedback + code, no video needed)
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

            # Edit with both feedbacks
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


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 7: Caption Bridge (inspired by PhyT2V)
# ═══════════════════════════════════════════════════════════════════════════

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
    dino_device: str = "cpu",
) -> dict:
    """Experiment 7: Caption both videos, compare as text (PhyT2V approach).

    Instead of asking the critic to directly compare two videos visually,
    caption each video separately into text, then compare the text descriptions.
    This bypasses the 'too much visual info' bottleneck.
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

        # iter_0: normal generation
        current_html, current_video = _run_iter_0(
            example_id, gt_video_path, models, recorder, example_dir,
            retries, backoff, logger, trace,
        )

        for i in range(1, max_iters + 1):
            iter_dir = example_dir / f"iter_{i}"
            iter_dir.mkdir(exist_ok=True)
            t0 = time.time()

            # Caption the generated video
            logger.info("[%s] iter_%d: captioning generated video", example_id, i)
            gen_caption = with_retries(
                lambda: models.critic.critique(
                    target_video=current_video,  # just pass one video
                    generated_video=current_video,
                    system_prompt=CAPTION_SYSTEM,
                    user_prompt=CAPTION_USER,
                ),
                retries, backoff, logger, f"{example_id}/iter_{i}/caption_gen",
            )
            (iter_dir / "generated_caption.txt").write_text(gen_caption, encoding="utf-8")

            # Compare captions as text (no video needed for this step)
            logger.info("[%s] iter_%d: comparing captions", example_id, i)
            from modules.pipeline import CRITIC_SYSTEM
            compare_prompt = CAPTION_COMPARE_USER.format(
                target_caption=target_caption, generated_caption=gen_caption,
            )
            # Use editor model for text-only comparison (no video input needed)
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

            # Edit
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
