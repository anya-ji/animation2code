"""Generate HTML for the test set with a fine-tuned checkpoint.

Writes <output_dir>/<id>/ with full.html and _raw_output.txt, or _error.txt on failure.
Finished examples are skipped, so a rerun resumes.
"""
import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import SYSTEM_PROMPT, USER_PROMPT

BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def parse_html(response: str) -> tuple[str | None, str]:
    """Extract a complete HTML document from a model response.

    Args:
        response: Raw model output.

    Returns:
        (html, "") on success, or (None, error) if it is truncated or missing.
    """
    m = re.search(r"```(?:\w+)?\s*(.*?)```", response, re.DOTALL)
    cleaned = m.group(1).strip() if m else response
    if not m:
        m = re.search(r"```(?:\w+)?\s*(.*)", response, re.DOTALL)
        if m:
            cleaned = m.group(1).strip()
    if "<html" not in cleaned.lower():
        return None, "Response did not contain a full HTML document (<html> tag missing)"
    if "</html>" not in cleaned.lower():
        return None, "Output truncated (missing </html>)"
    return cleaned.strip(), ""


def best_checkpoint(run_dir: Path) -> Path:
    """Find the checkpoint-N with the lowest eval_loss.

    Args:
        run_dir: Training output directory.

    Returns:
        Path to the best checkpoint.
    """
    checkpoints = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if not checkpoints:
        raise SystemExit(f"no checkpoint-* under {run_dir}")
    history = json.loads((checkpoints[-1] / "trainer_state.json").read_text())["log_history"]
    evals = {e["step"]: e["eval_loss"] for e in history if "eval_loss" in e}
    step = min(evals, key=evals.get)
    print(f"best checkpoint: checkpoint-{step} (eval_loss={evals[step]:.4f})")
    return run_dir / f"checkpoint-{step}"


def load_model(variant: str, checkpoint: Path):
    """Load a fine-tuned model and the base model's processor.

    Args:
        variant: "lora" or "full".
        checkpoint: A specific checkpoint, or a training output directory. For a directory, LoRA uses
            the lowest eval_loss checkpoint and full uses final/.

    Returns:
        (model, processor)
    """
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if variant == "lora":
        from peft import PeftModel
        if not (checkpoint / "adapter_config.json").exists():
            checkpoint = best_checkpoint(checkpoint)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            BASE_MODEL, dtype=torch.bfloat16, device_map="auto")
        model = PeftModel.from_pretrained(model, str(checkpoint)).merge_and_unload()
    else:
        if not (checkpoint / "config.json").exists():
            checkpoint = checkpoint / "final"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(checkpoint), dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return model, AutoProcessor.from_pretrained(BASE_MODEL)


def generate(model, processor, video: Path, fps: float, max_pixels: int,
             max_new_tokens: int, repetition_penalty: float) -> str:
    """Generate HTML for one video with greedy decoding.

    Args:
        model: Loaded model.
        processor: Qwen3-VL processor.
        video: Path to the video.
        fps: Frames per second sampled from the video.
        max_pixels: Pixel budget per frame.
        max_new_tokens: Maximum tokens to generate.
        repetition_penalty: Penalty for repeated tokens; 1.0 turns it off.

    Returns:
        Raw response text.
    """
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "video", "video": str(video), "fps": fps, "max_pixels": max_pixels},
            {"type": "text", "text": USER_PROMPT},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    if isinstance(video_kwargs.get("fps"), list):
        f = video_kwargs["fps"]
        video_kwargs["fps"] = float(f[0]) if f else fps

    inputs = processor(text=[text], images=images, videos=videos,
                       padding=True, return_tensors="pt", **video_kwargs).to(model.device)
    n_in = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             repetition_penalty=repetition_penalty)
    response = processor.batch_decode(out[:, n_in:], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0]
    del inputs, out
    gc.collect()
    torch.cuda.empty_cache()
    return response


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=["lora", "full"], required=True)
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="Training output dir, or one specific checkpoint inside it")
    ap.add_argument("--test-dir", type=Path, default=HERE.parent / "anim2code_data" / "data" / "test")
    ap.add_argument("--output-dir", type=Path, default=None, help="Default: output/<variant>_test")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="Only run the first N examples")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-pixels", type=int, default=112896)
    ap.add_argument("--max-new-tokens", type=int, default=16384)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    args = ap.parse_args()

    out_root = args.output_dir or HERE / "output" / f"{args.variant}_test"
    out_root.mkdir(parents=True, exist_ok=True)
    examples = sorted(p.parent.name for p in args.test_dir.glob("*/animation.mp4"))
    if not examples:
        raise SystemExit(f"no test videos under {args.test_dir}")
    mine = examples[:args.limit][args.shard::args.num_shards]
    todo = [e for e in mine if not (out_root / e / "full.html").exists()
            and not (out_root / e / "_error.txt").exists()]
    print(f"[{args.variant} shard {args.shard}/{args.num_shards}] "
          f"{len(mine)} assigned, {len(todo)} to do", flush=True)
    if not todo:
        return

    model, processor = load_model(args.variant, args.checkpoint)

    for i, ex in enumerate(todo, 1):
        d = out_root / ex
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            resp = generate(model, processor, args.test_dir / ex / "animation.mp4", args.fps,
                            args.max_pixels, args.max_new_tokens, args.repetition_penalty)
            (d / "_raw_output.txt").write_text(resp)
            html, err = parse_html(resp)
            if html is not None:
                (d / "full.html").write_text(html)
                tag = "OK"
            else:
                (d / "_error.txt").write_text(err)
                tag = f"ERROR {err}"
        except Exception as exc:
            (d / "_error.txt").write_text(f"{type(exc).__name__}: {exc}")
            tag = f"EXCEPTION {type(exc).__name__}"
        print(f"  [{i}/{len(todo)}] {ex} {tag} {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
