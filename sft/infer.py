"""
Inference script for SFT checkpoints on the test set.

Usage:
    python sft/infer.py --run-type lora \
        --checkpoint-dir checkpoints/lora/final \
        --output-dir results/sft_lora \
        --test-data-dir /path/to/dataset/test
"""

import argparse
import gc
import json
import os
import re
import time
from pathlib import Path

BASE_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
HF_DATASET = "anim2code/anim2code-dev"

VIDEO_FPS = 2.0
VIDEO_MAX_PIXELS = 128 * 28 * 28  # 100352
MAX_NEW_TOKENS = 8192

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


def find_best_checkpoint(base_dir: str) -> str:
    """Return the checkpoint with lowest eval loss, or latest if no trainer_state."""
    base = Path(base_dir)
    checkpoints = sorted(base.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if not checkpoints:
        return base_dir

    best_ckpt = None
    best_loss = float("inf")
    for ckpt in checkpoints:
        state_file = ckpt / "trainer_state.json"
        if state_file.exists():
            state = json.loads(state_file.read_text())
            logs = state.get("log_history", [])
            for entry in reversed(logs):
                if "eval_loss" in entry:
                    if entry["eval_loss"] < best_loss:
                        best_loss = entry["eval_loss"]
                        best_ckpt = str(ckpt)
                    break

    if best_ckpt:
        print(f"Best checkpoint: {best_ckpt} (eval_loss={best_loss:.4f})")
        return best_ckpt
    return str(checkpoints[-1])


def load_test_data(test_data_dir: str) -> tuple[list[str], Path]:
    test_dir = Path(test_data_dir)
    if not (test_dir / "metadata.jsonl").exists():
        # Try HF download
        from huggingface_hub import snapshot_download
        print(f"Downloading test split from HuggingFace: {HF_DATASET}")
        dl_dir = Path(snapshot_download(
            repo_id=HF_DATASET,
            repo_type="dataset",
            allow_patterns=["data/test/*"],
            token=os.environ.get("HF_TOKEN"),
        ))
        test_dir = dl_dir / "data" / "test"

    examples = []
    with open(test_dir / "metadata.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line)["id"])
    return sorted(examples), test_dir / "videos"


def load_model(run_type: str, checkpoint_path: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if run_type == "lora":
        from peft import PeftModel
        print(f"Loading base model: {BASE_MODEL_ID}")
        base = AutoModelForImageTextToText.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"Applying LoRA adapter from: {checkpoint_path}")
        model = PeftModel.from_pretrained(base, checkpoint_path)
        model = model.merge_and_unload()
    else:
        print(f"Loading full SFT model from: {checkpoint_path}")
        model = AutoModelForImageTextToText.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    processor = AutoProcessor.from_pretrained(
        checkpoint_path if run_type != "lora" else BASE_MODEL_ID,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    return model, processor


def generate(model, processor, video_path: Path) -> tuple[str, int]:
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video_path),
                    "max_pixels": VIDEO_MAX_PIXELS,
                    "fps": VIDEO_FPS,
                },
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
    )
    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None

    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        return_tensors="pt",
        padding=True,
        do_resize=False,
        **video_kwargs,
    ).to(model.device)

    input_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    generated_ids = output_ids[:, input_tokens:]
    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    del inputs, output_ids, generated_ids
    gc.collect()
    torch.cuda.empty_cache()

    return response, input_tokens


def parse_html(response: str) -> tuple[str | None, str]:
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


def run_inference(run_type: str, checkpoint_dir: str, output_dir: str, test_data_dir: str):
    from tqdm import tqdm

    checkpoint_path = find_best_checkpoint(checkpoint_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading test data from {test_data_dir}...")
    examples, video_dir = load_test_data(test_data_dir)
    print(f"Test examples: {len(examples)}")

    remaining = [
        ex for ex in examples
        if not (out_dir / ex / "full.html").exists()
        and not (out_dir / ex / "_error.txt").exists()
    ]
    print(f"Already done: {len(examples) - len(remaining)}  Remaining: {len(remaining)}")

    if not remaining:
        print("Nothing to process.")
        return

    model, processor = load_model(run_type, checkpoint_path)
    print(f"Model loaded. Running inference on {len(remaining)} examples ...")

    ok = err = 0
    t_start = time.time()

    for i, name in enumerate(tqdm(remaining, desc="inference", unit="ex")):
        video_path = video_dir / f"{name}.mp4"
        ex_dir = out_dir / name
        ex_dir.mkdir(parents=True, exist_ok=True)

        try:
            t0 = time.time()
            response, input_tokens = generate(model, processor, video_path)
            elapsed = time.time() - t0

            (ex_dir / "_raw_output.txt").write_text(response, encoding="utf-8")

            html, error = parse_html(response)
            if html is not None:
                (ex_dir / "full.html").write_text(html, encoding="utf-8")
                ok += 1
                print(
                    f"[{i+1}/{len(remaining)}] OK    {name}"
                    f"  tokens={input_tokens}  chars={len(response)}  {elapsed:.1f}s"
                )
            else:
                (ex_dir / "_error.txt").write_text(error, encoding="utf-8")
                err += 1
                print(f"[{i+1}/{len(remaining)}] ERROR {name}: {error}  {elapsed:.1f}s")

        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {e}"
            (ex_dir / "_error.txt").write_text(error_msg, encoding="utf-8")
            err += 1
            print(f"[{i+1}/{len(remaining)}] EXCEPTION {name}: {error_msg}")
            traceback.print_exc()
            gc.collect()

    total = time.time() - t_start
    print(f"\nDone. OK: {ok}  Errors: {err}  Total: {total/60:.1f}min")
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-type", choices=["lora", "full"], required=True)
    parser.add_argument("--checkpoint-dir", required=True, help="Path to checkpoint directory")
    parser.add_argument("--output-dir", required=True, help="Where to write per-example HTML")
    parser.add_argument("--test-data-dir", required=True,
                        help="Directory containing metadata.jsonl and videos/")
    args = parser.parse_args()
    run_inference(args.run_type, args.checkpoint_dir, args.output_dir, args.test_data_dir)
