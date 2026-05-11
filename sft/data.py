"""
HF dataset loader for anim2code SFT.

Reads `anim2code/anim2code-dev` data/{split}/metadata.jsonl + data/{split}/videos/*.mp4
Each example becomes a chat-formatted (video, html) pair.
"""
import json
import os
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset


SYSTEM_PROMPT = (
    "You are an expert web developer specializing in CSS animations and JavaScript. "
    "Watch the animation video and produce the complete HTML/CSS/JavaScript document "
    "that recreates it. Output only a single, complete HTML document."
)

USER_PROMPT = (
    "Watch this animation video carefully and write the complete HTML/CSS/JavaScript "
    "code that recreates it. Match timing, easing, colors, and motion as closely as "
    "possible. Output only the HTML document."
)


def load_split(local_dir: Path, split: str) -> list[dict]:
    """Read metadata.jsonl for a split and resolve video paths."""
    meta_path = local_dir / split / "metadata.jsonl"
    examples = []
    with open(meta_path) as f:
        for line in f:
            row = json.loads(line)
            video_path = local_dir / split / row["file_name"]
            if not video_path.exists():
                continue
            examples.append({
                "id": row["id"],
                "video_path": str(video_path),
                "full_html": row["full_html"],
            })
    return examples


class AnimSFTDataset(Dataset):
    """
    Wraps a list of (video, full_html) pairs.

    __getitem__ returns the messages list in chat format. Tokenization +
    video preprocessing happens in the collator so we can keep this
    process-fork friendly.
    """

    def __init__(self, examples: list[dict], fps: float = 1.0, max_pixels: int = 112896):
        self.examples = examples
        self.fps = float(fps)
        self.max_pixels = int(max_pixels)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "video", "video": ex["video_path"], "fps": self.fps, "max_pixels": self.max_pixels},
                {"type": "text", "text": USER_PROMPT},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": ex["full_html"]}]},
        ]
        return {"id": ex["id"], "messages": messages}


class Collator:
    """
    Apply Qwen3-VL's chat template + processor on a batch.

    Loss is computed only on assistant tokens (everything else is masked to -100).
    """

    def __init__(self, processor, max_text_len: int = 16384):
        self.processor = processor
        self.max_text_len = int(max_text_len)
        try:
            self.assistant_token_ids = self.processor.tokenizer.encode(
                "<|im_start|>assistant\n", add_special_tokens=False
            )
        except AttributeError:
            self.assistant_token_ids = None

    def __call__(self, batch: list[dict]) -> dict:
        from qwen_vl_utils import process_vision_info

        all_msgs = [b["messages"] for b in batch]
        # Full chat = with assistant content (target).
        texts_full = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in all_msgs
        ]
        # Prefix = same chat but stop at the assistant turn (so we know where labels start).
        texts_prefix = [
            self.processor.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True)
            for m in all_msgs
        ]
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            all_msgs, return_video_kwargs=True
        )
        # qwen_vl_utils 0.0.10+ returns fps as a list; the processor wants a float.
        if isinstance(video_kwargs.get("fps"), list):
            fps = video_kwargs["fps"]
            video_kwargs["fps"] = float(fps[0]) if fps else 1.0

        inputs = self.processor(
            text=texts_full,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
            **video_kwargs,
        )
        # Build label mask: only assistant tokens contribute to loss.
        labels = inputs["input_ids"].clone()
        for i, prefix_text in enumerate(texts_prefix):
            prefix_ids = self.processor.tokenizer(
                prefix_text, return_tensors="pt", add_special_tokens=False,
            )["input_ids"][0]
            n = min(prefix_ids.shape[0], labels.shape[1])
            labels[i, :n] = -100
        # Mask padding too.
        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = -100
        inputs["labels"] = labels
        return dict(inputs)
