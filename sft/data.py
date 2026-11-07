"""Dataset and collator for fine-tuning Qwen3-VL on (video, HTML) pairs.

Each example becomes a chat with the video and prompt as input and the HTML as the assistant reply.
"""
from pathlib import Path

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


def load_split(data_dir: Path, split: str) -> list[dict]:
    """Load one split.

    Args:
        data_dir: Directory containing <split>/<id>/animation.mp4 and full.html.
        split: "train" or "val".

    Returns:
        List of dicts with id, video_path and full_html, sorted by id.
    """
    examples = []
    for example_dir in sorted((data_dir / split).iterdir()):
        video = example_dir / "animation.mp4"
        html = example_dir / "full.html"
        if video.exists() and html.exists():
            examples.append({
                "id": example_dir.name,
                "video_path": str(video),
                "full_html": html.read_text(encoding="utf-8"),
            })
    return examples


class AnimSFTDataset(Dataset):
    """Return each example as chat messages.

    Tokenization and video decoding happen in the collator, so dataloader workers stay light.

    Args:
        examples: Output of load_split.
        fps: Frames per second sampled from each video.
        max_pixels: Pixel budget per frame.
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
    """Turn a list of examples into one batch.

    Applies the chat template, decodes the videos and tokenizes. Labels are -100 everywhere except
    assistant tokens, so the loss only covers the generated HTML.

    Args:
        processor: Qwen3-VL processor.
        max_text_len: Sequences longer than this are truncated, video tokens included.
    """

    def __init__(self, processor, max_text_len: int = 16384):
        self.processor = processor
        self.max_text_len = int(max_text_len)

    def __call__(self, batch: list[dict]) -> dict:
        from qwen_vl_utils import process_vision_info

        all_msgs = [b["messages"] for b in batch]
        texts_full = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in all_msgs
        ]
        texts_prefix = [
            self.processor.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True)
            for m in all_msgs
        ]
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            all_msgs, return_video_kwargs=True
        )
        # qwen_vl_utils returns fps as a list; the processor wants a float.
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
        labels = inputs["input_ids"].clone()
        for i, prefix_text in enumerate(texts_prefix):
            prefix_ids = self.processor.tokenizer(
                prefix_text, return_tensors="pt", add_special_tokens=False,
            )["input_ids"][0]
            n = min(prefix_ids.shape[0], labels.shape[1])
            labels[i, :n] = -100
        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = -100
        inputs["labels"] = labels
        return dict(inputs)
