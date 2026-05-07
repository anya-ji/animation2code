"""
Qwen3-VL model implementation using HuggingFace Transformers.

Model is loaded once per model_id and cached at module level — if generator,
critic, and editor all specify the same model_id, only one copy is loaded.

For multi-video inputs (critique stage), both videos are passed as separate
video items in the message content list.

Config keys (from config.yaml generator/critic/editor.qwen3vl block):
  model_id       : e.g. "Qwen/Qwen3-VL-8B-Instruct"
  fps            : frames-per-second to sample from each video
  max_pixels     : pixel budget per frame (e.g. 112896 = 336*336)
  max_new_tokens : max tokens to generate
  temperature    : float
  top_p          : float
"""

import gc
from pathlib import Path

# torch, transformers, and qwen_vl_utils are imported lazily inside _load_model()
# so that this file can be imported on machines without GPU/torch installed.
from modules.models.base import BaseModel

# Module-level cache: model_id → (model, processor)
_MODEL_CACHE: dict[str, tuple] = {}


def _load_model(model_id: str):
    """Load and cache a Qwen3VL model. Returns (model, processor)."""
    if model_id not in _MODEL_CACHE:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        print(f"[qwen3vl] Loading model: {model_id}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype="auto",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        _MODEL_CACHE[model_id] = (model, processor)
        print(f"[qwen3vl] Model loaded: {model_id}")
    return _MODEL_CACHE[model_id]


class Qwen3VLModel(BaseModel):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.model_id = cfg["model_id"]
        self.fps = float(cfg.get("fps", 2.0))
        self.max_pixels = int(cfg.get("max_pixels", 112896))
        self.max_new_tokens = int(cfg.get("max_new_tokens", 8192))
        self.temperature = float(cfg.get("temperature", 0.2))
        self.top_p = float(cfg.get("top_p", 0.9))
        # Model is loaded lazily on first call via _get_model()
        self._model = None
        self._processor = None

    def _get_model(self):
        if self._model is None:
            self._model, self._processor = _load_model(self.model_id)
        return self._model, self._processor

    def _video_item(self, video_path: Path) -> dict:
        """Build a Qwen3VL video content item."""
        return {
            "type": "video",
            "video": str(video_path),
            "fps": self.fps,
            "max_pixels": self.max_pixels,
        }

    def _infer(self, messages: list) -> str:
        """Run inference on a messages list and return the generated text."""
        import torch
        from qwen_vl_utils import process_vision_info

        model, processor = self._get_model()

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                top_p=self.top_p if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
            )

        generated_ids = output_ids[:, input_len:]
        response = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Free GPU memory from this batch
        del inputs, output_ids, generated_ids
        gc.collect()
        torch.cuda.empty_cache()

        return response

    # ── BaseModel interface ───────────────────────────────────────────────

    def generate_code(
        self,
        video_path: Path,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    self._video_item(video_path),
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        return self._infer(messages)

    def critique(
        self,
        target_video: Path,
        generated_video: Path,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Video 1 — TARGET animation:"},
                    self._video_item(target_video),
                    {"type": "text", "text": "Video 2 — CURRENT attempt:"},
                    self._video_item(generated_video),
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        return self._infer(messages)

    def edit_code(
        self,
        current_code: str,
        feedback: str,
        system_prompt: str,
        user_prompt: str,
        target_video: Path | None = None,
    ) -> str:
        # Editor is text-only; target_video is ignored for Qwen3VL
        # since passing large videos alongside long code exhausts context.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._infer(messages)
