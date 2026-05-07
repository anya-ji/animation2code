# Animation2Code

Code release for the Animation2Code paper.

## Structure

```
baselines/            Inference for all baseline models (Gemini, GPT, Qwen3-VL, Claude, Molmo)
sft/                  Supervised fine-tuning recipe (LoRA and full-parameter) via LLaMA-Factory
iterative_refinement/ Four iterative refinement variants (baseline, describe_first, dual_critic, caption_bridge)
recording/            Recording pipeline v2: HTML → MP4 via Puppeteer
evaluation/           Evaluation suite (precision, recall, fullframe, bbox, temporal_fidelity, tsed)
```

Each directory has its own README with setup instructions and usage examples.

## Dataset

The dataset is available on HuggingFace: `anim2code/anim2code`.

Each example has:
- `animation.mp4` — ground-truth animation video
- `full.html` — source HTML/CSS/JS code
- `metadata.jsonl` — id, category, title, url
