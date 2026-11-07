# Animation2Code

Code for [Animation2Code: Evaluating Temporal Visual Reasoning in Video-to-Code Generation](https://arxiv.org/abs/2606.28593).

| Folder | Contents |
|---|---|
| [`recording/`](recording) | Deterministic HTML-to-MP4 recorder |
| [`baselines/`](baselines) | Zero-shot inference for Gemini, GPT, Claude, LLaMA and Qwen3-VL |
| [`sft/`](sft) | LoRA and full supervised fine-tuning of Qwen3-VL |
| [`iterative_refinement/`](iterative_refinement) | Four iterative refinement variants |
| [`evaluation/`](evaluation) | Appearance, temporal and supporting metrics |

## Dataset

Download the [dataset](https://huggingface.co/datasets/anim2code/dataset) from the repo root. The code expects it in `anim2code_data/`.

```bash
hf download anim2code/dataset --repo-type dataset --local-dir anim2code_data
```

Each example is `anim2code_data/data/{train,val,test}/<id>/` with `animation.mp4`, `full.html` and `metadata.json`.

Run the commands in each folder's README from that folder.
