# SFT

Supervised fine-tuning of Qwen3-VL-8B-Instruct on the Animation2Code dataset using HuggingFace Trainer with Liger kernels.

## Setup

```bash
pip install torch transformers peft liger-kernel qwen-vl-utils pyyaml tqdm datasets huggingface_hub
```

Set `HF_TOKEN` in your environment if needed for private model/dataset access.

## Data

Training data is loaded from `anim2code/anim2code-dev` (HuggingFace). Pass `--data-dir` to use a local copy:

```
data_dir/
  train/
    metadata.jsonl
    videos/codepen-XXXX-yyyy.mp4
  val/
    metadata.jsonl
    videos/...
```

## Training

```bash
# LoRA (recommended for single GPU)
bash sft/scripts/train.sh lora

# Full fine-tuning (multi-GPU, vision encoder frozen)
bash sft/scripts/train.sh full

# Smoke test (1 step, no wandb)
bash sft/scripts/train.sh lora --smoke --report-to none

# With local data
python sft/train.py --method lora --config sft/configs/lora.yaml --data-dir /path/to/data
```

Checkpoints are saved to `checkpoints/lora/` or `checkpoints/full/`.

## Inference

```bash
python sft/infer.py \
    --run-type lora \
    --checkpoint-dir checkpoints/lora \
    --output-dir results/sft_lora \
    --test-data-dir /path/to/dataset/test
```

Output layout: `results/sft_lora/<example_id>/full.html`.

## Key config options

| Key | Description |
|-----|-------------|
| `fps` | Video frames per second fed to the model |
| `max_pixels` | Max visual tokens (lower = less GPU memory) |
| `max_text_len` | Sequence truncation length |
| `use_liger_kernel` | Enable Liger fused kernels (default: true) |
| `lora.r` | LoRA rank |
| `learning_rate` | Peak LR |
| `epochs` | Training epochs |
