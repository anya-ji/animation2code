# SFT

Supervised fine-tuning of Qwen3-VL-8B-Instruct on Animation2Code via [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

## Setup

```bash
pip install llamafactory deepspeed qwen-vl-utils
```

## Data preparation

Convert the dataset to LLaMA-Factory's ShareGPT format:

```bash
python prepare_data.py \
    --data-dir /path/to/dataset \
    --output-dir sft/data
```

This writes `sft/data/train.json`, `sft/data/val.json`, and `sft/data/dataset_info.json`.

## Training

Edit `configs/lora.yaml` or `configs/full.yaml` to set `dataset_dir` and `output_dir`, then:

```bash
# LoRA (rank=64, lr=1e-4, 3 epochs)
bash scripts/run_lora.sh configs/lora.yaml

# Full-parameter (lr=5e-6, 3 epochs, DeepSpeed ZeRO-3)
bash scripts/run_full.sh configs/full.yaml
```

Both scripts use `FORCE_TORCHRUN=1` with `NPROC_PER_NODE=4`. Adjust to match your GPU count.

Set `WANDB_API_KEY` in your environment (or `.env`) and `WANDB_PROJECT=anim2code-sft` for logging.

## Inference

After training, use the baseline inference scripts (`baselines/run.py`) with a config pointing to your checkpoint directory.
