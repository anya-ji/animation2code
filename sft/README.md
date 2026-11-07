# SFT

LoRA and full-parameter fine-tuning of Qwen3-VL-8B-Instruct, using a plain Hugging Face Trainer.

| Model | Config | Method | Checkpoint used |
|---|---|---|---|
| LoRA | `configs/lora.yaml` | r=64, 5 epochs, lr 1e-4, effective batch 8 | Lowest `eval_loss` |
| Full | `configs/full.yaml` | ZeRO-3, vision tower frozen, 3 epochs, lr 1e-5, effective batch 16 | Final epoch |

## Setup

```bash
pip install -r requirements.txt
```

Download the dataset as described in the [root README](../README.md). Both models were trained on 8 GPUs with Python 3.11, and `requirements.txt` pins the original package versions.

## Train

The GPU count must divide the config's accumulation (8 for LoRA, 16 for full). Scripts use `GPUS`, or `CUDA_VISIBLE_DEVICES` if it isn't set.

```bash
GPUS=0,1,2,3,4,5,6,7 bash scripts/train.sh lora
GPUS=0,1,2,3,4,5,6,7 bash scripts/train.sh full

# One-step test run
GPUS=0,1 bash scripts/train.sh lora --smoke --max-train 2 --max-eval 1 --report-to none
```

Checkpoints go to `checkpoints/{lora,full}/`.

## Infer

Decoding is greedy with `repetition_penalty=1.05` and `max_new_tokens=16384`. Given a training output folder, inference picks the lowest `eval_loss` checkpoint for LoRA and `final/` for full. You can also pass one checkpoint directly.

```bash
GPUS=0,1,2,3,4,5,6,7 bash scripts/infer.sh lora checkpoints/lora
GPUS=0,1,2,3,4,5,6,7 bash scripts/infer.sh full checkpoints/full
```

Outputs go to `output/{lora,full}_test/<id>/full.html`. Record them with `baselines/record.py`, then score them with [`evaluation/`](../evaluation).
