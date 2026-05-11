#!/bin/bash
# Train Qwen3-VL SFT (LoRA or full).
#
# Usage:
#   bash sft/scripts/train.sh lora           # LoRA (default)
#   bash sft/scripts/train.sh full           # Full fine-tuning
#   bash sft/scripts/train.sh lora --smoke   # Smoke test (1 step)
set -e

METHOD=${1:-lora}
shift || true
CONFIG="sft/configs/${METHOD}.yaml"

export WANDB_PROJECT=anim2code-sft
python sft/train.py --method "$METHOD" --config "$CONFIG" "$@"
