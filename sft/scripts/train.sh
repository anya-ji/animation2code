#!/usr/bin/env bash
# Train on every GPU in GPUS. Accumulation is divided by the rank count to keep the effective batch.
#   GPUS=0,1,2,3,4,5,6,7 bash scripts/train.sh lora|full [train.py args]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

METHOD="${1:?usage: train.sh lora|full [train.py args]}"
shift
CONFIG="configs/$METHOD.yaml"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:?set GPUS or CUDA_VISIBLE_DEVICES}}"
NPROC=$(awk -F, '{print NF}' <<< "$GPUS")
CFG_ACCUM=$(awk '/^gradient_accumulation_steps:/{print $2}' "$CONFIG")

if [ $(( CFG_ACCUM % NPROC )) -ne 0 ]; then
  echo "ERROR: accumulation ($CFG_ACCUM) is not divisible by the GPU count ($NPROC)." >&2
  exit 1
fi

# Offline after this download: ranks racing on the Hub cache once built the wrong architecture.
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-8B-Instruct')"
export HF_HUB_OFFLINE=1
export WANDB_PROJECT="${WANDB_PROJECT:-anim2code-sft}"

CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc_per_node="$NPROC" --nnodes=1 train.py \
  --method "$METHOD" --config "$CONFIG" --data-dir ../anim2code_data/data \
  --grad-accum $(( CFG_ACCUM / NPROC )) "$@"
