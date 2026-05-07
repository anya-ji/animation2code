#!/bin/bash
# LoRA SFT training with LLaMA-Factory
# Adjust NPROC_PER_NODE to match available GPUs.

CONFIG=${1:-sft/configs/lora.yaml}

export WANDB_PROJECT=anim2code-sft  # set WANDB_API_KEY in env or .env

export FORCE_TORCHRUN=1
export NPROC_PER_NODE=4
export NNODES=1
export RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

echo "Starting LoRA SFT with $NPROC_PER_NODE GPUs ..."
llamafactory-cli train "$CONFIG"
echo "Training finished with exit code $?"
