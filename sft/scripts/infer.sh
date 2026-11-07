#!/usr/bin/env bash
# Run test-set inference with one shard per GPU.
#   GPUS=0,1,2,3,4,5,6,7 bash scripts/infer.sh lora|full CHECKPOINT [infer.py args]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VARIANT="${1:?usage: infer.sh lora|full CHECKPOINT [infer.py args]}"
CHECKPOINT="${2:?usage: infer.sh lora|full CHECKPOINT [infer.py args]}"
shift 2
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:?set GPUS or CUDA_VISIBLE_DEVICES}}"
mkdir -p logs

IFS=, read -ra IDS <<< "$GPUS"
PIDS=()
for i in "${!IDS[@]}"; do
  CUDA_VISIBLE_DEVICES="${IDS[$i]}" python infer.py --variant "$VARIANT" --checkpoint "$CHECKPOINT" \
    --shard "$i" --num-shards "${#IDS[@]}" "$@" > "logs/infer_${VARIANT}_shard$i.log" 2>&1 &
  PIDS+=($!)
done
for pid in "${PIDS[@]}"; do
  wait "$pid"
done
