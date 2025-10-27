#!/bin/bash

# MAGI 24B Model Inference Script

set -e

# Default parameters
CONFIG_FILE="${CONFIG_FILE:-example/magi/configs/24B/24B_distill_quant_config.json}"
MODE="${MODE:-i2v}"
PROMPT="${PROMPT:-Good Boy}"
IMAGE_PATH="${IMAGE_PATH:-example/magi/assets/image.jpeg}"
OUTPUT_PATH="${OUTPUT_PATH:-example/magi/assets/output_24B_i2v.mp4}"

# Setup environment
export PYTHONPATH=`pwd`:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running MAGI 24B inference..."
echo "Config: $CONFIG_FILE"
echo "Mode: $MODE"
echo "Prompt: $PROMPT"
echo "Output: $OUTPUT_PATH"
echo

torchrun --nnodes=1 --nproc-per-node=4 \
    example/magi/run_magi.py \
    --config_file "$CONFIG_FILE" \
    --mode "$MODE" \
    --prompt "$PROMPT" \
    --image_path "$IMAGE_PATH" \
    --output_path "$OUTPUT_PATH"

echo "MAGI 24B inference completed. Output saved to: $OUTPUT_PATH"