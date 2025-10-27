#!/bin/bash

# MAGI 4.5B Model Inference Script

set -e

# Default parameters
CONFIG_FILE="${CONFIG_FILE:-example/magi/configs/4.5B/4.5B_base_config.json}"
MODE="${MODE:-t2v}"
PROMPT="${PROMPT:-Good Boy}"
OUTPUT_PATH="${OUTPUT_PATH:-example/magi/assets/output_4.5B_t2v.mp4}"

# Setup environment
export PYTHONPATH=`pwd`:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running MAGI 4.5B inference..."
echo "Config: $CONFIG_FILE"
echo "Mode: $MODE"
echo "Prompt: $PROMPT"
echo "Output: $OUTPUT_PATH"
echo

torchrun --nnodes=1 --nproc-per-node=1 \
    example/magi/run_magi.py \
    --config_file "$CONFIG_FILE" \
    --mode "$MODE" \
    --prompt "$PROMPT" \
    --output_path "$OUTPUT_PATH"

echo "MAGI 4.5B inference completed. Output saved to: $OUTPUT_PATH"