#!/bin/bash

# Self-forcing Model Inference Script

set -e

# Default parameters (can be overridden by environment variables)
CONFIG_PATH="${CONFIG_PATH:-example/self_forcing/configs/self_forcing_dmd.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./weights/self_forcing/checkpoints/self_forcing_dmd.pt}"
PROMPT="${PROMPT:-A cat dancing on the moon; A robot walking in a forest}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-example/self_forcing/outputs}"
ULYSSES_SIZE="${ULYSSES_SIZE:-1}"
RING_SIZE="${RING_SIZE:-2}"
WORLD_SIZE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Project setup
export PYTHONPATH=`pwd`:$PYTHONPATH

# Create output directory
mkdir -p "$OUTPUT_FOLDER"

echo "Running Self-forcing inference..."
echo "Config: $CONFIG_PATH"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output: $OUTPUT_FOLDER"
echo "Prompt: $PROMPT"
echo

torchrun --nnodes=1 --nproc-per-node=$WORLD_SIZE \
    example/self_forcing/run_self_forcing.py \
    --config_path "$CONFIG_PATH" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --prompt "$PROMPT" \
    --output_folder "$OUTPUT_FOLDER" \
    --use_ema \
    --ulysses_size="$ULYSSES_SIZE" \
    --ring_size="$RING_SIZE" 

echo "Self-forcing inference completed. Results saved in: $OUTPUT_FOLDER"