#!/bin/bash

# CausVid Model Inference Script

set -e

# Default parameters (can be overridden by environment variables)
CONFIG_PATH="${CONFIG_PATH:-example/causvid/configs/default_config.yaml}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-example/causvid/outputs}"
CHECKPOINT_FOLDER="${CHECKPOINT_FOLDER:-./weights/CausVid/autoregressive_checkpoint}"
WAN_BASE_MODEL_PATH="${WAN_BASE_MODEL_PATH:-./weights/Wan2.1-T2V-1.3B}"
PROMPT="${PROMPT:-A young man with curly hair, wearing a red jacket with a white hoodie underneath, sits in the drivers seat of a car. The man appears to be driving through a city, passing by a gas station and construction sites, with a focus on urban driving.}"
ULYSSES_SIZE="${ULYSSES_SIZE:-1}"
RING_SIZE="${RING_SIZE:-8}"
NUM_ROLLOUT="${NUM_ROLLOUT:-6}"
WORLD_SIZE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Project setup
export PYTHONPATH=`pwd`:$PYTHONPATH

# Create output directory
mkdir -p "$OUTPUT_FOLDER"

echo "Running CausVid inference..."
echo "Config: $CONFIG_PATH"
echo "Checkpoint: $CHECKPOINT_FOLDER"
echo "Wan Base Model: $WAN_BASE_MODEL_FOLDER"
echo "Output: $OUTPUT_FOLDER"
echo "Prompt: $PROMPT"
echo

torchrun --nnodes=1 --nproc-per-node=$WORLD_SIZE \
    example/quantization/run_causvid_quantized.py \
    --config_path "$CONFIG_PATH" \
    --output_folder "$OUTPUT_FOLDER" \
    --checkpoint_folder "$CHECKPOINT_FOLDER" \
    --wan_base_model_path "$WAN_BASE_MODEL_PATH" \
    --prompt "$PROMPT" \
    --ulysses_size="$ULYSSES_SIZE" \
    --ring_size="$RING_SIZE" \
    --num_rollout="$NUM_ROLLOUT"

echo "CausVid inference completed. Results saved in: $OUTPUT_FOLDER"