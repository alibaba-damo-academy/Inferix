#!/bin/bash

# Self-Forcing Quantized Inference Script
# Runs Self-Forcing model with DAX quantization for reduced memory usage

set -e

# Default parameters (can be overridden by environment variables)
CONFIG_PATH="${CONFIG_PATH:-example/self_forcing/configs/self_forcing_dmd.yaml}"
DEFAULT_CONFIG_PATH="${DEFAULT_CONFIG_PATH:-example/self_forcing/configs/default_config.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./weights/self_forcing/checkpoints/self_forcing_dmd.pt}"
PROMPT="${PROMPT:-A cat dancing on the moon}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-example/self_forcing/outputs_quantized}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-21}"
QUANT_TYPE="${QUANT_TYPE:-fp8}"  # fp8 or int8

# Project setup
export PYTHONPATH=`pwd`:$PYTHONPATH

# Create output directory
mkdir -p "$OUTPUT_FOLDER"

echo "=============================================="
echo "Self-Forcing Quantized Inference"
echo "=============================================="
echo "Config: $CONFIG_PATH"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output: $OUTPUT_FOLDER"
echo "Prompt: $PROMPT"
echo "Frames: $NUM_OUTPUT_FRAMES"
echo "Quantization: $QUANT_TYPE"
echo "=============================================="
echo

# Single GPU inference with quantization
python example/quantization/run_self_forcing_quantized.py \
    --config_path "$CONFIG_PATH" \
    --default_config_path "$DEFAULT_CONFIG_PATH" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --prompt "$PROMPT" \
    --output_folder "$OUTPUT_FOLDER" \
    --num_output_frames "$NUM_OUTPUT_FRAMES" \
    --quant_type "$QUANT_TYPE" \
    --use_ema

echo
echo "=============================================="
echo "✅ Quantized inference completed!"
echo "Results saved in: $OUTPUT_FOLDER"
echo "=============================================="
