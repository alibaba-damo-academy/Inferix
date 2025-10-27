#!/bin/bash
# Example script for running self-forcing pipeline with profiling in distributed mode

# Default parameters (can be overridden by environment variables)
CONFIG_PATH="${CONFIG_PATH:-example/self_forcing/configs/self_forcing_dmd.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./weights/self_forcing/checkpoints/self_forcing_dmd.pt}"
PROMPT="${PROMPT:-A cat dancing on the moon; A robot walking in a forest}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-example/self_forcing/outputs}"
ULYSSES_SIZE="${ULYSSES_SIZE:-1}"
RING_SIZE="${RING_SIZE:-2}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')}"

# Project setup
export PYTHONPATH=`pwd`:$PYTHONPATH

# Create output directory
mkdir -p "$OUTPUT_FOLDER"

# Create profiling output directory under output folder
PROFILING_OUTPUT="${PROFILING_OUTPUT:-$OUTPUT_FOLDER/profiling}"
mkdir -p "$PROFILING_OUTPUT"

echo "Running Self-forcing inference with profiling..."
echo "Config: $CONFIG_PATH"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output: $OUTPUT_FOLDER"
echo "Profiling output: $PROFILING_OUTPUT"
echo "Prompt: $PROMPT"
echo "GPUs: $NUM_GPUS"
echo

torchrun --nnodes=1 --nproc-per-node=$NUM_GPUS \
    example/profiling/self_forcing_profiling.py \
    --config_path "$CONFIG_PATH" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --prompt "$PROMPT" \
    --output_folder "$OUTPUT_FOLDER" \
    --use_ema \
    --ulysses_size="$ULYSSES_SIZE" \
    --ring_size="$RING_SIZE" \
    --enable_profiling \
    --profiling_config "example/profiling/configs/profiling_config.yaml" \
    --profiling_output_dir "$PROFILING_OUTPUT"

echo "Self-forcing inference with profiling completed."
echo "Results saved in: $OUTPUT_FOLDER"
echo "Profiling reports saved in: $PROFILING_OUTPUT"
echo
echo "To view the enhanced profiling report with FPS and BPS metrics:"
echo "  Open the HTML report in $PROFILING_OUTPUT"
echo "  Or view the JSON report for detailed metrics"