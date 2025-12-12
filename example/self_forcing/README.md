# Self-Forcing Inference Example

Self-Forcing video generation model inference example.

**GitHub Repository**: [Self-Forcing](https://github.com/guandeh17/Self-Forcing)

## Prerequisites

### Download Model Weights

Suppose `./weights` under the Inferix project is the model weight directory.

1. **Download Wan2.1-T2V-1.3B Base Model**:
   ```bash
   huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir ./weights/Wan2.1-T2V-1.3B
   ```

2. **Download Self-Forcing Checkpoint**:
   ```bash
   huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir ./weights/self_forcing
   ```

## Inference

### Basic Usage

```bash
export PYTHONPATH=`pwd`:$PYTHONPATH
torchrun --nnodes=1 --nproc-per-node=2 \
    example/self_forcing/run_self_forcing.py \
    --config_path example/self_forcing/configs/self_forcing_dmd.yaml \
    --output_folder example/self_forcing/outputs \
    --checkpoint_path ./weights/self_forcing/checkpoints/self_forcing_dmd.pt \
    --prompt "A cat dancing on the moon; A robot walking in a forest" \
    --use_ema --ulysses_size=1 --ring_size=2
```

### Parameter Description

- `--config_path`: Configuration file path
- `--output_folder`: Output video save directory
- `--checkpoint_path`: Self-Forcing model checkpoint path
- `--prompt`: Text prompt for video generation (multiple prompts separated by semicolons)
- `--image_path`: Input image path for Image-to-Video (I2V) generation
- `--i2v`: Enable Image-to-Video mode (requires --image_path)
- `--num_output_frames`: Number of frames to generate (default: 21)
- `--use_ema`: Use Exponential Moving Average weights
- `--seed`: Random seed for generation (default: 0)
- `--num_samples`: Number of samples to generate per prompt (default: 1)
- `--save_with_index`: Save videos using index instead of prompt as filename
- `--ulysses_size`: Ulysses parallel size (default: 1)
- `--ring_size`: Ring parallel size (default: 1)

### Configuration File

Use the `example/self_forcing/configs/self_forcing_dmd.yaml` configuration file, which contains detailed parameter settings for the Self-Forcing model.

Key configuration parameters:
- `denoising_step_list`: Denoising steps for the semi-autoregressive process
- `guidance_scale`: Classifier-free guidance scale
- `num_frame_per_block`: Number of frames per generation block
- `timestep_shift`: Time step shift parameter
- `warp_denoising_step`: Whether to warp denoising steps

## Test Data

The project provides the `MovieGenVideoBench_extended.txt` test file containing test samples for evaluating model performance. You can replace this file with your own test data.

## Performance Profiling

Inferix includes built-in performance profiling capabilities with enhanced diffusion model analysis. To enable profiling, add the `--enable_profiling` flag to your command:

```bash
torchrun --nnodes=1 --nproc-per-node=2 \
    example/self_forcing/run_self_forcing.py \
    --config_path example/self_forcing/configs/self_forcing_dmd.yaml \
    --output_folder example/self_forcing/outputs \
    --checkpoint_path ./weights/self_forcing/checkpoints/self_forcing_dmd.pt \
    --prompt "A cat dancing on the moon" \
    --use_ema --ulysses_size=1 --ring_size=2 \
    --enable_profiling
```

Profiling reports will be generated in the output directory, providing detailed performance metrics for different stages of the pipeline, including:

- **Diffusion Model Analysis**: FPS (Frames Per Second) and BPS (Blocks Per Second) metrics
- **Model Analysis**: Parameter count and computational complexity
- **Block Computation Analysis**: Performance metrics for each video block

The enhanced profiling system provides real-time performance insights that are particularly valuable for optimizing real-time video generation.