# CausVid Inference Example

CausVid video generation model inference example.

**GitHub Repository**: [CausVid](https://github.com/tianweiy/CausVid)

## Prerequisites

### Download Model Weights

Suppose `./weights` under the Inferix project is the model weight directory.
1. **Download Wan2.1-T2V-1.3B Base Model**:
   ```bash
   huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./weights/Wan2.1-T2V-1.3B
   ```

2. **Download CausVid Checkpoint**:
   ```bash
   huggingface-cli download tianweiy/CausVid --include "autoregressive_checkpoint/*" --local-dir ./weights/CausVid
   ```

   > **Note**: For CausVid, you only need to download checkpoints in the `autoregressive_checkpoint` subfolder.

## Inference

### Basic Usage

```bash
export PYTHONPATH=`pwd`:$PYTHONPATH
torchrun --nnodes=1 --nproc-per-node=8 example/causvid/run_causvid.py \
    --config_path example/causvid/configs/causvid.yaml \
    --output_folder example/causvid/outputs \
    --checkpoint_folder ./weights/CausVid \
    --wan_base_model_path ./weights/Wan2.1-T2V-1.3B \
    --prompt "A young man with curly hair, wearing a red jacket with a white hoodie underneath, sits in the driver's seat of a car. The man appears to be driving through a city, passing by a gas station and construction sites, with a focus on urban driving." \
    --ulysses_size=1 \
    --ring_size=8 \
    --num_rollout=6 \
    # --diff_prompt=True \
    # --is_interactive=True \
    # --diff_prompt_file example/causvid/configs/prompts.yaml \
```

### Parameter Description

- `--config_path`: Configuration file path
- `--output_folder`: Output video save directory
- `--checkpoint_folder`: CausVid checkpoint folder path
- `--wan_base_model_path`: Wan2.1 base model folder path
- `--prompt`: Text prompt for video generation (multiple prompts separated by semicolons), when `--diff_prompt` is `True`, multiple prompts will be assigned to different chunk
- `--num_rollout`: Number of rollout steps (default: 3)
- `--num_overlap_frames`: Number of overlap frames between rollouts (default: 3)
- `--ulysses_size`: Ulysses parallel size (default: 1)
- `--ring_size`: Ring parallel size (default: 1)
- `--diff_prompt`: Whether assign different prompt to different chunk (default: False)
- `--is_interactive`: Whether input prompt from shell when `--diff_prompt` is True (default: False)
- `--diff_prompt_file`: Prompt yaml file path, loading prompt for every chunk from this file when `--diff_prompt` is `True` (default: None)

### Quick Start Script

You can also use the provided startup script:

```bash
bash causvid.sh
```

Make sure to modify the model paths and other parameters in the script before use.

### Configuration File

The `causvid.yaml` configuration file contains key parameters for the CausVid model:
- `denoising_step_list`: Denoising steps for the semi-autoregressive process
- `real_guidance_scale`: Guidance scale for generation
- `num_frame_per_block`: Number of frames per generation block
- `timestep_shift`: Time step shift parameter
- `generator_task`: Task type for the generator

## Performance Profiling

Inferix includes built-in performance profiling capabilities with enhanced diffusion model analysis. To enable profiling, modify the run script to pass a profiling configuration to the pipeline:

```python
from inferix.profiling.config import ProfilingConfig

# Create profiling configuration
profiling_config = ProfilingConfig(
    enabled=True,
    output_dir="./profiling_reports",
    real_time_display=True,
    profile_diffusion_steps=True,
    profile_block_computation=True,
    profile_model_parameters=True
)

# Pass to pipeline initialization
pipeline = CausVidPipeline(
    config_path=args.config_path,
    wan_base_model_folder=args.wan_base_model_folder,
    device_id=device_id,
    rank=rank,
    ulysses_size=ulysses_degree,
    ring_size=ring_degree,
    profiling_config=profiling_config  # Enable profiling
)
```

Profiling reports will be generated in the specified output directory, providing detailed performance metrics including:

- **Diffusion Model Analysis**: FPS (Frames Per Second) and BPS (Blocks Per Second) metrics
- **Model Analysis**: Parameter count and computational complexity
- **Block Computation Analysis**: Performance metrics for each video block

The enhanced profiling system provides real-time performance insights that are particularly valuable for optimizing real-time video generation.