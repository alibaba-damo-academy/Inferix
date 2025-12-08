# Self-Forcing Inference with Video Streaming

Self-Forcing video generation model inference example with real-time video streaming support.

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

## Video Streaming Support

Inferix supports two streaming protocols for real-time video generation preview:

### 🎯 WebRTC (Recommended)

**Why WebRTC?**
- ✅ **Easier to Use**: No external server required, works out of the box
- ✅ **Better Ecosystem**: Rapidly growing community support and development
- ✅ **Native Web Integration**: Seamlessly integrates with Gradio and other WebUI frameworks
- ✅ **Lower Latency**: Direct peer-to-peer connection for real-time streaming
- ✅ **Built-in UI**: Automatic web interface at `http://localhost:8000`

**Installation**:
```bash
pip install fastrtc
```

**Basic Usage**:
```bash
export PYTHONPATH=`pwd`:$PYTHONPATH
python example/self_forcing/run_self_forcing.py \
    --config_path example/self_forcing/configs/self_forcing_dmd.yaml \
    --output_folder example/self_forcing/outputs \
    --checkpoint_path ./weights/self_forcing/checkpoints/self_forcing_dmd.pt \
    --prompt "A cat dancing on the moon; A robot walking in a forest" \
    --use_ema \
    --enable_webrtc
```

**Access**: Open `http://localhost:8000` in your browser to view the live stream.

### 📡 RTMP (Alternative)

RTMP is also supported for compatibility with existing streaming infrastructure.

**Setup**: You need an RTMP server such as SRS (Simple Realtime Server).

**Quick Start with SRS**:
```bash
# Using Docker (recommended)
docker run -d -p 1935:1935 -p 8080:8080 ossrs/srs:5
```
### RTMP Streaming Setup
To use --rtmp_url, you need an RTMP server such as SRS (Simple Realtime Server) or another RTMP-compatible streaming service.

We recommend installing SRS via Docker or binary. See:
https://ossrs.net/lts/en-us/docs/v5/doc/getting-started

If those are not available, you can build SRS from source:
```
git clone -b develop https://github.com/ossrs/srs.git
apt-get install tcl
cd srs/trunk
./configure
make
```
Run SRS:

./objs/srs -c conf/srs.conf

- Default RTMP port: 1935
- Default RTMP ingest URL: rtmp://localhost:1935/live/livestream
- Playback URL: http://localhost:8080/

For detailed installation, see: https://ossrs.net/lts/en-us/docs/v5/doc/getting-started

**Usage**:
```bash
export PYTHONPATH=`pwd`:$PYTHONPATH
python example/self_forcing/run_self_forcing.py \
    --config_path example/self_forcing/configs/self_forcing_dmd.yaml \
    --output_folder example/self_forcing/outputs \
    --checkpoint_path ./weights/self_forcing/checkpoints/self_forcing_dmd.pt \
    --prompt "A cat dancing on the moon; A robot walking in a forest" \
    --use_ema \
    --rtmp_url rtmp://localhost:1935/live/livestream \
    --rtmp_fps 16
```

**Playback**: Access `http://localhost:8080/` to view the stream.

## Inference Examples

### Single GPU Text-to-Video

```bash
export PYTHONPATH=`pwd`:$PYTHONPATH
python example/self_forcing/run_self_forcing.py \
    --config_path example/self_forcing/configs/self_forcing_dmd.yaml \
    --output_folder example/self_forcing/outputs \
    --checkpoint_path ./weights/self_forcing/checkpoints/self_forcing_dmd.pt \
    --prompt "A cat dancing on the moon; A robot walking in a forest" \
    --use_ema \
    --enable_webrtc
```

### Multi-GPU Distributed Inference

```bash
export PYTHONPATH=`pwd`:$PYTHONPATH
torchrun --nnodes=1 --nproc-per-node=2 \
    example/self_forcing/run_self_forcing.py \
    --config_path example/self_forcing/configs/self_forcing_dmd.yaml \
    --output_folder example/self_forcing/outputs \
    --checkpoint_path ./weights/self_forcing/checkpoints/self_forcing_dmd.pt \
    --prompt "A cat dancing on the moon; A robot walking in a forest" \
    --use_ema \
    --enable_webrtc \
    --ulysses_size=1 --ring_size=2
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
- `--enable_webrtc`: Enable WebRTC streaming (recommended)
- `--rtmp_url`: RTMP streaming URL for live streaming (e.g., `rtmp://localhost:1935/live/livestream`)
- `--rtmp_fps`: RTMP streaming frame rate (default: 16)

### Configuration File

Use the `example/self_forcing/configs/self_forcing_dmd.yaml` configuration file, which contains detailed parameter settings for the Self-Forcing model.

Key configuration parameters:
- `denoising_step_list`: Denoising steps for the semi-autoregressive process
- `guidance_scale`: Classifier-free guidance scale
- `num_frame_per_block`: Number of frames per generation block
- `timestep_shift`: Time step shift parameter
- `warp_denoising_step`: Whether to warp denoising steps

## Streaming Protocol Comparison

| Feature | WebRTC | RTMP |
|---------|--------|------|
| Setup Complexity | ⭐⭐⭐⭐⭐ No server needed | ⭐⭐⭐ Requires SRS/nginx |
| Latency | ⭐⭐⭐⭐⭐ Ultra-low | ⭐⭐⭐ Low |
| Browser Support | ⭐⭐⭐⭐⭐ Native | ⭐⭐ Requires player |
| Gradio Integration | ⭐⭐⭐⭐⭐ Native | ⭐⭐ Custom needed |
| Community Growth | ⭐⭐⭐⭐⭐ Rapidly growing | ⭐⭐⭐ Stable/mature |
| Use Case | Interactive demos, WebUI | Production streaming |

**Recommendation**: Use **WebRTC** for development, demos, and interactive applications. Use **RTMP** when integrating with existing streaming infrastructure.