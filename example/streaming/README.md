# Streaming Video Generation with Inferix

This guide covers **progressive streaming** (block-wise generation) for real-time video generation.

**Streaming Backends** (Priority: Gradio > WebRTC > RTMP):
- **Gradio** (Default) - Best for development and interactive demos
- **WebRTC** (Optional) - For real-time P2P communication
- **RTMP** (Production) - For live streaming to CDN

**GitHub Repository**: [Self-Forcing](https://github.com/guandeh17/Self-Forcing)

## Table of Contents

1. [Quick Start](#quick-start)
2. [Streaming Backends](#streaming-backends)
3. [Architecture: Block vs Segment](#architecture-block-vs-segment)
4. [Progressive Streaming API](#progressive-streaming-api)
5. [Examples](#examples)

---

## Quick Start

### Gradio Streaming (Default, Recommended)

**Use Case**: Development, demos, interactive testing.

```bash
export PYTHONPATH=`pwd`:$PYTHONPATH
python example/streaming/run_progressive_streaming.py \
    --config_path example/self_forcing/configs/self_forcing_dmd.yaml \
    --checkpoint_path ./weights/self_forcing/checkpoints/self_forcing_dmd.pt \
    --prompt "A cat walking in a garden" \
    --num_segments 1
```

**Access**: Open `http://localhost:8000` in your browser to see real-time generation.

### RTMP Streaming (Production)

**Use Case**: Live streaming to servers/CDN.

```bash
export PYTHONPATH=`pwd`:$PYTHONPATH
python example/streaming/run_progressive_streaming.py \
    --config_path example/self_forcing/configs/self_forcing_dmd.yaml \
    --checkpoint_path ./weights/self_forcing/checkpoints/self_forcing_dmd.pt \
    --prompt "A cat walking" \
    --streaming_backend rtmp \
    --rtmp_url rtmp://localhost:1935/live/stream
```

---

## Streaming Backends

### Backend Comparison

| Backend | Latency | Use Case | Features |
|---------|---------|----------|----------|
| **Gradio** | 1-2s | Development | Auto-refresh UI, loop playback, easy debugging |
| **WebRTC** | <100ms | P2P calls | Low latency, browser-to-browser |
| **RTMP** | 2-5s | Production | CDN compatible, reliable |

### Usage

```python
from inferix.core.media import create_streaming_backend

# Create backend (gradio/webrtc/rtmp)
streamer = create_streaming_backend("gradio")

# Connect
streamer.connect(width=832, height=480, fps=16, port=8000)

# Stream frames
streamer.stream_batch(frames)  # Tensor [T, H, W, C] uint8

# Disconnect
streamer.disconnect()
```

---

## Architecture: Block vs Segment

### Terminology

#### BLOCK
**Definition**: Model-specific atomic generation unit.

- **Size**: Self-Forcing = 3 frames (`num_frame_per_block=3`)
- **Generation**: ~500ms per block (hardware-dependent)
- **Purpose**: Smallest unit for autoregressive continuation with KV cache
- **Level**: Internal model implementation detail

#### SEGMENT
**Definition**: Framework-level complete generation cycle.

- **Size**: 21 frames (default) = 7 blocks × 3 frames/block
- **Generation**: ~3.5s per segment
- **Purpose**: Complete generation cycle with memory cleanup
- **Level**: User-facing API parameter

### Streaming Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│  FRAMEWORK LEVEL (run_streaming_generation)                 │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Segment 0  │  │  Segment 1  │  │  Segment 2  │  ...    │
│  │  21 frames  │  │  21 frames  │  │  21 frames  │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│    Memory cleanup   Memory cleanup   Memory cleanup         │
└─────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│  MODEL LEVEL (_generate_segment_with_streaming)             │
│                                                              │
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐                │
│  │ B0│ │ B1│ │ B2│ │ B3│ │ B4│ │ B5│ │ B6│  (7 blocks)    │
│  │ 3f│ │ 3f│ │ 3f│ │ 3f│ │ 3f│ │ 3f│ │ 3f│                │
│  └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘                │
│    │     │     │     │     │     │     │                    │
│    ▼     ▼     ▼     ▼     ▼     ▼     ▼                    │
│  Decode Decode Decode Decode Decode Decode Decode           │
│    │     │     │     │     │     │     │                    │
│    ▼     ▼     ▼     ▼     ▼     ▼     ▼                    │
│  Stream Stream Stream Stream Stream Stream Stream           │
└─────────────────────────────────────────────────────────────┘
```

**Key Points**:
- **Blocks** enable progressive streaming (see frames earlier)
- **Segments** enable memory management (avoid OOM for long videos)
- Both work together automatically in `run_streaming_generation()`

---

## Progressive Streaming API

### Usage Modes

#### Mode 1: Single-Segment Block-Wise Streaming

**Use Case**: Short video with real-time feedback.

```python
from inferix.pipeline.self_forcing.pipeline import SelfForcingPipeline
from inferix.core.media import create_streaming_backend

# Initialize pipeline
pipeline = SelfForcingPipeline(
    config_path="example/self_forcing/configs/self_forcing_dmd.yaml"
)
pipeline.load_checkpoint("./weights/self_forcing/checkpoints/self_forcing_dmd.pt")
pipeline.setup_devices()

# Initialize Gradio streaming (default)
streamer = create_streaming_backend("gradio")
streamer.connect(width=832, height=480, fps=16, port=8000)

# Generate with progressive streaming
pipeline.run_streaming_generation(
    prompts=['a cat walking'],
    stream_callback=streamer.stream_batch,
    num_segments=1,        # Single segment
    segment_length=21,     # 7 blocks × 3 frames/block
    num_samples=1
)
```

**Timeline** (21-frame generation):
```
Time    Block   Frames      User Experience
----    -----   ------      ---------------
0.0s    Start   -           Generation begins
0.5s    0       [0,1,2]     ✅ User sees first 3 frames!
1.0s    1       [3,4,5]     ✅ 3 more frames appear
1.5s    2       [6,7,8]     ✅ 3 more frames appear
...
3.5s    6       [18,19,20]  ✅ Final 3 frames, complete!
```

**Benefit**: User sees content after **0.5s** instead of waiting **3.5s**!

#### Mode 2: Multi-Segment Long-Video Streaming

**Use Case**: Long video for WebRTC testing and demos.

```python
# Generate long video (10 segments = ~183 frames)
pipeline.run_streaming_generation(
    prompts=['a cat walking in a garden'],
    stream_callback=streamer.stream_batch,
    num_segments=10,       # 10 segments
    segment_length=21,     # 21 frames per segment
    overlap_frames=3,      # 3 frames overlap between segments
    num_samples=1,
    low_memory=True        # Enable memory optimization
)
```

**Segment Flow**:
```
Segment 0: Frames [0-20]       (21 frames) → cleanup
Segment 1: Frames [18-38]      (21 frames, overlap 3) → cleanup
                  ↑ overlap
Segment 2: Frames [36-56]      (21 frames, overlap 3) → cleanup
...
Segment 9: Frames [162-182]    (21 frames, overlap 3) → cleanup

Total unique frames: 10×21 - 9×3 = 183 frames
Total generation time: ~35 seconds
```

**Memory Advantage**: CUDA cache cleared after each segment, preventing OOM.

### API Reference

```python
pipeline.run_streaming_generation(
    prompts: List[str],                              # Text prompts
    stream_callback: Optional[Callable] = None,      # Streaming callback
    num_segments: int = 1,                           # Number of segments
    segment_length: int = 21,                        # Frames per segment
    overlap_frames: int = 3,                         # Overlap between segments
    **kwargs                                         # num_samples, low_memory, etc.
) -> torch.Tensor
```

**Parameters**:
- `num_segments`: 
  - `1` = short video with block-wise streaming
  - `10-20` = long video for WebRTC testing
- `segment_length`: Must be multiple of 3 (block size) for Self-Forcing
  - Recommended: 21, 24, 30
- `overlap_frames`: Overlap between segments for smooth transitions
  - Recommended: 3 (1 block)
- `stream_callback`: Callback receiving decoded frames
  - Signature: `callback(frames: torch.Tensor)`  
  - frames: `[T, H, W, C]`, `uint8`, range `[0, 255]`

**Callback Example**:
```python
def my_stream_callback(frames: torch.Tensor):
    """
    Called for each decoded block.
    
    Args:
        frames: [T, H, W, C], uint8, range [0, 255]
                T = 3 for Self-Forcing (block size)
    """
    # Send to WebRTC
    webrtc_streamer.stream_batch(frames)
    
    # Or save to disk
    for i, frame in enumerate(frames):
        save_image(frame, f"frame_{i}.png")
```

---

## Traditional Streaming

### WebRTC (Recommended)

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

---

## Traditional Streaming

Traditional streaming streams the complete video after generation finishes.

### WebRTC (Recommended)

**Why WebRTC?**
- ✅ **Easier to Use**: No external server required
- ✅ **Better Ecosystem**: Rapidly growing community
- ✅ **Native Web Integration**: Works with Gradio and WebUI frameworks
- ✅ **Lower Latency**: Direct peer-to-peer connection
- ✅ **Built-in UI**: Automatic interface at `http://localhost:8000`

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
    --use_ema
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
    --use_ema
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

### Configuration File

Use the `example/self_forcing/configs/self_forcing_dmd.yaml` configuration file, which contains detailed parameter settings for the Self-Forcing model.

Key configuration parameters:
- `denoising_step_list`: Denoising steps for the semi-autoregressive process
- `guidance_scale`: Classifier-free guidance scale
- `num_frame_per_block`: Number of frames per generation block
- `timestep_shift`: Time step shift parameter
- `warp_denoising_step`: Whether to warp denoising steps

## Streaming Backend Comparison

| Feature | Gradio | WebRTC (experimental) | RTMP |
|---------|--------|----------------------|------|
| Setup Complexity | ⭐⭐⭐⭐⭐ Zero config | ⭐⭐⭐⭐ Requires fastrtc | ⭐⭐⭐ Requires SRS/nginx |
| Latency | ⭐⭐⭐⭐ Low (~1-2s) | ⭐⭐⭐⭐⭐ Ultra-low (<100ms) | ⭐⭐⭐ Low (~2-5s) |
| Browser Support | ⭐⭐⭐⭐⭐ Native | ⭐⭐⭐⭐⭐ Native | ⭐⭐ Requires player |
| Stability | ⭐⭐⭐⭐⭐ Production-ready | ⭐⭐⭐ Experimental | ⭐⭐⭐⭐ Mature |
| Interactive UI | ⭐⭐⭐⭐⭐ Built-in | ⭐⭐⭐ Custom needed | ⭐⭐ Custom needed |
| Use Case | Development, demos, testing | Real-time P2P (future) | Production streaming |

**Recommendation**: Use **Gradio** (default) for development and interactive applications. Use **RTMP** for production streaming infrastructure.

---

## Examples

### Example 1: Progressive Streaming (Recommended)

See [`run_progressive_streaming.py`](./run_progressive_streaming.py) for a complete example.

**Run with Gradio backend (default, recommended)**:
```bash
export PYTHONPATH=`pwd`:$PYTHONPATH
python example/streaming/run_progressive_streaming.py \
    --config_path example/self_forcing/configs/self_forcing_dmd.yaml \
    --checkpoint_path ./weights/self_forcing/checkpoints/self_forcing_dmd.pt \
    --prompt "A cat walking" \
    --num_segments 5 \
    --segment_length 21 \
    --overlap_frames 3
    # --streaming_backend gradio (default, can be omitted)
```

**Run with WebRTC backend (experimental)**:
```bash
python example/streaming/run_progressive_streaming.py \
    --config_path example/self_forcing/configs/self_forcing_dmd.yaml \
    --checkpoint_path ./weights/self_forcing/checkpoints/self_forcing_dmd.pt \
    --prompt "A cat walking" \
    --num_segments 5 \
    --streaming_backend webrtc
```

**Run with RTMP backend (production)**:
```bash
python example/streaming/run_progressive_streaming.py \
    --config_path example/self_forcing/configs/self_forcing_dmd.yaml \
    --checkpoint_path ./weights/self_forcing/checkpoints/self_forcing_dmd.pt \
    --prompt "A cat walking" \
    --num_segments 5 \
    --streaming_backend rtmp \
    --rtmp_url rtmp://localhost:1935/live/stream
```

### Example 2: Basic Inference

For simple generation without streaming:

```bash
export PYTHONPATH=`pwd`:$PYTHONPATH
python example/self_forcing/run_self_forcing.py \
    --config_path example/self_forcing/configs/self_forcing_dmd.yaml \
    --checkpoint_path ./weights/self_forcing/checkpoints/self_forcing_dmd.pt \
    --prompt "A cat dancing" \
    --output_folder outputs
```

### Example 3: Gradio Integration in Code

```python
from inferix.pipeline.self_forcing.pipeline import SelfForcingPipeline
from inferix.core.media import create_streaming_backend

# Setup pipeline
pipeline = SelfForcingPipeline(
    config_path="example/self_forcing/configs/self_forcing_dmd.yaml"
)
pipeline.load_checkpoint("./weights/self_forcing/checkpoints/self_forcing_dmd.pt")
pipeline.setup_devices()

# Setup Gradio streaming
streamer = create_streaming_backend("gradio")
streamer.connect(width=832, height=480, fps=16)

# Progressive streaming
pipeline.run_streaming_generation(
    prompts=['a dog running'],
    stream_callback=streamer.stream_batch,
    num_segments=10,
    segment_length=21,
    overlap_frames=3
)

print("Open http://localhost:8000 to view stream")
```

---

## Performance Benchmarking

### Overview

Performance testing leverages Inferix's built-in profiling module to collect detailed metrics.
The profiling system automatically tracks:
- Block-level computation and decoding times
- Diffusion step performance
- GPU memory usage and utilization
- Overall throughput (FPS)

### Running Benchmarks

To collect accurate performance metrics for your GPU:

```bash
export PYTHONPATH=`pwd`:$PYTHONPATH

# Step 1: Run streaming generation with profiling enabled
python example/streaming/run_progressive_streaming.py \
    --config_path example/self_forcing/configs/self_forcing_dmd.yaml \
    --default_config_path example/self_forcing/configs/default_config.yaml \
    --checkpoint_path ./weights/self_forcing/checkpoints/self_forcing_dmd.pt \
    --prompt "A cat walking" \
    --num_segments 10 \
    --enable_profiling \
    --profile_output_dir ./profiling_results \
    --use_ema

# Step 2: Extract metrics for documentation
python example/streaming/extract_streaming_metrics.py \
    --profile_dir ./profiling_results \
    --output_file benchmark_results.json \
    --print_markdown
```

**Output**: 
- HTML/JSON profiling reports in `./profiling_results/`
- Extracted metrics in `benchmark_results.json`
- Markdown-formatted results (if `--print_markdown` is used)

### Metrics Collected

The profiling system captures:

**Block-level Performance**:
- Diffusion step timing (ms per step)
- Block computation time (ms per block)
- Block FPS and Blocks Per Second (BPS)
- Memory usage per block

**Segment-level Performance**:
- Time per segment (seconds)
- Number of segments processed

**Overall Performance**:
- Total generation time
- Throughput (FPS)
- Peak GPU memory usage
- GPU utilization percentage

### Benchmark Results

> **Note**: Run the benchmark commands above to generate results for your specific GPU.
> The profiling module will automatically collect all metrics.

**Your GPU**: [To be filled after running benchmark]

**Block-level**:
- Block size: 3 frames
- Diffusion step time: [Run benchmark] ms per step
- Block computation: [Run benchmark] ms per block
- Block FPS: [Run benchmark]
- Blocks Per Second: [Run benchmark]

**Segment-level** (21 frames):
- Blocks per segment: 7
- Time per segment: [Run benchmark] s

**Long video** (10 segments, ~210 frames):
- Total time: [Run benchmark] s
- Throughput: [Run benchmark] FPS
- Peak memory: [Run benchmark] MB
- GPU utilization: [Run benchmark]%

### Updating Documentation

After running the benchmark:

1. Check the profiling reports in `./profiling_results/`
2. Run `extract_streaming_metrics.py` with `--print_markdown`
3. Copy the formatted output to update "Benchmark Results" section above

Example extracted metrics:
```json
{
  "system_info": {
    "gpu_name": "NVIDIA GeForce RTX 4060",
    "gpu_memory_total": 16.0
  },
  "block_level": {
    "avg_step_time_ms": 50.5,
    "avg_block_time_ms": 450.2,
    "block_fps": 6.67,
    "bps": 2.22
  },
  "segment_level": {
    "avg_segment_time_s": 3.15
  },
  "overall": {
    "throughput_fps": 6.67,
    "peak_memory_mb": 8192,
    "avg_gpu_utilization": 92.5
  }
}
```

---

## Comparison: Progressive vs Traditional

| Feature | Progressive Streaming | Traditional Streaming |
|---------|----------------------|----------------------|
| **First Frame Latency** | ~0.5s (first block) | ~3.5s (full video) |
| **Memory Management** | ✅ Automatic cleanup | ❌ Manual control |
| **Long Videos** | ✅ Unlimited with segments | ❌ OOM risk |
| **User Experience** | ✅ Progressive feedback | ❌ Wait then play |
| **Streaming Support** | ✅ Real-time streaming | ✅ Post-gen streaming |
| **Use Case** | Interactive demos, testing | Quick generation |

---

## FAQ

### Q: What's the difference between block and segment?

**A**: 
- **Block**: Model's 3-frame generation unit (internal detail)
- **Segment**: Framework's 21-frame cycle (user parameter)
- A segment contains 7 blocks

### Q: When should I use progressive streaming?

**A**: Use progressive streaming when:
- Testing streaming with long videos
- Need real-time user feedback
- Generating videos longer than GPU memory allows
- Building interactive applications

### Q: Can I customize segment_length?

**A**: Yes, but must be multiple of block size:
- Self-Forcing: multiples of 3 (e.g., 21, 24, 30)
- Will be validated at runtime

### Q: How do I calculate total frames with overlap?

**A**: 
```
Total frames = num_segments × segment_length - (num_segments - 1) × overlap_frames

Example: 10 × 21 - 9 × 3 = 183 frames
```

---

## Troubleshooting

### "segment_length must be multiple of 3"
**Solution**: Use 21, 24, 30, etc. for Self-Forcing.

### Gradio/WebRTC not connecting
**Solution**: 
1. Check port 8000 is not in use
2. For WebRTC backend: Install `fastrtc`: `pip install fastrtc`
3. Check firewall settings
4. For WSL: Use the WSL IP address shown in terminal output

### Out of memory with long videos
**Solution**: 
1. Use progressive streaming with `num_segments > 1`
2. Enable `low_memory=True`
3. Reduce `segment_length`

---

## Prerequisites (Detailed)

### Download Model Weights

1. **Wan2.1-T2V-1.3B Base Model**:
   ```bash
   huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
       --local-dir-use-symlinks False \
       --local-dir ./weights/Wan2.1-T2V-1.3B
   ```

2. **Self-Forcing Checkpoint**:
   ```bash
   huggingface-cli download gdhe17/Self-Forcing \
       checkpoints/self_forcing_dmd.pt \
       --local-dir ./weights/self_forcing
   ```