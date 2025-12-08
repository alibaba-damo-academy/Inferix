# Quantize with DAX

This guide demonstrates how to utilize [DAX](https://github.com/RiseAI-Sys/DAX) to enable quantized inference in Inferix.

## 🎯 Key Features

Inferix supports quantized inference with DAX:

- **Two 8-bit precision**: Supporting both INT8 Quantization and FP8 quantization, which is friendly for different GPU architecture.
- **Two quantization granularity**: Supporting two quantization granularity (Per-tensor activation, per-tensor weight / Per-token activation, per-channel weight), which can satisfy different speed-quality requirements.
- **Mixed precision**: Supporting keep some modules in high-precision, such as `time_embedder`. These modules consume little time but are important for generation quality.

## ⚙️ Usage

### 1. Install DAX

Clone DAX to `3rd_party` directory and install:

```bash
# From project root directory
mkdir -p 3rd_party
git clone https://github.com/RiseAI-Sys/DAX.git 3rd_party/DAX
cd 3rd_party/DAX && pip install -e .
```

### 2. Integrate DAX quantization in your pipeline

You can define a function to quantize your model. For example, if you want to utilize `per_token_act_per_channel_weight` quantization for your model, and keep `condition_embedder` and `proj_out` in high-precision, you can define the quantization function like this:

```python
from dax.quant.quantization.qconfig import get_dynamic_fp8_per_token_act_per_channel_weight_qconfig
from dax.quant.quantization import quantize_dynamic

def quantize_transformer(transformer):
    qconfig_dict = {
        "": get_dynamic_fp8_per_token_act_per_channel_weight_qconfig(),
        "condition_embedder": None,
        "proj_out": None,
    }
    quantize_dynamic(transformer, qconfig_dict)
```

Then you can apply this quantization function to your model:

```python
...
# Initialize the inference pipeline
pipeline = CausVidPipeline(
  config_path=args.config_path,
  default_config_path=args.default_config_path,
  wan_base_model_path=args.wan_base_model_path,
  enable_kv_offload=args.enable_kv_offload,
  parallel_config = parallel_config
)

...

# Quantize the transformer model
quantize_transformer(pipeline.pipeline.generator.model)

```

## 🚀 Quick Start

### CausVid Quantized Inference

```bash
# Run with default settings
./example/quantization/causvid_quantized.sh

# Or customize parameters
CHECKPOINT_FOLDER=./weights/causvid \
PROMPT="A beautiful sunset over the ocean" \
./example/quantization/causvid_quantized.sh
```

See `run_causvid_quantized.py` for implementation details.

### Self-Forcing Quantized Inference

```bash
# Run with default settings (FP8 quantization)
./example/quantization/self_forcing_quantized.sh

# Use INT8 quantization instead
QUANT_TYPE=int8 ./example/quantization/self_forcing_quantized.sh

# Customize parameters
CHECKPOINT_PATH=./weights/self_forcing/checkpoints/self_forcing_dmd.pt \
PROMPT="A cat dancing on the moon" \
NUM_OUTPUT_FRAMES=21 \
./example/quantization/self_forcing_quantized.sh
```

**With WebRTC streaming:**

```bash
python example/quantization/run_self_forcing_quantized.py \
    --config_path example/self_forcing/configs/self_forcing_dmd.yaml \
    --checkpoint_path ./weights/self_forcing/checkpoints/self_forcing_dmd.pt \
    --prompt "A cat dancing" \
    --output_folder outputs \
    --quant_type int8 \
    --enable_webrtc \
    --use_ema
```

## 📊 Memory Savings

| Model | FP32/BF16 | FP8 Quantized | Memory Saved |
|-------|-----------|---------------|-------------|
| CausVid (1.3B) | ~5.2 GB | ~2.6 GB | ~50% |
| Self-Forcing (1.3B) | ~5.2 GB | ~2.6 GB | ~50% |

*Note: Actual savings may vary based on batch size and sequence length.*
