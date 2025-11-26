# Quantize with DAX

This guide demonstrates how to utilize [DAX](https://github.com/RiseAI-Sys/DAX) to enable quantized inference in Inferix.

## 🎯 Key Features

Inferix supports quantized inference with DAX:

- **Two 8-bit precision**: Supporting both INT8 Quantization and FP8 quantization, which is friendly for different GPU architecture.
- **Two quantization granularity**: Supporting two quantization granularity (Per-tensor activation, per-tensor weight / Per-token activation, per-channel weight), which can satisfy different speed-quality requirements.
- **Mixed precision**: Supporting keep some modules in high-precision, such as `time_embedder`. These modules consume little time but are important for generation quality.

## ⚙️ Usage

### 1. Install DAX

First, you need to follow the instructions in [DAX](https://github.com/RiseAI-Sys/DAX) to install it. You can also download by following example:

```
git clone https://github.com/RiseAI-Sys/DAX.git
cd DAX
pip install -e .
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

We give an example in `run_causvid_quantized.py`, you can learn how to apply DAX quantization from it.  
