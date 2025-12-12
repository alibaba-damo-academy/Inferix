# Inferix Model Integration Guide

This guide provides detailed instructions on how to integrate new semi-autoregressive models with the Inferix framework.

## Overview

Inferix is designed with extensibility in mind. The framework uses a pipeline-based architecture where each model implements a standardized interface while maintaining the flexibility to handle model-specific logic.

## Core Components

### 1. Pipeline Architecture

All models in Inferix must implement a pipeline class that inherits from [AbstractInferencePipeline](file:///Users/tangjiasheng/Project/Inferix/inferix/pipeline/base_pipeline.py#L9-L236). This abstract base class defines the standard interface that all pipelines must implement.

Key methods to implement:
- `load_checkpoint()`: Load model weights
- `run_text_to_video()`: Text-to-video generation
- `run_image_to_video()`: Image-to-video generation
- `_initialize_pipeline()`: Custom initialization logic

### 2. Configuration System

Inferix uses a flexible configuration system based on YAML/JSON files. Each model should provide configuration files that define model-specific parameters.

### 3. Distributed Inference Support

The framework provides built-in support for distributed inference, which can be leveraged by new models through the parallel configuration system.

## Integration Steps

### Step 1: Create Model Directory Structure

Create a directory for your model under `inferix/models/`:

```
inferix/
└── models/
    └── your_model_name/
        ├── __init__.py
        ├── model.py              # Model architecture implementation
        ├── config.py             # Model-specific configuration handling
        └── utils.py              # Utility functions (optional)
```

### Step 2: Implement Model Architecture

Implement your model architecture in `model.py`. This should include:
- Model definition
- Forward pass logic
- Any model-specific components

### Step 3: Create Pipeline Class

Create a pipeline class in `inferix/pipeline/your_model_name/pipeline.py`:

```python
import torch
from typing import List, Optional, Any
from omegaconf import OmegaConf

from inferix.pipeline.base_pipeline import AbstractInferencePipeline
from inferix.models.your_model_name.model import YourModel
from inferix.models.your_model_name.config import YourModelConfig

class YourModelPipeline(AbstractInferencePipeline):
    """Your model's inference pipeline"""
    
    def __init__(self, config_path: str, **kwargs):
        """
        Initialize the pipeline
        
        Args:
            config_path: Path to the configuration file
            **kwargs: Additional arguments (profiling_config, etc.)
        """
        # Load configuration
        config = OmegaConf.load(config_path)
        super().__init__(config, kwargs.get('profiling_config'))
        
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        
        # Initialize pipeline components
        self._initialize_pipeline()
        
    def _initialize_pipeline(self):
        """Initialize model components"""
        # Initialize your model
        self.model = YourModel(self.config)
        self.model.to(device=self.device, dtype=torch.bfloat16)
        
    def load_checkpoint(self, checkpoint_path: str, **kwargs) -> None:
        """
        Load model checkpoint weights
        
        Args:
            checkpoint_path: Path to the checkpoint file
            **kwargs: Additional arguments
        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        
    def run_text_to_video(self, prompts: List[str], **kwargs) -> Any:
        """
        Execute text-to-video inference logic
        
        Args:
            prompts: List of text prompts
            **kwargs: Additional arguments (output_folder, etc.)
            
        Returns:
            Generated video result
        """
        # Implement your text-to-video logic here
        pass
        
    def run_image_to_video(self, prompts: List[str], image_path: str, **kwargs) -> Any:
        """
        Execute image-to-video inference logic
        
        Args:
            prompts: List of text prompts
            image_path: Path to input image
            **kwargs: Additional arguments
            
        Returns:
            Generated video result
        """
        # Implement your image-to-video logic here
        pass
```

### Step 4: Add Configuration Files

Create configuration files in `example/your_model_name/configs/`:

```yaml
# example/your_model_name/configs/your_model_config.yaml
model_params:
  hidden_size: 2048
  num_layers: 24
  num_attention_heads: 16
  # ... other model parameters

generation_params:
  guidance_scale: 3.0
  num_inference_steps: 50
  # ... other generation parameters

runtime_params:
  seed: 42
  # ... other runtime parameters
```

### Step 5: Create Example Scripts

Create example scripts in `example/your_model_name/`:

1. `run_your_model.py`: Main execution script
2. `your_model.sh`: Shell script for easy execution
3. `README.md`: Detailed usage instructions

Example `run_your_model.py`:

```python
import argparse
import torch
import os

from inferix.pipeline.your_model_name.pipeline import YourModelPipeline

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Your Model Pipeline")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Initialize pipeline
    pipeline = YourModelPipeline(config_path=args.config_path)
    
    # Load checkpoint
    pipeline.load_checkpoint(args.checkpoint_path)
    
    # Run inference
    pipeline.run_text_to_video(prompts=[args.prompt], output_folder=args.output_folder)
    
    print("✅ Video generation completed.")

if __name__ == "__main__":
    main()
```

### Step 6: Add Documentation

Create a comprehensive README.md in your model's example directory with:
- Installation instructions
- Weight downloading instructions
- Usage examples
- Parameter descriptions
- Troubleshooting tips

### Step 7: Testing

Add unit tests for your pipeline in the `tests/` directory:

```
tests/
├── unit/
│   └── test_your_model.py
└── configs/
    └── your_model_test_config.yaml
```

## Best Practices

### 1. Profiling Integration

Leverage Inferix's built-in profiling capabilities by using the profiling decorators for high-level pipeline methods:

```python
from inferix.profiling.decorators import profile_method, profile_session

class YourModelPipeline(AbstractInferencePipeline):
    @profile_method("model_initialization")
    def _initialize_pipeline(self):
        # Initialization logic
        pass
        
    @profile_session("video_generation", {'mode': 'text_to_video'})
    def run_text_to_video(self, prompts: List[str], **kwargs) -> Any:
        # Generation logic
        pass
```

For diffusion model specific profiling, see [EXTENDING_PROFILING.md](inferix/profiling/EXTENDING_PROFILING.md) for detailed integration examples.

### 2. Error Handling

Implement proper error handling and validation:

```python
def run_text_to_video(self, prompts: List[str], **kwargs) -> Any:
    if not prompts:
        raise ValueError("At least one prompt must be provided")
    
    # Validate other parameters
    output_folder = kwargs.get('output_folder')
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    
    # Implementation
```

### 3. Memory Management

Use Inferix's memory utilities for efficient GPU memory management:

```python
from inferix.core.memory.utils import gpu as get_gpu, DynamicSwapInstaller

def setup_devices(self, low_memory: bool = False):
    """Setup devices and memory management"""
    gpu = get_gpu()
        
    if low_memory:
        DynamicSwapInstaller.install_model(self.model, device=gpu)
    else:
        self.model.to(device=gpu)
```

### 4. Distributed Inference

Support distributed inference by integrating with Inferix's parallel configuration:

```python
def __init__(self, config_path: str, parallel_config: Optional[ParallelConfig] = None, **kwargs):
    config = OmegaConf.load(config_path)
    super().__init__(config, kwargs.get('profiling_config'))
    
    self.parallel_config = parallel_config or ParallelConfig()
    self.device = torch.device(f"cuda:{self.parallel_config.local_rank}")
```

## Advanced Features

### 1. Custom Configuration Parsing

Implement custom configuration handling in your model's config module:

```python
# inferix/models/your_model_name/config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class YourModelConfig:
    hidden_size: int = 2048
    num_layers: int = 24
    guidance_scale: float = 3.0
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(**config_dict)
```

### 2. Streaming Support

Implement video streaming capabilities using Inferix's streaming utilities:

```python
from inferix.core.media import create_streaming_backend

def _save_and_stream_video(self, video: torch.Tensor, backend: str = "rtmp", **kwargs):
    """Save and optionally stream video.
    
    Args:
        video: Video tensor to stream
        backend: Streaming backend ("gradio", "webrtc", "rtmp")
        **kwargs: Backend-specific parameters (e.g., rtmp_url for RTMP)
    """
    streamer = create_streaming_backend(backend)
    if streamer.connect(**kwargs):
        streamer.stream_batch(video)
        streamer.disconnect()
```

**Example usage**:

```python
# Gradio streaming (default, best for development)
self._save_and_stream_video(video, backend="gradio", width=832, height=480, fps=16)

# RTMP streaming (production)
self._save_and_stream_video(video, backend="rtmp", rtmp_url="rtmp://localhost/live/stream")
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all new modules have `__init__.py` files
2. **Device Placement**: Always move models and tensors to the correct device
3. **Memory Issues**: Use `low_memory` mode for systems with limited VRAM
4. **Distributed Setup**: Verify world size matches parallel configuration

### Debugging Tips

1. Use Inferix's profiling to identify performance bottlenecks
2. Enable verbose logging during development
3. Test with small configurations before scaling up

## Conclusion

By following this guide, you can successfully integrate your semi-autoregressive model with the Inferix framework. The modular design and standardized interfaces make it straightforward to add new models while maintaining compatibility with existing features like distributed inference and performance profiling.