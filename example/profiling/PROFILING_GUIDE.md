# Self Forcing Model Profiling Guide

This guide provides detailed instructions on how to use Inferix's profiling module for performance analysis and monitoring in the Self Forcing model.

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Configuring Profiling](#configuring-profiling)
4. [Running Profiling Tests](#running-profiling-tests)
5. [Analyzing Profiling Results](#analyzing-profiling-results)
6. [Aggregating Distributed Reports](#aggregating-distributed-reports)
7. [Advanced Configuration Options](#advanced-configuration-options)
8. [Using Profiling Decorators](#using-profiling-decorators)
9. [Troubleshooting](#troubleshooting)

## Introduction

The Inferix Profiling module provides comprehensive performance monitoring and analysis capabilities for video generation pipelines. It can:
- Monitor GPU memory usage, utilization, and temperature in real-time
- Monitor CPU usage and system memory
- Provide real-time performance metrics during video generation
- Generate detailed HTML and JSON format performance reports
- Identify performance bottlenecks and provide optimization suggestions
- Aggregate reports from distributed training runs

## Prerequisites

### System Requirements

Ensure the following dependencies are installed:

```bash
# Basic dependencies
pip install pynvml psutil

# For GPU monitoring (recommended)
pip install pynvml
```

### Model Weights

Ensure the Self Forcing model weights have been downloaded and placed in the correct location:
```bash
# Default weights path
./weights/self_forcing/checkpoints/self_forcing_dmd.pt
```

## Configuring Profiling

### Basic Configuration File

Configuration file location: `example/self_forcing/configs/profiling_config.yaml`

```yaml
profiling:
  enabled: true                    # Enable profiling system
  real_time_display: true         # Display real-time metrics
  display_interval: 3.0           # Update display every 3 seconds
  
  # GPU monitoring
  gpu_monitor_interval: 0.5       # Sample GPU metrics every 0.5 seconds
  monitor_gpu_memory: true        # Monitor GPU memory usage
  monitor_gpu_utilization: true   # Monitor GPU utilization
  monitor_gpu_temperature: true   # Monitor GPU temperature
  monitor_gpu_power: true         # Monitor GPU power consumption
  
  # CPU monitoring
  cpu_monitor_interval: 1.0       # Sample CPU metrics every 1 second
  monitor_cpu_usage: true         # Monitor CPU usage
  monitor_system_memory: true     # Monitor system memory
  
  # Report generation
  generate_final_report: true     # Generate final report
  report_format: "both"           # Generate both HTML and JSON formats
  output_dir: "./example/self_forcing/profiling_reports"  # Report output directory
  
  # Advanced options
  max_data_points: 10000          # Limit memory usage
  profile_inference_steps: true   # Analyze each inference step
  profile_vae_decode: true        # Analyze VAE decoding
  profile_text_encoding: true     # Analyze text encoding
  
  # Session metadata
  session_tags:
    model: "self_forcing"
    experiment: "performance_analysis"
```

## Running Profiling Tests

### Using the Dedicated Profiling Script

```bash
# Navigate to project root directory
cd /path/to/Inferix

```

### Manual Execution

```bash
# Set environment variables (optional)
export CHECKPOINT_PATH=/path/to/your/checkpoint.pt
export PROMPT="A beautiful sunset over the ocean"

# Run inference with profiling enabled
torchrun --nnodes=1 --nproc-per-node=1 \
    example/self_forcing/run_self_forcing.py \
    --config_path "example/self_forcing/configs/self_forcing_dmd.yaml" \
    --default_config_path "example/self_forcing/configs/default_config.yaml" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --prompt "$PROMPT" \
    --output_folder "example/self_forcing/profiling_outputs" \
    --use_ema \
    --enable_profiling \
    --profiling_config "example/self_forcing/configs/profiling_config.yaml"
```

### Using the Profiling Script

```bash
# Navigate to project root directory
cd /path/to/Inferix

# Run the dedicated profiling script
./example/profiling/run_distributed_self_forcing.sh
```

### Parameter Description

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--enable_profiling` | Enable profiling functionality | false |
| `--profiling_config` | Profiling configuration file path | None |
| `--config_path` | Model configuration file path | Required |
| `--checkpoint_path` | Model weight file path | Required |
| `--prompt` | Text prompt | Required |
| `--output_folder` | Output folder | Required |

## Analyzing Profiling Results

### Real-time Monitoring Output

During video generation, you will see real-time metrics similar to:

```
=== Profiling Metrics (session_abc123) ===
GPU: Memory 8420MB / 24564MB (34.3%), Util 85.2%, Temp 72.1°C
CPU: 45.3%, Memory 12840MB / 32768MB (39.2%)
```

### Generated Reports

After completion, two formats of reports will be generated in the specified output directory:

1. **HTML Report**: Interactive visual report containing:
   - Performance summary
   - Timeline visualization
   - Phase analysis
   - Optimization suggestions
   - System information

2. **JSON Report**: Machine-readable report containing:
   - Detailed metrics timeline
   - Phase time breakdown
   - Performance statistics
   - Suggestion data

## Aggregating Distributed Reports

When running distributed training with multiple GPUs, each rank generates its own profiling reports. 
The `aggregate_profiling_reports.py` tool can combine these into a single comprehensive report.

### Using the Aggregation Tool

```bash
# Navigate to project root directory
cd /path/to/Inferix

# Aggregate reports from all ranks
python example/profiling/aggregate_profiling_reports.py \
    --reports_dir ./example/self_forcing/profiling_reports \
    --output ./aggregated_report.json
```

### Understanding Aggregated Reports

The aggregated report provides:
- **Summary Statistics**: Min/max/average metrics across all ranks
- **Rank Comparison**: Performance comparison between different ranks
- **Bottleneck Identification**: System-wide performance bottlenecks
- **Resource Utilization**: Overall resource usage efficiency

Example output:
```
=== Aggregation Summary ===
Total reports processed: 8
Ranks: 8

=== GPU Metrics (Aggregated) ===
GPU Memory - Min: 7200.0MB, Max: 9800.0MB, Avg: 8420.0MB
GPU Utilization - Min: 75.2%, Max: 92.1%, Avg: 85.2%
```

## Advanced Configuration Options

### Development Mode Configuration

Suitable for detailed debugging and analysis:

```yaml
profiling:
  enabled: true
  real_time_display: true
  display_interval: 1.0          # More frequent updates
  
  # High-frequency monitoring
  gpu_monitor_interval: 0.1      # Higher frequency sampling
  cpu_monitor_interval: 0.5
  
  # Save all data
  generate_final_report: true
  report_format: "both"
  save_raw_data: true            # Save raw monitoring data
  output_dir: "./debug_profiling"
  
  # Analyze everything
  profile_inference_steps: true
  profile_vae_decode: true
  profile_text_encoding: true
  
  session_tags:
    mode: "debug"
    detailed_analysis: true
```

### Production Mode Configuration

Suitable for production environments with minimal performance overhead:

```yaml
profiling:
  enabled: true
  real_time_display: false       # Disable real-time display to reduce overhead
  
  # Reduce monitoring frequency
  gpu_monitor_interval: 1.0
  cpu_monitor_interval: 2.0
  
  # Monitor only key metrics
  monitor_gpu_memory: true
  monitor_gpu_utilization: true
  monitor_gpu_temperature: false
  monitor_gpu_power: false
  monitor_cpu_usage: false
  monitor_system_memory: true
  
  # Generate only final report
  generate_final_report: true
  report_format: "json"          # Only JSON format for automated processing
  save_raw_data: false
  output_dir: "./production_profiling"
  
  # Reduce data points to save memory
  max_data_points: 5000
  
  # Analyze only major phases
  profile_inference_steps: true
  profile_vae_decode: true
  profile_text_encoding: false
  
  session_tags:
    mode: "production"
    minimal_overhead: true
```

## Using Profiling Decorators

Inferix provides a set of decorators to simplify profiling integration in custom pipeline implementations. These decorators reduce boilerplate code and make profiling more maintainable.

### Available Decorators

1. **@profile_method**: Profiles a single method execution
2. **@profile_session**: Wraps a method in a complete profiling session
3. **@add_profiling_event**: Adds a custom event after method execution
4. **@profile_diffusion_step**: Profiles diffusion model steps
5. **@profile_model_parameters**: Profiles model parameters
6. **@profile_block_computation**: Profiles block computation in diffusion models

### Example Usage

```python
from inferix.profiling.decorators import profile_method, profile_session, add_profiling_event, profile_diffusion_step, profile_model_parameters, profile_block_computation

class CustomPipeline:
    @profile_method("model_initialization")
    def _initialize_model(self):
        # Model initialization code
        pass

    @profile_session("video_generation", {"mode": "t2v"})
    @add_profiling_event("generation_completed", lambda result, **kwargs: {
        "output_shape": list(result.shape) if result is not None else None
    })
    def run_text_to_video(self, prompts, **kwargs):
        # Video generation code
        result = self._generate_video(prompts, **kwargs)
        return result

    @profile_method("checkpoint_loading")
    @add_profiling_event("checkpoint_loaded", lambda _, checkpoint_path, **kwargs: {
        "checkpoint_path": checkpoint_path
    })
    def load_checkpoint(self, checkpoint_path, **kwargs):
        # Checkpoint loading code
        pass
    
    @profile_diffusion_step(lambda args, kwargs: {
        "step": args[0],
        "timestep": args[1],
        "block_size": args[2]
    })
    def diffusion_step(self, step, timestep, block_size):
        # Diffusion step implementation
        pass
    
    @profile_model_parameters(lambda args, kwargs: {
        "model_name": args[0],
        "parameters_count": args[1],
        "model_type": args[2]
    })
    def record_model_parameters(self, model_name, parameters_count, model_type):
        # Record model parameters
        pass
    
    @profile_block_computation(lambda args, kwargs: {
        "block_index": args[0],
        "block_size": args[1],
        "computation_time_ms": args[2],
        "memory_usage_mb": args[3]
    })
    def process_block(self, block_index, block_size, computation_time_ms, memory_usage_mb):
        # Block processing implementation
        pass
```

For more details on integration approaches, see [EXTENDING_PROFILING.md](../../inferix/profiling/EXTENDING_PROFILING.md).

### Benefits of Using Decorators

1. **Reduced Boilerplate**: Eliminates manual context manager usage
2. **Consistent Interface**: Standardized way to add profiling across all methods
3. **Automatic Cleanup**: Decorators handle session start/end automatically
4. **Flexible Metadata**: Easy to add custom metadata and events
5. **Conditional Execution**: Decorators automatically check if profiling is enabled

### Best Practices

1. **Use @profile_session for main entry points** like [run_text_to_video](../../../inferix/pipeline/self_forcing/pipeline.py) and [run_image_to_video](../../../inferix/pipeline/self_forcing/pipeline.py)
2. **Use @profile_method for internal methods** like initialization and setup functions
3. **Use @add_profiling_event for important milestones** like checkpoint loading or completion
4. **Provide meaningful stage names** that clearly describe what's being profiled
5. **Add relevant metadata** to make reports more informative

## Troubleshooting

### GPU Monitoring Not Working

```bash
# Install NVIDIA Management Library
pip install pynvml

# Check GPU availability
nvidia-smi
```

### High Memory Usage

```yaml
# Reduce data points
max_data_points: 5000

# Increase monitoring intervals
gpu_monitor_interval: 1.0
cpu_monitor_interval: 2.0
```

### Excessive Performance Overhead

```yaml
# Use production configuration
real_time_display: false
monitor_gpu_temperature: false
monitor_gpu_power: false
profile_text_encoding: false
```

### Common Issues

1. **"No module named 'pynvml'"**
   ```bash
   pip install pynvml
   ```

2. **"No module named 'psutil'"**
   ```bash
   pip install psutil
   ```

3. **GPU temperature monitoring not working**
   - Ensure using NVIDIA drivers that support NVML
   - Some GPUs may not support temperature monitoring

4. **Report generation failed**
   - Check output directory permissions
   - Ensure sufficient disk space

## Best Practices

1. **Start Simple**: Begin with basic profiling and add details as needed
2. **Focus on GPU**: GPU metrics are most important for video generation
3. **Analyze by Phases**: Break down the pipeline into logical phases for analysis
4. **Monitor Trends**: Observe performance pattern changes over time
5. **Iterative Optimization**: Analyze → Optimize → Re-analyze → Repeat
6. **Production Monitoring**: Maintain lightweight profiling in production environments
7. **Save Reports**: Archive reports for performance trend analysis
8. **Use Diffusion Analysis**: Leverage the enhanced diffusion model analysis for real-time performance optimization
9. **Monitor FPS and BPS**: Pay attention to Frames Per Second and Blocks Per Second metrics for streaming performance

## Support and Feedback

If you encounter issues or have improvement suggestions, please:
1. Check [GitHub Issues](https://github.com/your-repo/issues)
2. Submit a new issue describing the problem
3. Contribute code to improve the profiling module