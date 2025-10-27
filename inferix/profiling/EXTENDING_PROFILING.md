# Extending Inferix Profiling Capabilities

This document describes how to extend Inferix's profiling capabilities to provide deeper insights into model performance, with a focus on diffusion model analysis.

## Overview

The Inferix profiling system provides detailed diffusion model analysis, including:

1. Diffusion step analysis
2. Block-wise semi-AR corresponding block-size analysis
3. Model parameter count/computation analysis
4. Multi-GPU scalability analysis
5. Real-time streaming performance analysis
6. FPS (Frames Per Second) and BPS (Blocks Per Second) performance metrics

## Diffusion Model Analysis Features

### DiffusionAnalyzer Class

The `DiffusionAnalyzer` class provides specialized analysis capabilities for diffusion models. This class is integrated into the existing `InferixProfiler` and can be used without additional configuration.

### Configuration Options

The following configuration options are available in `ProfilingConfig`:

```yaml
profiling:
  # ... other configuration options ...
  
  # Diffusion model specific analysis
  profile_diffusion_steps: true
  profile_block_computation: true
  profile_model_parameters: true
```

### Report Enhancements

The report generator includes the following sections:

- Diffusion model analysis (including FPS and BPS metrics)
- Model analysis
- Block computation analysis (including FPS and BPS metrics)

## Integration Approaches

Inferix supports two main approaches for integrating profiling capabilities:

### Decorator-Based Integration

The decorator-based approach is suitable for high-level pipeline methods and provides a clean, declarative way to add profiling. This approach is ideal for:

- High-level pipeline methods like `run_text_to_video` or `run_image_to_video`
- Initialization methods like `_initialize_pipeline`
- Methods that represent distinct pipeline stages

Example:

```python
from inferix.profiling.decorators import profile_method, profile_session

class VideoPipeline:
    @profile_session("video_generation", {"mode": "t2v"})
    def run_text_to_video(self, prompts):
        # Generation logic
        pass
    
    @profile_method("model_initialization")
    def _initialize_model(self):
        # Initialization logic
        pass
```

### Direct Code Integration

Direct code integration is necessary for capturing detailed performance metrics within complex pipeline implementations. This approach is required when:

1. **Fine-Grained Metrics Are Needed**: Capturing per-diffusion-step timing or block computation metrics
2. **Working with External Code**: Integrating with externally sourced pipelines (like open-source CausalInferencePipeline) where you cannot modify method signatures
3. **Access to Intermediate Values**: Recording performance data that requires access to intermediate calculations
4. **Complex Algorithm Profiling**: Profiling specific parts of complex algorithms where decorators would be insufficient

The reason we use direct code integration in the CausalInferencePipeline is primarily because:

1. **External Code Compatibility**: The CausalInferencePipeline is often sourced from external open-source projects, and we need to integrate profiling without modifying the core method signatures
2. **Detailed Diffusion Analysis**: We need to capture performance data at specific points within the diffusion process, which requires direct access to intermediate values
3. **Granular Performance Tracking**: Recording metrics for each diffusion step and block computation requires precise placement of profiling code within the algorithm

Example of direct integration in complex pipeline code:

```python
# In a complex pipeline's inference method
for block_index in range(num_blocks):
    # Start timing for block computation
    block_start_time = torch.cuda.Event(enable_timing=True)
    block_end_time = torch.cuda.Event(enable_timing=True)
    block_start_time.record()
    
    # Complex diffusion algorithm
    for step_index, timestep in enumerate(denoising_steps):
        # Record diffusion step metrics
        if profiler is not None:
            profiler.record_diffusion_step(
                step=step_index,
                timestep=timestep,
                block_size=block_size,
                computation_time_ms=step_time_ms  # Calculated separately
            )
    
    # End timing for block computation
    block_end_time.record()
    torch.cuda.synchronize()
    block_time_ms = block_start_time.elapsed_time(block_end_time)
    
    # Record block computation metrics
    if profiler is not None:
        profiler.record_block_computation(
            block_index=block_index,
            block_size=block_size,
            computation_time_ms=block_time_ms,
            memory_usage_mb=peak_memory_mb
        )
```

This approach provides maximum flexibility for capturing detailed performance metrics while maintaining compatibility with externally sourced pipeline code.

## Using the Existing Self-Forcing Profiling Example

Inferix provides a complete example demonstrating how to use the enhanced profiling capabilities:

### 1. Running the Example

Use the existing script to run the self-forcing pipeline with profiling:

```bash
cd /path/to/Inferix
./example/profiling/run_distributed_self_forcing.sh
```

### 2. Viewing Results

After completion, detailed profiling reports will be generated in the output directory, including:

- HTML format visual reports
- JSON format detailed data

The reports will include the new FPS and BPS metrics to help evaluate the model's real-time performance.

## Performance Optimization Recommendations

Based on the analysis results, the system will generate the following types of optimization recommendations:

1. **Model Size Optimization**: For models with more than 1 billion parameters, recommend using quantization or pruning techniques
2. **Diffusion Steps Optimization**: For diffusion processes with more than 30 steps, recommend reducing steps to improve generation speed
3. **Block Size Optimization**: For average block sizes exceeding 8 frames, recommend reducing block size to improve latency
4. **FPS Optimization**: For FPS below 10, recommend optimizing the model to achieve real-time generation
5. **BPS Optimization**: For BPS below 2, recommend optimizing block processing to improve throughput
6. **Multi-GPU Scaling**: Recommend using multiple GPUs to accelerate generation

## Best Practices

1. **Use Existing Examples**: Directly use the `example/profiling/run_distributed_self_forcing.sh` script to run pipelines with profiling
2. **Review Reports**: Analyze the generated HTML and JSON reports, focusing on FPS and BPS metrics
3. **Optimize Based on Recommendations**: Adjust model configurations and pipeline parameters based on the optimization recommendations in the reports
4. **Integrate Profiling Early**: Add profiling integration during pipeline development to catch performance issues early
5. **Monitor Real-time Metrics**: Pay attention to the real-time display of FPS and BPS metrics during generation

## Troubleshooting

### 1. Analysis Features Not Working

Check the following:

- Ensure the corresponding analysis options are enabled in the configuration file
- Ensure `profiling.enabled` is set to `true`
- Check for any error messages indicating analyzer initialization failure

### 2. Missing Analysis Data in Reports

Check the following:

- Ensure the recording methods are correctly called in the code
- Ensure data is recorded at the correct timing
- Check if any conditions are preventing data recording

### 3. Integration Issues

When integrating profiling into pipelines:

- Ensure the profiler object is properly passed to the pipeline constructor
- Verify that the profiler has the required methods (record_diffusion_step, record_block_computation)
- Handle potential exceptions when calling profiler methods to avoid breaking the pipeline

## Conclusion

With these enhancements, the Inferix profiling system can now provide deeper insights into model performance, helping developers optimize models and pipelines for better performance. The new FPS and BPS metrics are particularly useful for evaluating real-time video generation performance. Simply use the existing `example/profiling/run_distributed_self_forcing.sh` example to experience these new features.