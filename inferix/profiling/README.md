# Inferix Profiling Module

The Inferix Profiling Module provides comprehensive performance monitoring and analysis capabilities for video generation pipelines. It enables **profiling-driven pipeline design** by offering real-time monitoring during minute-long video generation and detailed performance reports to inspire more efficient model designs and real-time generation processes.

## 🚀 Key Features

### GPU-Focused Monitoring
- **Memory Usage**: Track GPU memory allocation, utilization, and peak usage
- **GPU Utilization**: Monitor GPU compute utilization across all devices
- **Temperature Monitoring**: Track thermal performance to prevent throttling
- **Power Consumption**: Monitor energy usage for efficiency analysis
- **Multi-GPU Support**: Monitor multiple GPUs simultaneously

### CPU & System Monitoring (Minimal)
- **CPU Usage**: Basic CPU utilization monitoring
- **System Memory**: Track system memory usage and availability
- **Load Average**: Monitor system load (Unix systems)

### Real-Time Performance Insights
- **Live Metrics Display**: Real-time performance metrics during generation
- **Stage-by-Stage Timing**: Profile individual pipeline stages
- **Bottleneck Identification**: Automatically identify performance bottlenecks
- **Custom Events**: Track custom application events

### Comprehensive Reporting
- **HTML Reports**: Rich, interactive performance reports with visualizations
- **JSON Reports**: Machine-readable reports for automated analysis
- **Performance Recommendations**: AI-driven optimization suggestions
- **Raw Data Export**: Export raw monitoring data for detailed analysis

### Report Aggregation Tools
- **Multi-Rank Aggregation**: Aggregate reports from distributed training runs
- **Cross-Session Analysis**: Compare performance across different sessions
- **Statistical Summaries**: Generate min/max/avg statistics across ranks

### Minimal Performance Impact
- **Optional Profiling**: Zero overhead when disabled
- **Configurable Sampling**: Adjust monitoring frequency to balance detail vs. performance
- **Background Monitoring**: Non-blocking monitoring in separate threads
- **Memory Efficient**: Configurable data point limits to prevent memory issues

## 📦 Installation

The profiling module is included with Inferix. Optional dependencies for enhanced GPU monitoring:

```bash
# For comprehensive GPU monitoring (recommended)
pip install pynvml

# For CPU monitoring (usually included)
pip install psutil
```

## 🎯 Quick Start

### Basic Usage

```python
from inferix.profiling import InferixProfiler, ProfilingConfig

# Configure profiling
config = ProfilingConfig(
    enabled=True,
    real_time_display=True,
    generate_final_report=True
)

# Create profiler
profiler = InferixProfiler(config)

# Profile your video generation
with profiler.session("my_video_generation") as session_id:
    
    with profiler.stage("text_encoding"):
        # Your text encoding code
        pass
    
    with profiler.stage("video_generation"):
        # Your video generation code
        pass
    
    with profiler.stage("vae_decode"):
        # Your VAE decoding code
        pass

# Reports are automatically generated when the session ends
```

### Integration with Existing Pipelines

```python
from inferix.utils.profiling_integration import ProfilingIntegration

# Easy integration with minimal code changes
profiling = ProfilingIntegration(config)

# Profile your entire pipeline
with profiling.session(prompt="A cat playing in the garden"):
    
    # Wrap existing functions/stages
    with profiling.stage("inference"):
        result = your_existing_function()
    
    # Add custom events
    profiling.add_event("milestone_reached", {"progress": 0.5})

# Get real-time metrics
metrics = profiling.get_current_metrics()
print(f"GPU Memory: {metrics['gpu']['memory_used_mb']}MB")
```

## ⚙️ Configuration

### YAML Configuration

```yaml
profiling:
  enabled: true
  real_time_display: true
  display_interval: 5.0
  
  # GPU monitoring
  gpu_monitor_interval: 0.5
  monitor_gpu_memory: true
  monitor_gpu_utilization: true
  monitor_gpu_temperature: true
  monitor_gpu_power: true
  
  # CPU monitoring (minimal)
  cpu_monitor_interval: 1.0
  monitor_cpu_usage: true
  monitor_system_memory: true
  
  # Reporting
  generate_final_report: true
  report_format: "both"  # "html", "json", "both"
  output_dir: "./profiling_reports"
  
  # Advanced options
  max_data_points: 10000
  profile_inference_steps: true
  profile_vae_decode: true
  profile_text_encoding: true
```

### Python Configuration

```python
from inferix.profiling.config import ProfilingConfig

config = ProfilingConfig(
    enabled=True,
    gpu_monitor_interval=0.5,
    cpu_monitor_interval=1.0,
    real_time_display=True,
    generate_final_report=True,
    report_format="both",
    output_dir="./profiling_reports",
    session_tags={"experiment": "optimization_v1"}
)
```

## 📊 Understanding Reports

### Real-Time Display

During generation, you'll see live metrics like:

```
=== Profiling Metrics (session_abc123) ===
GPU: Memory 8420MB / 24564MB (34.3%), Util 85.2%, Temp 72.1°C
CPU: 45.3%, Memory 12840MB / 32768MB (39.2%)
```

### HTML Reports

Rich interactive reports include:
- **Performance Summary**: Key metrics and bottlenecks
- **Timeline Visualizations**: GPU/CPU usage over time
- **Stage Analysis**: Time spent in each pipeline stage
- **Optimization Recommendations**: AI-driven performance suggestions
- **System Information**: Hardware and environment details

### JSON Reports

Machine-readable reports for automated analysis:
- Detailed metrics timelines
- Stage timing breakdowns
- Performance statistics
- Recommendation data

## 🛠️ Report Aggregation Tools

### Multi-Rank Report Aggregation

When running distributed training, each rank generates its own profiling reports. 
The `aggregate_reports` module can combine these into a single comprehensive report:

```python
from inferix.profiling.aggregate_reports import aggregate_reports

# Aggregate reports from all ranks
aggregate_reports(
    reports_dir="./distributed_profiling_reports",
    output_path="./aggregated_report.json"
)
```

### Command Line Usage

```bash
python -m inferix.profiling.aggregate_reports \
    --reports_dir ./distributed_profiling_reports \
    --output ./aggregated_report.json
```

## 🔧 Advanced Usage

### Custom Stage Profiling

```python
# Profile specific operations
with profiler.stage("custom_operation", metadata={"batch_size": 32}):
    result = custom_function()

# Profile functions with decorator
@profiler.profile_function("my_function")
def my_function():
    # Your code here
    pass
```

### Multi-GPU Monitoring

```python
# Monitor specific GPUs
gpu_monitor = GPUMonitor(device_ids=[0, 1, 2, 3])
```

### Diffusion Model Profiling

For diffusion models, Inferix provides specialized profiling capabilities that can be integrated using either decorators or direct code integration:

1. **Diffusion Step Analysis**: Track performance of each diffusion step with timing and memory usage
2. **Block Computation Analysis**: Monitor block-wise computation performance with FPS and BPS metrics
3. **Model Parameter Analysis**: Record model parameter counts for performance modeling

See [EXTENDING_PROFILING.md](EXTENDING_PROFILING.md) for detailed integration examples and best practices.

### Integration Approaches: Decorators vs. Direct Code Integration

Inferix supports two main approaches for integrating profiling capabilities into pipelines:

#### 1. Decorator-Based Integration

The decorator-based approach is suitable for high-level pipeline methods and provides a clean, declarative way to add profiling:

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

**Advantages**:
- Clean, declarative syntax
- Minimal code changes required
- Automatic session management
- Easy to add/remove profiling

**Limitations**:
- Limited access to fine-grained metrics (e.g., per-diffusion-step timing)
- Cannot capture detailed performance data within complex method implementations
- Not suitable for recording specific events like diffusion steps or block computations

#### 2. Direct Code Integration

Direct code integration is necessary for capturing detailed performance metrics within complex pipeline implementations:

```python
# In pipeline's inference method
for block_index in range(num_blocks):
    # Record diffusion steps with external profiler
    if self.profiler is not None and hasattr(self.profiler, 'record_diffusion_step'):
        self.profiler.record_diffusion_step(
            step=index,
            timestep=current_timestep.item() / 1000.0,
            block_size=current_num_frames,
            computation_time_ms=diffusion_time_ms,
            guidance_scale=getattr(self.args, 'guidance_scale', None)
        )
    
    # Record block computation with external profiler
    if self.profiler is not None and hasattr(self.profiler, 'record_block_computation'):
        self.profiler.record_block_computation(
            block_index=block_index,
            block_size=current_num_frames,
            computation_time_ms=block_time,
            memory_usage_mb=memory_used_mb
        )
```

**Advantages**:
- Fine-grained control over what gets profiled
- Access to detailed metrics like diffusion steps and block computations
- Ability to record specific performance data at precise moments
- Full access to intermediate values and calculations

**When to Use Direct Integration**:
- When profiling externally sourced pipelines (like open-source CausalInferencePipeline)
- When detailed diffusion model analysis is required
- When capturing performance data at specific points within complex algorithms
- When intermediate values are needed for accurate metrics

### Performance Optimization Workflow

1. **Enable Profiling**: Start with detailed monitoring
2. **Identify Bottlenecks**: Use reports to find performance issues
3. **Optimize Code**: Focus on stages taking >30% of total time
4. **Measure Impact**: Re-profile to verify improvements
5. **Production Deploy**: Use minimal profiling for monitoring

## 🎛️ Configuration Presets

### Development Mode
- High-frequency monitoring
- All features enabled
- Detailed reports and raw data
- Real-time display

### Production Mode
- Low-frequency monitoring
- Essential monitoring only
- JSON reports for automation
- Minimal overhead

### Debugging Mode
- Maximum detail
- Save raw monitoring data
- Frequent real-time updates
- All profiling features

## 📈 Performance Impact

The profiling system is designed for minimal impact:
- **Disabled**: Zero overhead
- **Basic Profiling**: < 2% overhead
- **Full Profiling**: < 5% overhead
- **Background Monitoring**: Non-blocking operation

When integrated into pipelines, the profiling system adds minimal overhead while providing detailed performance insights. The enhanced diffusion model analysis features provide FPS and BPS metrics that are particularly valuable for real-time video generation optimization.

## 🔍 Troubleshooting

### GPU Monitoring Not Working

```bash
# Install NVIDIA Management Library
pip install pynvml

# Check GPU availability
nvidia-smi
```

### High Memory Usage

```python
# Reduce data points
config.max_data_points = 5000

# Increase monitoring intervals
config.gpu_monitor_interval = 1.0
config.cpu_monitor_interval = 2.0
```

### Performance Overhead

```python
# Use production configuration
config.real_time_display = False
config.monitor_gpu_temperature = False
config.monitor_gpu_power = False
config.profile_text_encoding = False
```

## 🚀 Integration Examples

See the `example/` directory for complete integration examples:
- `profiling_example.py`: Complete pipeline with profiling
- `profiling_configs.yaml`: Configuration examples
- Integration with existing Inferix pipelines

## Pipeline Integration

Inferix provides flexible profiling integration options for pipelines. See [EXTENDING_PROFILING.md](EXTENDING_PROFILING.md) for detailed examples of integrating profiling capabilities into different pipeline implementations.

## 📝 API Reference

### Core Classes
- `InferixProfiler`: Main profiler class
- `ProfilingConfig`: Configuration management
- `ProfilingIntegration`: Easy pipeline integration
- `GPUMonitor`: GPU performance monitoring
- `CPUMonitor`: CPU and system monitoring
- `ProfilingReporter`: Report generation
- `DiffusionAnalyzer`: Diffusion model specific analysis

### Context Managers
- `profiler.session()`: Profile entire sessions
- `profiler.stage()`: Profile pipeline stages
- `integration.session()`: Easy session management

### Utilities
- `create_profiler_from_config()`: Create profiler from various config formats
- `get_profiling_summary()`: Get current performance summary
- `profile_pipeline_function()`: Function decorator for profiling
- `aggregate_reports()`: Aggregate reports from multiple ranks

### Decorators
- `@profile_method`: Profile a single method execution
- `@profile_session`: Wrap a method in a complete profiling session
- `@profile_diffusion_step`: Profile diffusion model steps
- `@profile_block_computation`: Profile block computation in diffusion models
- `@profile_model_parameters`: Profile model parameters
- `@add_profiling_event`: Add a custom event after method execution

### Diffusion Model Analysis Methods
- `profiler.record_diffusion_step()`: Record metrics for diffusion steps
- `profiler.record_block_computation()`: Record block computation metrics
- `profiler.record_model_parameters()`: Record model parameter counts
- `diffusion_analyzer.get_diffusion_analysis()`: Get diffusion step analysis
- `diffusion_analyzer.get_block_analysis()`: Get block computation analysis
- `diffusion_analyzer.get_model_analysis()`: Get model parameter analysis

### Command Line Tools
- `inferix.profiling.aggregate_reports`: Command line tool for report aggregation

## 🎯 Best Practices

1. **Start Simple**: Begin with basic profiling, add detail as needed
2. **Focus on GPU**: GPU metrics are most important for video generation
3. **Profile Stages**: Break down your pipeline into logical stages
4. **Monitor Trends**: Look for patterns in performance over time
5. **Optimize Iteratively**: Profile → Optimize → Re-profile → Repeat
6. **Production Monitoring**: Keep lightweight profiling in production
7. **Save Reports**: Archive reports for performance trend analysis
8. **Use Diffusion Analysis**: Leverage the enhanced diffusion model analysis for real-time performance optimization
9. **Monitor FPS and BPS**: Pay attention to Frames Per Second and Blocks Per Second metrics for streaming performance

## 🤝 Contributing

The profiling module is part of the Inferix project. Contributions are welcome:
- Bug reports and fixes
- New monitoring capabilities
- Performance optimizations
- Documentation improvements

## 📄 License

Licensed under the Apache License 2.0. See the LICENSE file for details.