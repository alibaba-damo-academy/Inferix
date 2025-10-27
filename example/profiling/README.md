# Inferix Enhanced Profiling Guide

This guide demonstrates how to use the enhanced profiling capabilities of Inferix through different configuration files.

## 🎯 Key Features

The enhanced profiling system provides:

### 1. Integrated Enhanced Monitoring
All enhanced profiling features are now part of the main [InferixProfiler](../../inferix/profiling/profiler.py) class:
- **Video Streaming Performance Analysis**: Monitor encoding performance, network latency, and buffer occupancy
- **Model Computation Analysis**: Detailed timing of individual model layers with parameter counting
- **Memory Pattern Analysis**: Track memory allocation patterns and fragmentation
- **Cache Efficiency Monitoring**: Monitor transformer model cache performance (hits/misses/evictions)

All features are only active when profiling is explicitly enabled, ensuring zero overhead when disabled.

### 2. Simple Configuration
Enhanced features are seamlessly integrated into the main profiler with no complex configuration needed.

## 📁 Project Structure

```
example/profiling/
├── configs/
│   └── profiling_config.yaml         # Basic profiling configuration
├── self_forcing_profiling.py         # Self-Forcing pipeline with profiling
├── run_distributed_self_forcing.sh   # Distributed profiling script
├── PROFILING_GUIDE.md                # Detailed profiling guide
└── README.md                         # This file
```

## ⚙️ Usage

### Basic Profiling
Use the existing [profiling_config.yaml](configs/profiling_config.yaml) configuration file with any pipeline:

```bash
# Run with basic profiling
cd example/profiling
python self_forcing_profiling.py --config configs/profiling_config.yaml
```

### Enhanced Profiling
The same configuration file automatically enables all enhanced features when `enabled: true`.

## 🎯 Benefits for Real-time Video Generation

### Performance Optimization
- **Bottleneck Elimination**: Identify and remove performance bottlenecks at layer granularity
- **Resource Balancing**: Optimize GPU/CPU utilization with detailed metrics
- **Memory Efficiency**: Reduce memory fragmentation with allocation pattern analysis

### Quality of Service
- **Streaming Reliability**: Ensure consistent frame delivery with drop rate monitoring
- **Latency Reduction**: Minimize processing and network delays with detailed timing
- **Adaptive Optimization**: Real-time adjustments for optimal performance

### Model Design Guidance
- **Computational Load Balancing**: Distribute work evenly across model layers
- **Memory-aware Design**: Optimize models for memory-constrained environments
- **Cache-efficient Architectures**: Design for optimal cache utilization

## 📈 Key Metrics Tracked

### Streaming Performance
- Frame encoding time
- Network latency
- Buffer occupancy
- Drop rate
- Current FPS

### Computation Performance
- Layer execution time
- Parameter counts
- Memory usage during computation
- Bottleneck detection

### Memory Patterns
- Allocation sizes and frequencies
- Fragmentation levels
- Deallocation patterns
- Peak memory usage

### Cache Efficiency
- Hit rates
- Miss rates
- Eviction rates
- Cache size optimization

## 📄 License

This example is part of the Inferix project and follows the same licensing terms.