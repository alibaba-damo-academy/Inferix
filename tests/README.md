# Tests Directory

This directory contains all test files for the Inferix project, organized by different types.

## Directory Structure

```
tests/
├── unit/               # Unit tests
│   └── test_profiling.py   # Performance profiling module unit tests
├── integration/        # Integration tests
├── configs/            # Test configuration files
│   ├── causvid.yaml
│   ├── default_config.yaml
│   └── self_forcing_dmd.yaml
└── data/               # Test data
```

## Running Tests

### Unit Tests

```bash
# Run all unit tests
python -m pytest tests/unit/

# Run specific unit tests
python tests/unit/test_profiling.py
```

### Integration Tests

```bash
# Run integration tests
python -m pytest tests/integration/
```

## Test Configuration

Test configuration files are located in the `tests/configs/` directory, used for various test scenarios.

## Adding New Tests

1. Unit tests should be placed in the `tests/unit/` directory
2. Integration tests should be placed in the `tests/integration/` directory  
3. Test configuration files should be placed in the `tests/configs/` directory
4. Test data should be placed in the `tests/data/` directory

## Writing Tests for New Models

When integrating new models with Inferix, you should also provide corresponding tests:

### Unit Tests

Create unit tests for your model's core components in `tests/unit/`:

```python
# tests/unit/test_your_model.py
import pytest
import torch
from inferix.pipeline.your_model.pipeline import YourModelPipeline

def test_model_initialization():
    """Test model initialization"""
    pipeline = YourModelPipeline(config_path="path/to/test_config.yaml")
    assert pipeline is not None
    assert pipeline.model is not None

def test_checkpoint_loading():
    """Test checkpoint loading"""
    pipeline = YourModelPipeline(config_path="path/to/test_config.yaml")
    # Add your test logic here
```

### Integration Tests

Create integration tests in `tests/integration/`:

```python
# tests/integration/test_your_model_integration.py
import pytest
from inferix.pipeline.your_model.pipeline import YourModelPipeline

@pytest.mark.integration
def test_full_pipeline_execution():
    """Test full pipeline execution"""
    pipeline = YourModelPipeline(config_path="path/to/test_config.yaml")
    pipeline.load_checkpoint("path/to/test_checkpoint.pt")
    
    result = pipeline.run_text_to_video(prompts=["test prompt"])
    assert result is not None
```

## Performance Testing

For performance-critical components, add benchmarks:

```python
# tests/unit/test_performance.py
import pytest
import time
from inferix.pipeline.your_model.pipeline import YourModelPipeline

@pytest.mark.benchmark
def test_inference_performance(benchmark):
    """Benchmark inference performance"""
    pipeline = YourModelPipeline(config_path="path/to/test_config.yaml")
    
    def inference():
        return pipeline.run_text_to_video(prompts=["test prompt"])
    
    result = benchmark(inference)
    assert result is not None
```

## Profiling Tests

To test the profiling functionality of your model:

```python
# tests/unit/test_your_model_profiling.py
from inferix.pipeline.your_model.pipeline import YourModelPipeline
from inferix.profiling.config import ProfilingConfig

def test_profiling_integration():
    """Test profiling integration"""
    profiling_config = ProfilingConfig(enabled=True, output_dir="./test_reports")
    
    pipeline = YourModelPipeline(
        config_path="path/to/test_config.yaml",
        profiling_config=profiling_config
    )
    
    assert pipeline._profiling_enabled == True
    assert pipeline._profiler is not None
```