"""Inferix Profiling Module

This module provides comprehensive profiling capabilities for video generation pipelines,
enabling real-time monitoring and performance analysis to inspire more efficient model designs
and real-time generation processes.

Key Features:
- GPU-focused performance monitoring (memory, utilization, temperature)
- Minimal CPU monitoring (CPU usage, system memory)
- Real-time metrics collection during minute-long video generation
- Configurable profiling with optional enable/disable
- Comprehensive reporting after generation completion
- Report aggregation tools for distributed training
"""

from .profiler import InferixProfiler
from .monitors import GPUMonitor, CPUMonitor
from .reporter import ProfilingReporter
from .config import ProfilingConfig
from .decorators import profile_method, profile_session, add_profiling_event

# Conditional import for aggregate_reports to avoid circular dependencies
try:
    from .aggregate_reports import aggregate_reports
    __all__ = ['InferixProfiler', 'GPUMonitor', 'CPUMonitor', 'ProfilingReporter', 'ProfilingConfig', 'aggregate_reports', 'profile_method', 'profile_session', 'add_profiling_event']
except ImportError:
    __all__ = ['InferixProfiler', 'GPUMonitor', 'CPUMonitor', 'ProfilingReporter', 'ProfilingConfig', 'profile_method', 'profile_session', 'add_profiling_event']

# Conditional import for diffusion_analyzer
try:
    # Use importlib to avoid static analysis issues
    import importlib
    diffusion_analyzer_module = importlib.import_module('.diffusion_analyzer', package='inferix.profiling')
    DiffusionAnalyzer = getattr(diffusion_analyzer_module, 'DiffusionAnalyzer')
    if 'DiffusionAnalyzer' not in __all__:
        __all__ += ['DiffusionAnalyzer']
except (ImportError, AttributeError):
    pass