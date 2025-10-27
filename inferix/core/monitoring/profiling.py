"""Profiling integration utilities for Inferix pipelines.

This module provides utilities to easily integrate profiling capabilities 
into existing video generation pipelines with minimal code changes.
"""

from typing import Optional, Dict, Any, Union
from contextlib import contextmanager
import hashlib
import time

from ...profiling import InferixProfiler
from ...profiling.config import ProfilingConfig


class ProfilingIntegration:
    """Helper class to integrate profiling into pipelines."""
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize profiling integration.
        
        Args:
            config: Either a ProfilingConfig or any config object containing profiling_config
        """
        if config is None:
            self.profiler = None
        elif isinstance(config, ProfilingConfig):
            # Direct ProfilingConfig
            if config.enabled:
                self.profiler = InferixProfiler(config)
            else:
                self.profiler = None
        elif hasattr(config, 'profiling_config'):
            # External config with profiling_config attribute
            profiling_config_attr = getattr(config, 'profiling_config', None)
            if profiling_config_attr and hasattr(profiling_config_attr, 'enabled') and profiling_config_attr.enabled:
                # Convert external config's ProfilingConfig to our ProfilingConfig
                profiling_config = ProfilingConfig(
                    enabled=getattr(profiling_config_attr, 'enabled', False),
                    gpu_monitor_interval=getattr(profiling_config_attr, 'gpu_monitor_interval', 0.5),
                    cpu_monitor_interval=getattr(profiling_config_attr, 'cpu_monitor_interval', 1.0),
                    monitor_gpu_memory=getattr(profiling_config_attr, 'monitor_gpu_memory', True),
                    monitor_gpu_utilization=getattr(profiling_config_attr, 'monitor_gpu_utilization', True),
                    monitor_gpu_temperature=getattr(profiling_config_attr, 'monitor_gpu_temperature', True),
                    monitor_gpu_power=getattr(profiling_config_attr, 'monitor_gpu_power', True),
                    monitor_cpu_usage=getattr(profiling_config_attr, 'monitor_cpu_usage', True),
                    monitor_system_memory=getattr(profiling_config_attr, 'monitor_system_memory', True),
                    real_time_display=getattr(profiling_config_attr, 'real_time_display', True),
                    display_interval=getattr(profiling_config_attr, 'display_interval', 5.0),
                    generate_final_report=getattr(profiling_config_attr, 'generate_final_report', True),
                    save_raw_data=getattr(profiling_config_attr, 'save_raw_data', False),
                    output_dir=getattr(profiling_config_attr, 'output_dir', None),
                    report_format=getattr(profiling_config_attr, 'report_format', 'both'),
                    max_data_points=getattr(profiling_config_attr, 'max_data_points', 10000),
                    profile_inference_steps=getattr(profiling_config_attr, 'profile_inference_steps', True),
                    profile_vae_decode=getattr(profiling_config_attr, 'profile_vae_decode', True),
                    profile_text_encoding=getattr(profiling_config_attr, 'profile_text_encoding', True),
                    session_tags=getattr(profiling_config_attr, 'session_tags', {})
                )
                self.profiler = InferixProfiler(profiling_config)
            else:
                self.profiler = None
        else:
            # Unknown config type, assume not enabled
            self.profiler = None
    
    @property
    def enabled(self) -> bool:
        """Check if profiling is enabled."""
        return self.profiler is not None
    
    def start_session(self, session_id: Optional[str] = None, 
                     prompt: Optional[str] = None, 
                     tags: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Start a profiling session.
        
        Args:
            session_id: Optional session ID. If not provided, will generate one from prompt
            prompt: Optional prompt text to use for session ID generation
            tags: Optional metadata tags
            
        Returns:
            Session ID if profiling is enabled, None otherwise
        """
        if not self.profiler:
            return None
        
        if session_id is None:
            if prompt:
                # Generate session ID from prompt hash and timestamp
                prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()[:8]
                timestamp = int(time.time())
                session_id = f"session_{prompt_hash}_{timestamp}"
            else:
                session_id = f"session_{int(time.time())}"
        
        # Add prompt to session tags if provided
        session_tags = tags or {}
        if prompt:
            session_tags['prompt'] = prompt
            session_tags['prompt_length'] = len(prompt)
        
        return self.profiler.start_session(session_id, session_tags)
    
    def end_session(self) -> Optional[str]:
        """End the current profiling session."""
        if not self.profiler:
            return None
        return self.profiler.end_session()
    
    @contextmanager
    def session(self, session_id: Optional[str] = None, 
                prompt: Optional[str] = None, 
                tags: Optional[Dict[str, Any]] = None):
        """Context manager for profiling sessions.
        
        Args:
            session_id: Optional session ID
            prompt: Optional prompt text
            tags: Optional metadata tags
        """
        if not self.profiler:
            yield None
            return
        
        session_id = self.start_session(session_id, prompt, tags)
        try:
            yield session_id
        finally:
            self.end_session()
    
    @contextmanager
    def stage(self, stage_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for profiling pipeline stages.
        
        Args:
            stage_name: Name of the pipeline stage
            metadata: Optional metadata for the stage
        """
        if not self.profiler:
            yield
            return
        
        with self.profiler.stage(stage_name, metadata):
            yield
    
    def add_event(self, event_name: str, data: Optional[Dict[str, Any]] = None):
        """Add a custom event to the current session.
        
        Args:
            event_name: Name of the event
            data: Optional additional data
        """
        if not self.profiler:
            return
        self.profiler.add_event(event_name, data)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        if not self.profiler:
            return {}
        return self.profiler.get_current_metrics()
    
    def cleanup(self):
        """Clean up profiling resources."""
        if self.profiler:
            self.profiler.cleanup()


def profile_pipeline_function(stage_name: str, profiling_integration: Optional[ProfilingIntegration] = None):
    """Decorator to profile pipeline functions.
    
    Args:
        stage_name: Name of the pipeline stage
        profiling_integration: Optional profiling integration instance
        
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if profiling_integration and profiling_integration.enabled:
                with profiling_integration.stage(stage_name):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator


# Convenience functions for quick integration
def create_profiler_from_config(config: Union[Dict[str, Any], ProfilingConfig, Any, None]) -> Optional[ProfilingIntegration]:
    """Create a profiling integration from various config formats.
    
    Args:
        config: Configuration in various formats
        
    Returns:
        ProfilingIntegration instance or None if profiling is disabled
    """
    if config is None:
        return None
    
    if isinstance(config, dict):
        # Handle dictionary config (e.g., from YAML/JSON)
        profiling_dict = config.get('profiling', {})
        if profiling_dict.get('enabled', False):
            profiling_config = ProfilingConfig(**profiling_dict)
            return ProfilingIntegration(profiling_config)
        return None
    
    # Handle external config or ProfilingConfig
    return ProfilingIntegration(config)


def get_profiling_summary(profiling_integration: Optional[ProfilingIntegration]) -> Dict[str, Any]:
    """Get a summary of current profiling metrics.
    
    Args:
        profiling_integration: Profiling integration instance
        
    Returns:
        Summary of profiling metrics
    """
    if not profiling_integration or not profiling_integration.enabled:
        return {'profiling_enabled': False}
    
    metrics = profiling_integration.get_current_metrics()
    summary = {'profiling_enabled': True, 'timestamp': metrics.get('timestamp', 0)}
    
    # GPU summary
    if 'gpu' in metrics:
        gpu = metrics['gpu']
        summary['gpu'] = {
            'memory_used_mb': gpu.get('total_gpu_memory_used_mb', 0),
            'memory_total_mb': gpu.get('total_gpu_memory_total_mb', 0),
            'utilization_percent': gpu.get('max_gpu_utilization', 0),
            'temperature_celsius': gpu.get('max_gpu_temperature', 0)
        }
        
        if summary['gpu']['memory_total_mb'] > 0:
            summary['gpu']['memory_utilization_percent'] = (
                summary['gpu']['memory_used_mb'] / summary['gpu']['memory_total_mb'] * 100
            )
    
    # CPU summary
    if 'cpu' in metrics:
        cpu = metrics['cpu']
        summary['cpu'] = {
            'usage_percent': cpu.get('cpu_usage_percent', 0),
            'memory_used_mb': cpu.get('memory_used_mb', 0),
            'memory_total_mb': cpu.get('memory_total_mb', 0)
        }
    
    return summary