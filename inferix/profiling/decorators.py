"""Profiling decorators for Inferix pipelines.

This module provides decorator utilities to simplify profiling integration
in pipeline methods, reducing boilerplate code and improving maintainability.
Includes enhanced decorators for streaming, computation, and memory profiling.
"""

from functools import wraps
from typing import Optional, Dict, Any, Callable
from contextlib import nullcontext
import time


def profile_method(stage_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
    """Decorator to profile a method using the pipeline's profiler.
    
    This decorator simplifies the integration of profiling by automatically
    wrapping the method with the appropriate profiler context.
    
    Args:
        stage_name: Name of the profiling stage (defaults to method name)
        metadata: Optional metadata to attach to the profiling stage
        
    Example:
        @profile_method("video_generation")
        def run_text_to_video(self, prompts):
            # Method implementation
            pass
            
        @profile_method(metadata={"mode": "t2v"})
        def run_text_to_video(self, prompts):
            # Method implementation
            pass
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Get the stage name, defaulting to the function name if not provided
            name = stage_name or func.__name__
            
            # Use the profiler context if profiling is enabled, otherwise use nullcontext
            if hasattr(self, '_profiling_enabled') and self._profiling_enabled and hasattr(self, '_profiler') and self._profiler is not None:
                context = self._profiler.stage(name, metadata)
            else:
                context = nullcontext()
            
            with context:
                return func(self, *args, **kwargs)
        return wrapper
    return decorator


def profile_layer(layer_name: Optional[str] = None, parameters_count: Optional[int] = None):
    """Decorator to profile a model layer computation.
    
    This decorator automatically times the layer computation and records metrics.
    
    Args:
        layer_name: Name of the layer (defaults to function name)
        parameters_count: Optional number of parameters in the layer
        
    Example:
        @profile_layer("attention_layer", parameters_count=1000000)
        def compute_attention(self, x):
            # Layer implementation
            pass
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            name = layer_name or func.__name__
            
            # Start timing if profiling is enabled
            start_time = None
            if hasattr(self, '_profiling_enabled') and self._profiling_enabled and hasattr(self, '_profiler') and self._profiler is not None:
                start_time = self._profiler.start_layer_computation(name)
            else:
                start_time = time.time()
            
            try:
                result = func(self, *args, **kwargs)
                return result
            finally:
                # End timing and record metrics
                if hasattr(self, '_profiling_enabled') and self._profiling_enabled and hasattr(self, '_profiler') and self._profiler is not None:
                    self._profiler.end_layer_computation(name, start_time, parameters_count)
        return wrapper
    return decorator


def profile_session(session_id: str, tags: Optional[Dict[str, Any]] = None):
    """Decorator to wrap a method in a profiling session.
    
    This decorator automatically starts a profiling session before the method
    execution and ends it afterwards.
    
    Args:
        session_id: Identifier for the profiling session
        tags: Optional tags to attach to the session
        
    Example:
        @profile_session("t2v_generation", {"mode": "text_to_video"})
        def run_text_to_video(self, prompts):
            # Method implementation
            pass
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Start session if profiling is enabled
            if hasattr(self, '_profiling_enabled') and self._profiling_enabled and hasattr(self, '_profiler') and self._profiler is not None:
                self._profiler.start_session(session_id, tags)
            
            try:
                result = func(self, *args, **kwargs)
                return result
            finally:
                # End session if profiling is enabled
                if hasattr(self, '_profiling_enabled') and self._profiling_enabled and hasattr(self, '_profiler') and self._profiler is not None:
                    self._profiler.end_session()
        return wrapper
    return decorator


def profile_streaming(batch_size_func: Optional[Callable] = None):
    """Decorator to profile video streaming performance.
    
    This decorator automatically records streaming performance metrics.
    
    Args:
        batch_size_func: Optional function to determine batch size from args/kwargs
        
    Example:
        @profile_streaming(lambda args, kwargs: len(args[0]))
        def stream_frames(self, frames):
            # Streaming implementation
            pass
            
        @profile_streaming()
        def stream_batch(self, batch):
            # Streaming implementation
            pass
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(self, *args, **kwargs)
                return result
            finally:
                # Record streaming metrics if profiling is enabled
                if hasattr(self, '_profiling_enabled') and self._profiling_enabled and hasattr(self, '_profiler') and self._profiler is not None:
                    end_time = time.time()
                    encoding_time_ms = (end_time - start_time) * 1000
                    
                    # Determine batch size
                    batch_size = 1
                    if batch_size_func:
                        try:
                            batch_size = batch_size_func(args, kwargs)
                        except:
                            batch_size = 1
                    
                    self._profiler.record_streaming_event(
                        batch_size=batch_size,
                        encoding_time_ms=encoding_time_ms
                    )
        return wrapper
    return decorator


def profile_diffusion_step(step_func: Optional[Callable] = None):
    """Decorator to profile diffusion model steps.
    
    This decorator automatically records diffusion step metrics.
    
    Args:
        step_func: Optional function to extract step information from args/kwargs
        
    Example:
        @profile_diffusion_step(lambda args, kwargs: {"step": args[0], "timestep": args[1]})
        def diffusion_step(self, step, timestep, ...):
            # Diffusion step implementation
            pass
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Record start time
            start_time = time.time()
            
            try:
                result = func(self, *args, **kwargs)
                return result
            finally:
                # Record diffusion step metrics if profiling is enabled
                if hasattr(self, '_profiling_enabled') and self._profiling_enabled and hasattr(self, '_profiler') and self._profiler is not None:
                    end_time = time.time()
                    computation_time_ms = (end_time - start_time) * 1000
                    
                    # Extract step information
                    step_info = {}
                    if step_func:
                        try:
                            step_info = step_func(args, kwargs)
                        except:
                            step_info = {}
                    
                    # Add default values if not provided
                    if 'step' not in step_info:
                        step_info['step'] = 0
                    if 'timestep' not in step_info:
                        step_info['timestep'] = 0.0
                    if 'block_size' not in step_info:
                        step_info['block_size'] = 1
                    if 'guidance_scale' not in step_info:
                        step_info['guidance_scale'] = None
                    
                    self._profiler.record_diffusion_step(
                        step=step_info['step'],
                        timestep=step_info['timestep'],
                        block_size=step_info['block_size'],
                        computation_time_ms=computation_time_ms,
                        guidance_scale=step_info['guidance_scale']
                    )
        return wrapper
    return decorator


def profile_model_parameters(model_name_func: Optional[Callable] = None):
    """Decorator to profile model parameters.
    
    This decorator automatically records model parameter counts.
    
    Args:
        model_name_func: Optional function to extract model information from args/kwargs
        
    Example:
        @profile_model_parameters(lambda args, kwargs: {"model_name": args[0], "parameters_count": args[1]})
        def record_model_parameters(self, model_name, parameters_count, model_type):
            # Record model parameters
            pass
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            
            # Record model parameters if profiling is enabled
            if hasattr(self, '_profiling_enabled') and self._profiling_enabled and hasattr(self, '_profiler') and self._profiler is not None:
                # Extract model information
                model_info = {}
                if model_name_func:
                    try:
                        model_info = model_name_func(args, kwargs)
                    except:
                        model_info = {}
                
                # Add default values if not provided
                if 'model_name' not in model_info:
                    model_info['model_name'] = "unknown_model"
                if 'parameters_count' not in model_info:
                    model_info['parameters_count'] = 0
                if 'model_type' not in model_info:
                    model_info['model_type'] = "unknown"
                
                self._profiler.record_model_parameters(
                    model_name=model_info['model_name'],
                    parameters_count=model_info['parameters_count'],
                    model_type=model_info['model_type']
                )
                
            return result
        return wrapper
    return decorator


def profile_block_computation(block_info_func: Optional[Callable] = None):
    """Decorator to profile block computation in diffusion models.
    
    This decorator automatically records block computation metrics.
    
    Args:
        block_info_func: Optional function to extract block information from args/kwargs
        
    Example:
        @profile_block_computation(lambda args, kwargs: {
            "block_index": args[0],
            "block_size": args[1],
            "computation_time_ms": args[2]
        })
        def process_block(self, block_index, block_size, computation_time_ms):
            # Block processing implementation
            pass
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Execute the function first
            result = func(self, *args, **kwargs)
            
            # Record block computation metrics if profiling is enabled
            if hasattr(self, '_profiling_enabled') and self._profiling_enabled and hasattr(self, '_profiler') and self._profiler is not None:
                # Extract block information
                block_info = {}
                if block_info_func:
                    try:
                        block_info = block_info_func(args, kwargs)
                    except:
                        block_info = {}
                
                # Add default values if not provided
                if 'block_index' not in block_info:
                    block_info['block_index'] = 0
                if 'block_size' not in block_info:
                    block_info['block_size'] = 1
                if 'computation_time_ms' not in block_info:
                    block_info['computation_time_ms'] = 0.0
                if 'memory_usage_mb' not in block_info:
                    block_info['memory_usage_mb'] = 0.0
                
                # Record the block computation
                try:
                    self._profiler.record_block_computation(
                        block_index=block_info['block_index'],
                        block_size=block_info['block_size'],
                        computation_time_ms=block_info['computation_time_ms'],
                        memory_usage_mb=block_info['memory_usage_mb']
                    )
                except Exception as e:
                    # Silently ignore profiler errors to avoid breaking the pipeline
                    pass
                
            return result
        return wrapper
    return decorator


def add_profiling_event(event_name: str, data_func: Optional[Callable] = None):
    """Decorator to add a profiling event after method execution.
    
    This decorator adds a custom profiling event after the method completes.
    
    Args:
        event_name: Name of the event to add
        data_func: Optional function to generate event data.
                  The function signature should be: data_func(result, *args, **kwargs)
                  Where:
                  - result: The return value of the decorated method
                  - *args: Positional arguments passed to the decorated method
                  - **kwargs: Keyword arguments passed to the decorated method
        
    Example:
        @add_profiling_event("checkpoint_loaded")
        def load_checkpoint(self, checkpoint_path, **kwargs):
            # Method implementation
            pass
            
        @add_profiling_event("generation_completed", lambda result, *args, **kwargs: {
            "output_shape": list(result.shape) if result is not None else None
        })
        def run_text_to_video(self, prompts, **kwargs):
            # Method implementation
            return result
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            
            # Add event if profiling is enabled
            if hasattr(self, '_profiling_enabled') and self._profiling_enabled and hasattr(self, '_profiler') and self._profiler is not None:
                if data_func:
                    try:
                        data = data_func(result, *args, **kwargs)
                    except Exception:
                        # If the callback function fails, log the error and continue
                        data = None
                else:
                    data = None
                self._profiler.add_event(event_name, data)
                
            return result
        return wrapper
    return decorator
