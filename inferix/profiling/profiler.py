import time
import threading
from typing import Dict, Any, Optional, List, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
import os

from .config import ProfilingConfig
from .monitors import GPUMonitor, CPUMonitor, MetricSnapshot
from .reporter import ProfilingReporter

# Try to import DiffusionAnalyzer, but make it optional
try:
    # Use importlib to avoid static analysis issues
    import importlib
    diffusion_analyzer_module = importlib.import_module('.diffusion_analyzer', package='inferix.profiling')
    DiffusionAnalyzer = getattr(diffusion_analyzer_module, 'DiffusionAnalyzer')
    DIFFUSION_ANALYZER_AVAILABLE = True
except (ImportError, AttributeError):
    DiffusionAnalyzer = None  # type: ignore
    DIFFUSION_ANALYZER_AVAILABLE = False


@dataclass
class ProfilingSession:
    """Represents a profiling session with metadata."""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    custom_events: List[Dict[str, Any]] = field(default_factory=list)
    stage_timings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    streaming_events: List[Dict[str, Any]] = field(default_factory=list)
    dropped_frames: int = 0
    layer_timings: Dict[str, List[float]] = field(default_factory=dict)
    memory_events: List[Dict[str, Any]] = field(default_factory=list)
    cache_stats: Dict[str, int] = field(default_factory=lambda: {
        'hits': 0,
        'misses': 0,
        'evictions': 0,
        'total_requests': 0
    })
    
    @property
    def duration(self) -> Optional[float]:
        """Get session duration if ended."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return None
    
    @property
    def is_active(self) -> bool:
        """Check if session is still active."""
        return self.end_time is None


class InferixProfiler:
    """Main profiler class for Inferix video generation pipelines.
    
    This profiler provides comprehensive monitoring capabilities:
    - GPU performance monitoring (memory, utilization, temperature, power)
    - Minimal CPU monitoring (usage, system memory)
    - Real-time metrics display during generation
    - Detailed reporting after completion
    - Pipeline stage timing and analysis
    - Video streaming performance analysis
    - Model computation analysis
    - Communication overhead monitoring
    - Memory usage pattern analysis
    - KV Cache efficiency monitoring
    """
    
    def __init__(self, config: Optional[ProfilingConfig] = None):
        self.config = config or ProfilingConfig()
        
        # Initialize monitors
        self.gpu_monitor: Optional[GPUMonitor] = None
        self.cpu_monitor: Optional[CPUMonitor] = None
        self.reporter: Optional[ProfilingReporter] = None
        
        # Session management
        self.current_session: Optional[ProfilingSession] = None
        self.sessions: Dict[str, ProfilingSession] = {}
        
        # Real-time display
        self._display_thread: Optional[threading.Thread] = None
        self._display_stop_event = threading.Event()
        self._last_display_time = 0.0
        
        # Stage timing
        self._stage_stack: List[Dict[str, Any]] = []
        
        # Streaming performance tracking
        self._streaming_events: List[Dict[str, Any]] = []
        self._dropped_frames = 0
        
        # Computation tracking
        self._layer_timings: Dict[str, List[float]] = {}
        
        # Memory tracking
        self._memory_events: List[Dict[str, Any]] = []
        
        # Cache tracking
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        
        # Diffusion model analysis
        self.diffusion_analyzer = None
        if self.config.enabled and DIFFUSION_ANALYZER_AVAILABLE and DiffusionAnalyzer is not None:
            self.diffusion_analyzer = DiffusionAnalyzer(self)
        
        if self.config.enabled:
            self._initialize_monitors()
    
    def _initialize_monitors(self):
        """Initialize monitoring components based on configuration."""
        if not self.config.enabled:
            return
        
        # Initialize GPU monitor
        if (self.config.monitor_gpu_memory or 
            self.config.monitor_gpu_utilization or 
            self.config.monitor_gpu_temperature or 
            self.config.monitor_gpu_power):
            try:
                self.gpu_monitor = GPUMonitor(
                    interval=self.config.gpu_monitor_interval,
                    max_data_points=self.config.max_data_points
                )
            except Exception as e:
                print(f"Warning: Failed to initialize GPU monitor: {e}")
        
        # Initialize CPU monitor
        if self.config.monitor_cpu_usage or self.config.monitor_system_memory:
            try:
                self.cpu_monitor = CPUMonitor(
                    interval=self.config.cpu_monitor_interval,
                    max_data_points=self.config.max_data_points
                )
            except Exception as e:
                print(f"Warning: Failed to initialize CPU monitor: {e}")
        
        # Initialize reporter
        self.reporter = ProfilingReporter(self.config)
    
    def start_session(self, session_id: str, tags: Optional[Dict[str, Any]] = None) -> str:
        """Start a new profiling session.
        
        Args:
            session_id: Unique identifier for this session
            tags: Optional metadata tags for the session
            
        Returns:
            The session ID
        """
        if not self.config.enabled:
            return session_id
        
        if self.current_session is not None:
            print(f"Warning: Starting new session '{session_id}' while session "
                  f"'{self.current_session.session_id}' is still active")
            self.end_session()
        
        # Create new session
        session_tags = dict(self.config.session_tags)
        if tags:
            session_tags.update(tags)
        
        self.current_session = ProfilingSession(
            session_id=session_id,
            start_time=time.time(),
            tags=session_tags
        )
        self.sessions[session_id] = self.current_session
        
        # Start monitors
        if self.gpu_monitor:
            self.gpu_monitor.start_monitoring()
        if self.cpu_monitor:
            self.cpu_monitor.start_monitoring()
        
        # Start real-time display if enabled
        if self.config.real_time_display:
            self._start_real_time_display()
        
        print(f"Profiling session '{session_id}' started")
        return session_id
    
    def end_session(self) -> Optional[str]:
        """End the current profiling session.
        
        Returns:
            The session ID if there was an active session, None otherwise
        """
        if not self.config.enabled or self.current_session is None:
            return None
        
        session_id = self.current_session.session_id
        self.current_session.end_time = time.time()
        
        # Save enhanced profiling data
        self.current_session.streaming_events = self._streaming_events.copy()
        self.current_session.dropped_frames = self._dropped_frames
        self.current_session.layer_timings = {k: v.copy() for k, v in self._layer_timings.items()}
        self.current_session.memory_events = self._memory_events.copy()
        self.current_session.cache_stats = self._cache_stats.copy()
        
        # Stop monitors
        if self.gpu_monitor:
            self.gpu_monitor.stop_monitoring()
        if self.cpu_monitor:
            self.cpu_monitor.stop_monitoring()
        
        # Stop real-time display
        self._stop_real_time_display()
        
        # Generate final report
        if self.config.generate_final_report and self.reporter:
            self._generate_session_report(self.current_session)
        
        print(f"Profiling session '{session_id}' ended (duration: "
              f"{self.current_session.duration:.2f}s)")
        
        # Reset enhanced profiling data
        self._streaming_events.clear()
        self._dropped_frames = 0
        self._layer_timings.clear()
        self._memory_events.clear()
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        
        self.current_session = None
        return session_id
    
    def add_event(self, event_name: str, data: Optional[Dict[str, Any]] = None):
        """Add a custom event to the current session.
        
        Args:
            event_name: Name of the event
            data: Optional additional data for the event
        """
        if not self.config.enabled or self.current_session is None:
            return
        
        event = {
            'name': event_name,
            'timestamp': time.time(),
            'data': data or {}
        }
        self.current_session.custom_events.append(event)
    
    # Enhanced profiling methods
    
    def record_streaming_event(self, batch_size: int, encoding_time_ms: float, 
                             network_latency_ms: float = 0.0, buffer_occupancy: float = 0.0):
        """Record a video streaming event.
        
        Args:
            batch_size: Number of frames in the batch
            encoding_time_ms: Time taken to encode the batch in milliseconds
            network_latency_ms: Network latency in milliseconds
            buffer_occupancy: Buffer occupancy percentage
        """
        if not self.config.enabled or self.current_session is None:
            return
        
        event = {
            'timestamp': time.time(),
            'batch_size': batch_size,
            'encoding_time_ms': encoding_time_ms,
            'network_latency_ms': network_latency_ms,
            'buffer_occupancy': buffer_occupancy
        }
        self._streaming_events.append(event)
    
    def record_dropped_frames(self, count: int = 1):
        """Record dropped frames during streaming.
        
        Args:
            count: Number of dropped frames
        """
        if not self.config.enabled:
            return
        
        self._dropped_frames += count
    
    def start_layer_computation(self, layer_name: str) -> float:
        """Start timing a layer computation.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Start time as timestamp
        """
        return time.time()
    
    def end_layer_computation(self, layer_name: str, start_time: float, 
                            parameters_count: Optional[int] = None):
        """End timing a layer computation and record metrics.
        
        Args:
            layer_name: Name of the layer
            start_time: Start time from start_layer_computation
            parameters_count: Optional number of parameters in the layer
            
        Returns:
            Computation time in milliseconds
        """
        if not self.config.enabled:
            return (time.time() - start_time) * 1000
        
        end_time = time.time()
        computation_time_ms = (end_time - start_time) * 1000
        
        # Record timing for this layer
        if layer_name not in self._layer_timings:
            self._layer_timings[layer_name] = []
        self._layer_timings[layer_name].append(computation_time_ms)
        
        # Add event for detailed tracking
        self.add_event("layer_computation", {
            'layer_name': layer_name,
            'computation_time_ms': computation_time_ms,
            'parameters_count': parameters_count
        })
        
        return computation_time_ms
    
    def record_memory_event(self, event_type: str, size_bytes: int, 
                          allocator: str = "unknown", context: str = ""):
        """Record a memory allocation/deallocation event.
        
        Args:
            event_type: Type of event ("alloc", "dealloc", etc.)
            size_bytes: Size of allocation in bytes
            allocator: Allocator used
            context: Context information
        """
        if not self.config.enabled:
            return
        
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'size_bytes': size_bytes,
            'allocator': allocator,
            'context': context
        }
        self._memory_events.append(event)
    
    def record_cache_event(self, event_type: str, cache_size: int, 
                          requested_size: int = 0, evicted_size: int = 0):
        """Record a cache event.
        
        Args:
            event_type: Type of event ("hits", "misses", "evictions")
            cache_size: Current cache size
            requested_size: Size of requested data
            evicted_size: Size of evicted data
        """
        if not self.config.enabled:
            return
        
        if event_type in self._cache_stats:
            self._cache_stats[event_type] += 1
        self._cache_stats['total_requests'] += 1
        
        # Add event for detailed tracking
        self.add_event("cache_event", {
            'event_type': event_type,
            'cache_size': cache_size,
            'requested_size': requested_size,
            'evicted_size': evicted_size
        })
    
    # Diffusion model specific methods
    
    def record_diffusion_step(self, step: int, timestep: float, 
                            block_size: int, computation_time_ms: float,
                            guidance_scale: Optional[float] = None):
        """Record metrics for a diffusion step.
        
        Args:
            step: Diffusion step number
            timestep: Timestep value (0.0 to 1.0)
            block_size: Number of frames per block
            computation_time_ms: Time taken for this step in milliseconds
            guidance_scale: Classifier guidance scale used
        """
        if self.diffusion_analyzer is not None:
            self.diffusion_analyzer.record_diffusion_step(
                step, timestep, block_size, computation_time_ms, guidance_scale
            )
    
    def record_model_parameters(self, model_name: str, parameters_count: int, 
                              model_type: str):
        """Record model parameter count.
        
        Args:
            model_name: Name of the model
            parameters_count: Number of parameters
            model_type: Type of model (diffusion, text_encoder, vae, etc.)
        """
        if self.diffusion_analyzer is not None:
            self.diffusion_analyzer.record_model_parameters(
                model_name, parameters_count, model_type
            )
    
    def record_block_computation(self, block_index: int, block_size: int,
                               computation_time_ms: float, memory_usage_mb: float):
        """Record metrics for a block computation.
        
        Args:
            block_index: Index of the block
            block_size: Number of frames in the block
            computation_time_ms: Time taken for this block in milliseconds
            memory_usage_mb: Peak memory usage during block computation in MB
        """
        if self.diffusion_analyzer is not None:
            self.diffusion_analyzer.record_block_computation(
                block_index, block_size, computation_time_ms, memory_usage_mb
            )
    
    @contextmanager
    def stage(self, stage_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for timing pipeline stages.
        
        Args:
            stage_name: Name of the stage
            metadata: Optional metadata for the stage
        """
        if not self.config.enabled or self.current_session is None:
            yield
            return
        
        stage_info = {
            'name': stage_name,
            'start_time': time.time(),
            'metadata': metadata or {}
        }
        self._stage_stack.append(stage_info)
        
        try:
            yield
        finally:
            if self._stage_stack:
                stage_info = self._stage_stack.pop()
                end_time = time.time()
                duration = end_time - stage_info['start_time']
                
                # Store stage timing
                if stage_name not in self.current_session.stage_timings:
                    self.current_session.stage_timings[stage_name] = {
                        'total_time': 0.0,
                        'call_count': 0,
                        'min_time': float('inf'),
                        'max_time': 0.0,
                        'metadata': []
                    }
                
                timing = self.current_session.stage_timings[stage_name]
                timing['total_time'] += duration  # type: ignore
                timing['call_count'] += 1  # type: ignore
                timing['min_time'] = min(timing['min_time'], duration)  # type: ignore
                timing['max_time'] = max(timing['max_time'], duration)  # type: ignore
                timing['metadata'].append(stage_info['metadata'])  # type: ignore
    
    def profile_function(self, func: Callable, stage_name: Optional[str] = None, 
                        metadata: Optional[Dict[str, Any]] = None):
        """Decorator/wrapper to profile a function.
        
        Args:
            func: Function to profile
            stage_name: Stage name (defaults to function name)
            metadata: Optional metadata
            
        Returns:
            Wrapped function
        """
        stage_name = stage_name or func.__name__
        
        def wrapper(*args, **kwargs):
            with self.stage(stage_name, metadata):
                return func(*args, **kwargs)
        
        return wrapper
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics.
        
        Returns:
            Dictionary with latest metrics from all monitors
        """
        if not self.config.enabled:
            return {}
        
        metrics: Dict[str, Any] = {'timestamp': time.time()}
        
        if self.gpu_monitor:
            gpu_metrics = self.gpu_monitor.get_latest_metrics()
            if gpu_metrics:
                metrics['gpu'] = gpu_metrics.metrics
        
        if self.cpu_monitor:
            cpu_metrics = self.cpu_monitor.get_latest_metrics()
            if cpu_metrics:
                metrics['cpu'] = cpu_metrics.metrics
        
        return metrics
    
    def _start_real_time_display(self):
        """Start real-time metrics display in a separate thread."""
        if self._display_thread is not None:
            return
        
        self._display_stop_event.clear()
        self._display_thread = threading.Thread(
            target=self._real_time_display_loop, 
            daemon=True
        )
        self._display_thread.start()
    
    def _stop_real_time_display(self):
        """Stop real-time metrics display."""
        if self._display_thread is None:
            return
        
        self._display_stop_event.set()
        self._display_thread.join(timeout=self.config.display_interval + 1.0)
        self._display_thread = None
    
    def _real_time_display_loop(self):
        """Main loop for real-time metrics display."""
        while not self._display_stop_event.is_set():
            current_time = time.time()
            
            if current_time - self._last_display_time >= self.config.display_interval:
                self._display_current_metrics()
                self._last_display_time = current_time
            
            self._display_stop_event.wait(1.0)  # Check every second
    
    def _display_current_metrics(self):
        """Display current metrics to console."""
        metrics = self.get_current_metrics()
        if not metrics:
            return
        
        print(f"\n=== Profiling Metrics ({self.current_session.session_id if self.current_session else 'No Session'}) ===")
        
        # GPU metrics
        if 'gpu' in metrics:
            gpu = metrics['gpu']
            print(f"GPU: Memory {gpu.get('total_gpu_memory_used_mb', 0):.0f}MB / "
                  f"{gpu.get('total_gpu_memory_total_mb', 0):.0f}MB "
                  f"({gpu.get('total_gpu_memory_used_mb', 0) / max(gpu.get('total_gpu_memory_total_mb', 1), 1) * 100:.1f}%), "
                  f"Util {gpu.get('max_gpu_utilization', 0):.1f}%, "
                  f"Temp {gpu.get('max_gpu_temperature', 0):.1f}°C")
        
        # CPU metrics
        if 'cpu' in metrics:
            cpu = metrics['cpu']
            print(f"CPU: {cpu.get('cpu_usage_percent', 0):.1f}%, "
                  f"Memory {cpu.get('memory_used_mb', 0):.0f}MB / "
                  f"{cpu.get('memory_total_mb', 0):.0f}MB "
                  f"({cpu.get('memory_usage_percent', 0):.1f}%)")
    
    def _generate_session_report(self, session: ProfilingSession):
        """Generate a comprehensive report for the session."""
        if not self.reporter:
            return
        
        # Collect all data
        gpu_data = self.gpu_monitor.get_all_data() if self.gpu_monitor else []
        cpu_data = self.cpu_monitor.get_all_data() if self.cpu_monitor else []
        
        # Generate report
        report_path = self.reporter.generate_report(
            session=session,
            gpu_data=gpu_data,
            cpu_data=cpu_data
        )
        
        if report_path:
            print(f"Profiling report generated: {report_path}")
    
    def cleanup(self):
        """Clean up resources and end any active sessions."""
        if self.current_session:
            self.end_session()
        
        if self.gpu_monitor:
            self.gpu_monitor.stop_monitoring()
        if self.cpu_monitor:
            self.cpu_monitor.stop_monitoring()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()