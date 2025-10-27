import time
import threading
import psutil
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import deque

try:
    import pynvml  # type: ignore
    NVML_AVAILABLE = True
except ImportError:
    pynvml = None  # type: ignore
    NVML_AVAILABLE = False

try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False


@dataclass
class MetricSnapshot:
    """A snapshot of system metrics at a specific timestamp."""
    timestamp: float
    metrics: Dict[str, Any]
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


class BaseMonitor(ABC):
    """Base class for system monitors."""
    
    def __init__(self, interval: float = 1.0, max_data_points: int = 10000):
        self.interval = interval
        self.max_data_points = max_data_points
        self.data_points: deque = deque(maxlen=max_data_points)
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    @abstractmethod
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect current metrics. Must be implemented by subclasses."""
        pass
    
    def start_monitoring(self):
        """Start monitoring in a separate thread."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self._stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring and join the thread."""
        if not self.is_monitoring:
            return
        
        self._stop_event.set()
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=self.interval * 2)
    
    def _monitor_loop(self):
        """Main monitoring loop running in separate thread."""
        while not self._stop_event.is_set():
            try:
                metrics = self.collect_metrics()
                snapshot = MetricSnapshot(time.time(), metrics)
                self.data_points.append(snapshot)
            except Exception as e:
                print(f"Error collecting metrics in {self.__class__.__name__}: {e}")
            
            self._stop_event.wait(self.interval)
    
    def get_latest_metrics(self) -> Optional[MetricSnapshot]:
        """Get the most recent metrics snapshot."""
        return self.data_points[-1] if self.data_points else None
    
    def get_all_data(self) -> List[MetricSnapshot]:
        """Get all collected data points."""
        return list(self.data_points)
    
    def clear_data(self):
        """Clear all collected data points."""
        self.data_points.clear()


class GPUMonitor(BaseMonitor):
    """Monitor GPU performance metrics using NVML and PyTorch."""
    
    def __init__(self, interval: float = 0.5, max_data_points: int = 10000, 
                 device_ids: Optional[List[int]] = None):
        super().__init__(interval, max_data_points)
        self.device_ids = device_ids or []
        self.nvml_initialized = False
        
        # Initialize NVML if available
        if NVML_AVAILABLE and pynvml is not None:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                # Auto-detect devices if not specified
                if not self.device_ids:
                    device_count = pynvml.nvmlDeviceGetCount()
                    self.device_ids = list(range(device_count))
            except Exception as e:
                print(f"Failed to initialize NVML: {e}")
                self.nvml_initialized = False
        
        # Fallback to PyTorch for basic GPU info
        if not self.nvml_initialized and TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
            if not self.device_ids:
                self.device_ids = list(range(torch.cuda.device_count()))
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect GPU metrics from all monitored devices."""
        metrics = {
            'gpu_devices': {},
            'total_gpu_memory_used_mb': 0,
            'total_gpu_memory_total_mb': 0,
            'max_gpu_utilization': 0,
            'max_gpu_temperature': 0,
            'total_gpu_power_watts': 0
        }
        
        for device_id in self.device_ids:
            device_metrics = self._collect_device_metrics(device_id)
            metrics['gpu_devices'][f'gpu_{device_id}'] = device_metrics
            
            # Aggregate metrics
            if 'memory_used_mb' in device_metrics:
                metrics['total_gpu_memory_used_mb'] += device_metrics['memory_used_mb']
            if 'memory_total_mb' in device_metrics:
                metrics['total_gpu_memory_total_mb'] += device_metrics['memory_total_mb']
            if 'utilization_percent' in device_metrics:
                metrics['max_gpu_utilization'] = max(
                    metrics['max_gpu_utilization'], 
                    device_metrics['utilization_percent']
                )
            if 'temperature_celsius' in device_metrics:
                metrics['max_gpu_temperature'] = max(
                    metrics['max_gpu_temperature'], 
                    device_metrics['temperature_celsius']
                )
            if 'power_watts' in device_metrics:
                metrics['total_gpu_power_watts'] += device_metrics['power_watts']
        
        return metrics
    
    def _collect_device_metrics(self, device_id: int) -> Dict[str, Any]:
        """Collect metrics for a specific GPU device."""
        device_metrics = {'device_id': device_id}
        
        # Try NVML first for comprehensive metrics
        if self.nvml_initialized and pynvml is not None:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                
                # Memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                device_metrics['memory_used_mb'] = mem_info.used // (1024 * 1024)
                device_metrics['memory_total_mb'] = mem_info.total // (1024 * 1024)
                device_metrics['memory_free_mb'] = mem_info.free // (1024 * 1024)
                device_metrics['memory_utilization_percent'] = (mem_info.used / mem_info.total) * 100
                
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                device_metrics['utilization_percent'] = util.gpu
                device_metrics['memory_util_percent'] = util.memory
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                device_metrics['temperature_celsius'] = temp
                
                # Power consumption
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    device_metrics['power_watts'] = power
                except:
                    pass  # Power monitoring not available on all GPUs
                
                # Clock speeds
                try:
                    graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                    memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                    device_metrics['graphics_clock_mhz'] = graphics_clock
                    device_metrics['memory_clock_mhz'] = memory_clock
                except:
                    pass
                
                # Device name
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                device_metrics['device_name'] = str(name)  # type: ignore
                
            except Exception as e:
                print(f"Error collecting NVML metrics for GPU {device_id}: {e}")
        
        # Fallback to PyTorch metrics
        if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available() and device_id < torch.cuda.device_count():
            try:
                # PyTorch memory info
                with torch.cuda.device(device_id):
                    memory_allocated = torch.cuda.memory_allocated(device_id)
                    memory_reserved = torch.cuda.memory_reserved(device_id)
                    
                device_metrics['torch_memory_allocated_mb'] = memory_allocated // (1024 * 1024)
                device_metrics['torch_memory_reserved_mb'] = memory_reserved // (1024 * 1024)
                
                # Device properties
                props = torch.cuda.get_device_properties(device_id)
                device_metrics['torch_device_name'] = props.name
                device_metrics['torch_total_memory_mb'] = props.total_memory // (1024 * 1024)
                
            except Exception as e:
                print(f"Error collecting PyTorch metrics for GPU {device_id}: {e}")
        
        return device_metrics


class CPUMonitor(BaseMonitor):
    """Monitor minimal CPU and system memory metrics using psutil."""
    
    def __init__(self, interval: float = 1.0, max_data_points: int = 10000):
        super().__init__(interval, max_data_points)
        
        # Check if psutil is available
        try:
            psutil.cpu_percent()  # Test call
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CPU monitoring: {e}")
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect CPU and system memory metrics."""
        metrics = {}
        
        try:
            # CPU usage (non-blocking)
            cpu_percent = psutil.cpu_percent(interval=None)
            metrics['cpu_usage_percent'] = cpu_percent
            
            # Per-core CPU usage
            cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
            metrics['cpu_per_core_percent'] = cpu_per_core
            metrics['cpu_core_count'] = len(cpu_per_core)
            
            # System memory
            memory = psutil.virtual_memory()
            metrics['memory_total_mb'] = memory.total // (1024 * 1024)
            metrics['memory_used_mb'] = memory.used // (1024 * 1024)
            metrics['memory_available_mb'] = memory.available // (1024 * 1024)
            metrics['memory_usage_percent'] = memory.percent
            
            # CPU frequency
            try:
                freq = psutil.cpu_freq()
                if freq:
                    metrics['cpu_freq_current_mhz'] = freq.current
                    metrics['cpu_freq_max_mhz'] = freq.max
            except:
                pass  # CPU frequency not available on all systems
            
            # Load average (Unix systems only)
            try:
                load = psutil.getloadavg()
                metrics['load_average_1min'] = load[0]
                metrics['load_average_5min'] = load[1]
                metrics['load_average_15min'] = load[2]
            except:
                pass  # Load average not available on Windows
            
        except Exception as e:
            print(f"Error collecting CPU metrics: {e}")
        
        return metrics