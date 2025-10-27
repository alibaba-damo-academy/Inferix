from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class ProfilingConfig:
    """Configuration for profiling system."""
    
    # Enable/disable profiling
    enabled: bool = False
    
    # Monitoring intervals (in seconds)
    gpu_monitor_interval: float = 0.5
    cpu_monitor_interval: float = 1.0
    
    # GPU monitoring options
    monitor_gpu_memory: bool = True
    monitor_gpu_utilization: bool = True
    monitor_gpu_temperature: bool = True
    monitor_gpu_power: bool = True
    
    # CPU monitoring options (minimal)
    monitor_cpu_usage: bool = True
    monitor_system_memory: bool = True
    
    # Reporting options
    real_time_display: bool = True
    display_interval: float = 5.0  # seconds
    generate_final_report: bool = True
    save_raw_data: bool = False
    
    # Output settings
    output_dir: Optional[str] = None
    report_format: str = "both"  # "html", "json", "both"
    
    # Advanced options
    max_data_points: int = 10000  # Prevent excessive memory usage
    profile_inference_steps: bool = True
    profile_vae_decode: bool = True
    profile_text_encoding: bool = True
    
    # Diffusion model specific profiling
    profile_diffusion_steps: bool = True
    profile_block_computation: bool = True
    profile_model_parameters: bool = True
    
    # Custom tags for profiling sessions
    session_tags: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.gpu_monitor_interval <= 0:
            raise ValueError("gpu_monitor_interval must be positive")
        if self.cpu_monitor_interval <= 0:
            raise ValueError("cpu_monitor_interval must be positive")
        if self.display_interval <= 0:
            raise ValueError("display_interval must be positive")
        if self.max_data_points <= 0:
            raise ValueError("max_data_points must be positive")
        if self.report_format not in ["html", "json", "both"]:
            raise ValueError("report_format must be 'html', 'json', or 'both'")