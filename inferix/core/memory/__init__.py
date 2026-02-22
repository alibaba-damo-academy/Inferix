"""Memory management utilities for Inferix framework."""

from .utils import (
    DynamicSwapInstaller,
    fake_diffusers_current_device,
    get_cuda_free_memory_gb,
    move_model_to_device_with_memory_preservation,
    offload_model_from_device_for_memory_preservation,
    unload_complete_models,
    load_model_as_complete,
    merge_dict_list,
    cpu,
    gpu,
)

from .manager import (
    AsyncMemoryManager,
    MemoryUnit,
    Granularity,
)

__all__ = [
    # Utils
    "DynamicSwapInstaller",
    "fake_diffusers_current_device",
    "get_cuda_free_memory_gb",
    "move_model_to_device_with_memory_preservation",
    "offload_model_from_device_for_memory_preservation", 
    "unload_complete_models",
    "load_model_as_complete",
    "merge_dict_list",
    "cpu",
    "gpu",
    # Manager
    "AsyncMemoryManager",
    "MemoryUnit",
    "Granularity",
]