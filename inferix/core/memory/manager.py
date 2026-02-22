"""
Async Memory Manager for dynamic model component offload/reload.

This module provides a unified interface for managing GPU memory across model components
with support for:
- Async offload/reload using CUDA streams
- Component-level and layer-level granularity
- LRU-based eviction strategy
- Prefetch and exclusive mode operations

Requirements:
- Requires >= 24GB VRAM for effective use
- On smaller GPUs, DynamicSwapInstaller (lazy parameter-level swap) is more efficient

Note: This component provides explicit component-level control, while DynamicSwapInstaller
provides transparent parameter-level lazy loading. Choose based on your use case:
- AsyncMemoryManager: Explicit control, prefetch support, suitable for >= 24GB VRAM
- DynamicSwapInstaller: Transparent lazy loading, suitable for memory-constrained GPUs
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union, Iterator
import threading
import torch
import torch.nn as nn


class Granularity(Enum):
    """Memory management granularity level."""
    COMPONENT = "component"  # Entire model (Generator, VAE)
    LAYER = "layer"          # Single layer (Block_0, Block_1)
    TENSOR = "tensor"        # Individual tensor (for future use)


@dataclass
class MemoryUnit:
    """Represents a manageable memory unit (component, layer, or tensor)."""
    name: str
    module: Union[nn.Module, torch.Tensor]
    size_bytes: int
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    
    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024**3)
    
    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024**2)


class AsyncMemoryManager:
    """
    Async memory manager for dynamic model component offload/reload.
    
    Features:
    - Async offload/reload using CUDA streams
    - Component-level and layer-level granularity
    - LRU-based automatic eviction
    - Prefetch support for overlapping data transfer with computation
    - Exclusive mode for memory-constrained operations
    
    Example:
        ```python
        mm = AsyncMemoryManager(device=torch.device('cuda:0'))
        mm.register_component('generator', generator, layer_names=['blocks'])
        mm.register_component('vae', vae)
        
        # Basic usage
        with mm.use('generator'):
            latent = generator(noise)
        
        # Prefetch optimization
        mm.prefetch('vae')
        latent = generator(noise)  # VAE loading in background
        with mm.use('vae'):
            video = vae.decode(latent)
        
        # Exclusive mode for memory-constrained ops
        with mm.exclusive('vae'):
            video = vae.decode(latent)
        ```
    """
    
    def __init__(
        self,
        device: torch.device,
        budget_gb: Optional[float] = None,
        default_granularity: Granularity = Granularity.COMPONENT
    ):
        """
        Initialize the memory manager.
        
        Args:
            device: Target GPU device
            budget_gb: Memory budget in GB (auto-detected if None)
            default_granularity: Default granularity for operations
        """
        self.device = device
        self.cpu_device = torch.device('cpu')
        self.budget_gb = budget_gb or self._detect_budget()
        self.default_granularity = default_granularity
        
        # Memory unit registry
        self.units: Dict[str, MemoryUnit] = {}
        self.unit_devices: Dict[str, torch.device] = {}
        
        # Async scheduling
        self.memory_stream = torch.cuda.Stream(device=device)
        self.pending_ops: Dict[str, torch.cuda.Event] = {}
        
        # LRU tracking
        self._access_order: List[str] = []
        self._lock = threading.Lock()
    
    def _detect_budget(self) -> float:
        """Auto-detect memory budget based on device properties."""
        props = torch.cuda.get_device_properties(self.device)
        return props.total_memory / (1024**3) * 0.85  # 85% of total
    
    # ========== Registration API ==========
    
    def register_component(
        self,
        name: str,
        module: nn.Module,
        layer_names: Optional[List[str]] = None
    ) -> None:
        """
        Register a model component for memory management.
        
        Args:
            name: Unique identifier for the component (e.g., 'generator')
            module: PyTorch module to manage
            layer_names: Optional list of attribute names to register as sub-layers
                        (e.g., ['blocks'] to register model.blocks[0], model.blocks[1], ...)
        """
        size = self._compute_size(module)
        children = []
        
        # Register sub-layers if specified
        if layer_names:
            for attr_name in layer_names:
                if hasattr(module, attr_name):
                    layers = getattr(module, attr_name)
                    if isinstance(layers, nn.ModuleList):
                        for i, layer in enumerate(layers):
                            layer_full_name = f"{name}.{attr_name}.{i}"
                            self._register_unit(layer_full_name, layer, parent=name)
                            children.append(layer_full_name)
                    elif isinstance(layers, nn.Module):
                        layer_full_name = f"{name}.{attr_name}"
                        self._register_unit(layer_full_name, layers, parent=name)
                        children.append(layer_full_name)
        
        self.units[name] = MemoryUnit(
            name=name,
            module=module,
            size_bytes=size,
            children=children
        )
        self.unit_devices[name] = self._get_module_device(module)
    
    def _register_unit(self, name: str, module: nn.Module, parent: str) -> None:
        """Register a single memory unit."""
        size = self._compute_size(module)
        self.units[name] = MemoryUnit(
            name=name,
            module=module,
            size_bytes=size,
            parent=parent
        )
        self.unit_devices[name] = self._get_module_device(module)
    
    def unregister(self, name: str) -> None:
        """Unregister a component and its children."""
        if name in self.units:
            unit = self.units[name]
            # Remove children first
            for child_name in unit.children:
                if child_name in self.units:
                    del self.units[child_name]
                if child_name in self.unit_devices:
                    del self.unit_devices[child_name]
            # Remove the component
            del self.units[name]
            if name in self.unit_devices:
                del self.unit_devices[name]
    
    # ========== Async Operations API ==========
    
    def load_async(
        self,
        name: str,
        granularity: Optional[Granularity] = None,
        _log: bool = True
    ) -> torch.cuda.Event:
        """
        Asynchronously load a component to GPU.
        
        Args:
            name: Component name to load
            granularity: Override default granularity
            _log: Whether to log the operation (internal use)
            
        Returns:
            CUDA Event that can be used to synchronize
        """
        granularity = granularity or self.default_granularity
        unit = self.units[name]
        
        if _log:
            print(f"  [MemoryManager] Loading '{name}' to GPU ({unit.size_gb:.2f} GB)")
        
        with torch.cuda.stream(self.memory_stream):
            if granularity == Granularity.LAYER and unit.children:
                # Layer-level loading
                for child_name in unit.children:
                    self._move_to_device(child_name, self.device)
            else:
                # Component-level loading
                self._move_to_device(name, self.device)
            
            # Record completion event
            event = torch.cuda.Event()
            event.record(self.memory_stream)
            self.pending_ops[name] = event
        
        self._update_access(name)
        return event
    
    def offload_async(self, name: str, _log: bool = True) -> torch.cuda.Event:
        """
        Asynchronously offload a component to CPU.
        
        Args:
            name: Component name to offload
            _log: Whether to log the operation (internal use)
            
        Returns:
            CUDA Event that can be used to synchronize
        """
        unit = self.units[name]
        
        if _log:
            print(f"  [MemoryManager] Offloading '{name}' to CPU ({unit.size_gb:.2f} GB)")
        
        with torch.cuda.stream(self.memory_stream):
            if unit.children:
                for child_name in unit.children:
                    self._move_to_device(child_name, self.cpu_device)
            else:
                self._move_to_device(name, self.cpu_device)
            
            event = torch.cuda.Event()
            event.record(self.memory_stream)
            self.pending_ops[name] = event
        
        return event
    
    def wait(self, name: str) -> None:
        """Wait for a specific async operation to complete."""
        if name in self.pending_ops:
            self.pending_ops[name].synchronize()
            del self.pending_ops[name]
    
    def wait_all(self) -> None:
        """Wait for all pending async operations to complete."""
        self.memory_stream.synchronize()
        self.pending_ops.clear()
    
    # ========== Prefetch API ==========
    
    def prefetch(self, name: str) -> None:
        """
        Prefetch a component in background without blocking.
        
        Use this to overlap data transfer with computation:
        
            mm.prefetch('vae')           # Start loading VAE
            latent = generator(noise)    # Do computation
            with mm.use('vae'):          # VAE may already be loaded
                video = vae.decode(latent)
        """
        if self.unit_devices.get(name) != self.device:
            self.load_async(name)
    
    # ========== Context Manager API ==========
    
    @contextmanager
    def use(
        self,
        *names: str,
        granularity: Optional[Granularity] = None,
        wait: bool = True
    ) -> Iterator[None]:
        """
        Context manager to ensure components are on GPU.
        
        Args:
            names: Component names to load
            granularity: Override default granularity
            wait: Whether to wait for loading to complete before yielding
        """
        events = []
        
        for name in names:
            if self.unit_devices.get(name) != self.device:
                # Ensure space is available
                self._ensure_space(self.units[name].size_gb)
                # Start async loading
                event = self.load_async(name, granularity)
                events.append((name, event))
        
        if wait:
            for name, event in events:
                event.synchronize()
        
        try:
            yield
        finally:
            pass  # Optional: auto-offload after use
    
    @contextmanager
    def exclusive(self, name: str) -> Iterator[None]:
        """
        Exclusive mode: offload all other components to make room.
        
        Use this for memory-constrained operations like VAE decode:
        
            with mm.exclusive('vae'):
                video = vae.decode(latent)  # Maximum memory for VAE
        """
        print(f"[MemoryManager] Entering EXCLUSIVE mode for '{name}'")
        
        # Record current state for potential restoration
        original_on_gpu = [
            n for n in self.units
            if self.unit_devices.get(n) == self.device and n != name
        ]
        
        # Offload all other components
        if original_on_gpu:
            print(f"[MemoryManager] Offloading {len(original_on_gpu)} component(s) to make room")
            for other_name in original_on_gpu:
                self.offload_async(other_name, _log=False)
            print(f"  [MemoryManager] Offloaded: {', '.join(original_on_gpu)}")
        
        self.wait_all()
        torch.cuda.empty_cache()
        
        # Load target component
        if self.unit_devices.get(name) != self.device:
            self.load_async(name, _log=False)
            self.wait(name)
            print(f"  [MemoryManager] Loaded '{name}' ({self.units[name].size_gb:.2f} GB)")
        
        # Show memory status
        mem_info = self.get_memory_info()
        print(f"[MemoryManager] Memory: {mem_info['free_gb']:.1f} GB free / {mem_info['total_gb']:.1f} GB total")
        
        try:
            yield
        finally:
            print(f"[MemoryManager] Exiting EXCLUSIVE mode for '{name}'")
    
    # ========== Layer-Level API ==========
    
    def load_layers(
        self,
        component: str,
        layer_indices: Union[List[int], range]
    ) -> None:
        """
        Load specific layers of a component (for large model pipeline).
        
        Args:
            component: Component name
            layer_indices: Indices of layers to load
        """
        unit = self.units.get(component)
        if not unit or not unit.children:
            raise ValueError(f"{component} has no registered layers")
        
        for idx in layer_indices:
            if idx < len(unit.children):
                layer_name = unit.children[idx]
                self.load_async(layer_name)
    
    def offload_layers(
        self,
        component: str,
        layer_indices: Union[List[int], range]
    ) -> None:
        """
        Offload specific layers of a component.
        
        Args:
            component: Component name
            layer_indices: Indices of layers to offload
        """
        unit = self.units.get(component)
        if not unit or not unit.children:
            return
        
        for idx in layer_indices:
            if idx < len(unit.children):
                layer_name = unit.children[idx]
                self.offload_async(layer_name)
    
    # ========== Status API ==========
    
    def get_status(self) -> Dict[str, Dict]:
        """Get current status of all registered components."""
        status = {}
        for name, unit in self.units.items():
            if unit.parent is None:  # Only top-level components
                device = self.unit_devices.get(name, 'unknown')
                status[name] = {
                    'device': str(device),
                    'size_gb': round(unit.size_gb, 2),
                    'num_layers': len(unit.children) if unit.children else 0,
                    'pending': name in self.pending_ops
                }
        return status
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory information."""
        props = torch.cuda.get_device_properties(self.device)
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        
        return {
            'total_gb': props.total_memory / (1024**3),
            'allocated_gb': allocated / (1024**3),
            'reserved_gb': reserved / (1024**3),
            'free_gb': (props.total_memory - allocated) / (1024**3),
            'budget_gb': self.budget_gb
        }
    
    # ========== Internal Methods ==========
    
    def _move_to_device(self, name: str, device: torch.device) -> None:
        """Move a unit to the specified device."""
        unit = self.units[name]
        if isinstance(unit.module, nn.Module):
            unit.module.to(device, non_blocking=True)
        elif isinstance(unit.module, torch.Tensor):
            unit.module.data = unit.module.data.to(device, non_blocking=True)
        self.unit_devices[name] = device
    
    def _ensure_space(self, required_gb: float) -> None:
        """Ensure enough GPU memory is available using LRU eviction."""
        free_gb = self._get_free_memory()
        
        with self._lock:
            while free_gb < required_gb and self._access_order:
                lru_name = self._access_order[0]
                if self.unit_devices.get(lru_name) == self.device:
                    self.offload_async(lru_name)
                    self.wait(lru_name)
                    torch.cuda.empty_cache()
                    free_gb = self._get_free_memory()
                self._access_order.pop(0)
    
    def _update_access(self, name: str) -> None:
        """Update LRU access order."""
        with self._lock:
            if name in self._access_order:
                self._access_order.remove(name)
            self._access_order.append(name)
    
    def _get_free_memory(self) -> float:
        """Get free GPU memory in GB."""
        props = torch.cuda.get_device_properties(self.device)
        allocated = torch.cuda.memory_allocated(self.device)
        return (props.total_memory - allocated) / (1024**3)
    
    def _compute_size(self, module: nn.Module) -> int:
        """Compute total size of module parameters in bytes."""
        total = 0
        for p in module.parameters():
            total += p.numel() * p.element_size()
        for b in module.buffers():
            total += b.numel() * b.element_size()
        return total
    
    def _get_module_device(self, module: nn.Module) -> torch.device:
        """Get the device of a module."""
        try:
            return next(module.parameters()).device
        except StopIteration:
            return self.cpu_device
