"""Streaming backend abstraction for Inferix framework.

Priority: Gradio > WebRTC > RTMP
"""
from abc import ABC, abstractmethod
from typing import Optional, Callable
import torch


class StreamingBackend(ABC):
    """Abstract base class for streaming backends."""
    
    def __init__(self):
        self.is_connected = False
    
    @abstractmethod
    def connect(self, width: int, height: int, fps: int = 16, **kwargs) -> bool:
        """Establish connection and start streaming service.
        
        Args:
            width: Video width
            height: Video height
            fps: Frames per second
            **kwargs: Backend-specific parameters
            
        Returns:
            True if connection successful
        """
        pass
    
    @abstractmethod
    def stream_batch(self, frames: torch.Tensor) -> bool:
        """Stream a batch of frames.
        
        Args:
            frames: Tensor of shape [T, H, W, C], dtype uint8, range [0, 255]
            
        Returns:
            True if streaming successful
        """
        pass
    
    @abstractmethod
    def disconnect(self):
        """Close connection and cleanup resources."""
        pass
    
    def stop(self):
        """Alias for disconnect (for compatibility)."""
        self.disconnect()


def create_streaming_backend(backend: str = "gradio", **kwargs) -> StreamingBackend:
    """Factory function to create streaming backend.
    
    Args:
        backend: Backend type ("gradio", "webrtc", "rtmp")
        **kwargs: Backend-specific initialization parameters
        
    Returns:
        Streaming backend instance
        
    Example:
        >>> streamer = create_streaming_backend("gradio")
        >>> streamer.connect(width=832, height=480, fps=16, port=8000)
        >>> streamer.stream_batch(frames)
    """
    backend = backend.lower()
    
    if backend == "gradio":
        from inferix.core.media.gradio_streaming import GradioStreamingBackend
        return GradioStreamingBackend(**kwargs)
    elif backend == "webrtc":
        from inferix.core.media.webrtc_streaming import WebRTCStreamingBackend
        return WebRTCStreamingBackend(**kwargs)
    elif backend == "rtmp":
        from inferix.core.media.rtmp_streaming import RTMPStreamingBackend
        return RTMPStreamingBackend(**kwargs)
    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Supported backends: gradio, webrtc, rtmp"
        )
