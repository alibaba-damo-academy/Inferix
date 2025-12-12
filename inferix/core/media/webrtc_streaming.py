"""WebRTC Streaming Backend using fastrtc

This module provides WebRTC streaming functionality using the fastrtc library.
Currently experimental - use Gradio backend for more stable streaming.
"""

from fastrtc import Stream
import numpy as np
import torch
import time
from typing import Optional
import queue
import threading

from inferix.core.media.streaming_backend import StreamingBackend


class WebRTCStreamingBackend(StreamingBackend):
    """WebRTC streaming backend using fastrtc library.
    
    Note: This is experimental. For stable streaming, use GradioStreamingBackend.
    """
    
    def __init__(self):
        super().__init__()
        self.stream: Optional[Stream] = None
        self.frame_queue = queue.Queue()
        self.server_thread = None

    def connect(self, width: int, height: int, fps: int = 16, host="0.0.0.0", port=8000, **kwargs) -> bool:
        """Connect WebRTC stream using fastrtc.
        
        Args:
            width: Video width (not used in current implementation)
            height: Video height (not used in current implementation)
            fps: Frames per second (not used in current implementation)
            host: Server host
            port: Server port
            **kwargs: Additional parameters (ignored)
            
        Returns:
            True if connection successful
        """
        def frame_generator():
            while self.running:
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                    if frame is None:
                        break
                    yield frame
                except queue.Empty:
                    continue

        try:
            self.stream = Stream(
                handler=frame_generator,
                modality="video",
                mode="receive"
            )
            
            def run_server():
                self.stream.ui.launch(server_name=host, server_port=port)
            
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            
            self.running = True
            self.is_connected = True
            
            print(f"✅ WebRTC server started")
            print(f"   Access: http://{host}:{port}")
            return True
            
        except Exception as e:
            print(f"❌ WebRTC connection failed: {e}")
            return False

    def stream_batch(self, x: torch.Tensor) -> bool:
        """Stream a batch of frames via WebRTC.
        
        Args:
            x: Tensor of shape [T, H, W, C], dtype uint8, range [0, 255]
            
        Returns:
            True if streaming successful
        """
        if not self.running:
            print("⚠️  WebRTC not running, cannot stream")
            return False

        x = x.cpu().numpy()
        T = x.shape[0]

        for i in range(T):
            frame = x[i]
            try:
                self.frame_queue.put(frame, timeout=1.0)
            except queue.Full:
                print("⚠️  WebRTC queue full")
                return False
        return True

    def disconnect(self):
        """Disconnect WebRTC stream."""
        self.running = False
        self.is_connected = False
        try:
            self.frame_queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        print("✅ WebRTC disconnected")