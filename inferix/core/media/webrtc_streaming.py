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
        self.last_frame = None  # For keep-alive mechanism

    def connect(self, width: int, height: int, fps: int = 16, host="0.0.0.0", port=8000, 
                 frame_timeout: float = 10.0, **kwargs) -> bool:
        """Connect WebRTC stream using fastrtc.
        
        Args:
            width: Video width
            height: Video height
            fps: Frames per second for keep-alive mechanism
            host: Server host
            port: Server port
            frame_timeout: Timeout for waiting new frames (seconds)
                          Should be >= max inference time per block
                          Consumer GPU: 5-30s, Datacenter GPU: 3-5s
            **kwargs: Additional parameters (ignored)
            
        Returns:
            True if connection successful
        """
        # [Critical] Set to True before thread start to avoid race condition
        self.running = True
        
        def frame_generator():
            """Generator with first-frame blocking and keep-alive mechanism"""
            print("⏳ WebRTC generator started, waiting for first frame...")
            
            # 1. Block and wait for first frame to avoid timeout before inference starts
            try:
                first_frame = self.frame_queue.get(block=True, timeout=None)
                if first_frame is None:
                    print("🛑 Received stop signal before first frame.")
                    return
                
                # Ensure uint8 format
                if first_frame.dtype != np.uint8:
                    first_frame = (np.clip(first_frame, 0, 1) * 255).astype(np.uint8)
                
                self.last_frame = first_frame
                yield first_frame
                
            except Exception as e:
                print(f"❌ Error waiting for first frame: {e}")
                return
            
            # 2. Continuous streaming
            while self.running:
                try:
                    # Long timeout to avoid frequent keep-alive triggers
                    frame = self.frame_queue.get(timeout=frame_timeout)
                    
                    if frame is None:  # Stop signal received
                        print("🛑 WebRTC stream received stop signal.")
                        break
                    
                    # Ensure uint8 format
                    if frame.dtype != np.uint8:
                        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                    
                    self.last_frame = frame
                    yield frame
                    
                except queue.Empty:
                    # No new frame after timeout, yield last frame for keep-alive
                    print(f"⏱️  No new frame for {frame_timeout}s, holding last frame...")
                    if self.last_frame is not None:
                        yield self.last_frame
                        time.sleep(1.0/fps)  # Simulate frame rate
                    else:
                        continue

        try:
            self.stream = Stream(
                handler=frame_generator,
                modality="video",
                mode="receive"
            )
            
            def run_server():
                print(f"🚀 Starting WebRTC server on {host}:{port}...")
                # Note: WebRTC must run on localhost or HTTPS
                self.stream.ui.launch(
                    server_name=host, 
                    server_port=port,
                    ssl_verify=False
                )
            
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            
            self.is_connected = True
            
            print(f"✅ WebRTC server started")
            print(f"   ⚠️  IMPORTANT: WebRTC works ONLY on localhost or HTTPS.")
            print(f"   👉 Local: http://localhost:{port}")
            if host == "0.0.0.0":
                print(f"   👉 Remote: You MUST set up SSL or use SSH Tunneling")
                print(f"      Example: ssh -L {port}:localhost:{port} user@server")
            return True
            
        except Exception as e:
            print(f"❌ WebRTC connection failed: {e}")
            self.running = False
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
            # Ensure uint8 format, otherwise fastrtc may fail
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                
            try:
                self.frame_queue.put(frame, timeout=2.0)
            except queue.Full:
                print("⚠️  WebRTC queue full, dropping frames")
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