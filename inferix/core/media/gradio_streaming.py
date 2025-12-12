"""Gradio-based streaming backend (default, highest priority)."""
from inferix.core.media.streaming_backend import StreamingBackend
import numpy as np
import torch
import time
from typing import Optional, Callable
import queue
import threading
import gradio as gr


class GradioStreamingBackend(StreamingBackend):
    """Gradio-based streaming with auto-refresh UI (default backend)."""
    
    def __init__(self):
        super().__init__()
        self.stream = None
        self.frame_queue = queue.Queue()
        self.running = False
        self.server_thread = None
        self.video_buffer = []
        self.loop_playback = False
        self.current_frame_idx = 0
        self.input_callback = None

    def connect(self, width: int, height: int, fps: int = 16, host="0.0.0.0", port=8000, loop=True, input_callback=None, **kwargs) -> bool:
        """Connect Gradio streaming server
        
        Args:
            width: Video width
            height: Video height
            fps: Frames per second  
            host: Server host
            port: Server port
            loop: Enable loop playback
            input_callback: Optional callback for user input
        """
        self.loop_playback = loop
        self.input_callback = input_callback
        self.running = True
        self.current_frame_idx = 0
        
        print(f"🎞️  Initializing video stream ({width}x{height} @ {fps}fps)...")
        
        # Create Gradio interface with streaming generator
        with gr.Blocks() as demo:
            gr.Markdown("# 🎥 Real-time Video Generation")
            
            with gr.Row():
                image_display = gr.Image(label="Current Frame", type="numpy", height=480, streaming=True)
            
            with gr.Row():
                status_text = gr.Textbox(label="Status", value="Waiting for frames...")
            
            # Frame generator for streaming
            def frame_generator():
                """Generator that yields frames continuously"""
                frame_count = 0
                while self.running:
                    try:
                        # Try to get new frame from queue (non-blocking)
                        frame = self.frame_queue.get(timeout=0.1)
                        if frame is not None:
                            self.video_buffer.append(frame.copy())
                            # Convert to uint8 for display
                            if frame.dtype != np.uint8:
                                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                            frame_count += 1
                            yield frame
                    except queue.Empty:
                        # No new frames, loop existing buffer
                        if self.video_buffer:
                            idx = self.current_frame_idx % len(self.video_buffer)
                            frame = self.video_buffer[idx]
                            self.current_frame_idx += 1
                            # Convert to uint8 for display
                            if frame.dtype != np.uint8:
                                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                            yield frame
                        else:
                            # Yield black frame while waiting
                            yield np.zeros((height, width, 3), dtype=np.uint8)
                    
                    time.sleep(1.0/fps)
            
            # Dummy input to trigger streaming
            start_btn = gr.Button("Start Streaming", visible=False)
            start_btn.click(fn=frame_generator, outputs=image_display)
            
            # Auto-start on load
            demo.load(fn=lambda: None, outputs=start_btn)
        
        self.stream = demo
        
        def run_server():
            try:
                print("🚀 Starting Gradio server...")
                # Launch with proper network config for WSL
                demo.launch(
                    server_name="0.0.0.0",  # Listen on all interfaces
                    server_port=port,
                    share=False,
                    quiet=False,
                    # Enable CORS for WSL access from Windows browser
                    allowed_paths=[],
                    # Disable unnecessary features for performance
                    favicon_path=None
                )
            except Exception as e:
                print(f"❌ Failed to launch: {e}")
                import traceback
                traceback.print_exc()
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        time.sleep(2.0)
        
        if self.server_thread.is_alive():
            self.is_connected = True
            # Provide WSL-specific access instructions
            print(f"✅ Server ready!")
            print(f"   🌐 Local access: http://localhost:{port}")
            
            # Try to detect WSL and provide Windows access URL
            try:
                import subprocess
                result = subprocess.run(
                    ['hostname', '-I'],
                    capture_output=True,
                    text=True,
                    timeout=1
                )
                if result.returncode == 0:
                    wsl_ip = result.stdout.strip().split()[0]
                    print(f"   🖥️  WSL access (from Windows): http://{wsl_ip}:{port}")
            except Exception:
                pass
            
            return True
        else:
            print("❌ Server failed to start")
            return False

    def stream_batch(self, x: torch.Tensor):
        if not self.running:
            print("⚠️  Gradio streaming not running, cannot stream")
            return False

        x = x.cpu().numpy()
        T = x.shape[0]
        
        print(f"📤 Streaming {T} frames (buffer size: {len(self.video_buffer)})")

        for i in range(T):
            frame = x[i]
            # Store frame in buffer for looping
            self.video_buffer.append(frame.copy())
            try:
                self.frame_queue.put(frame, timeout=1.0)
            except queue.Full:
                print("⚠️  Frame queue full")
                return False
        
        print(f"✅ Streamed {T} frames successfully (total buffered: {len(self.video_buffer)})")
        return True

    def disconnect(self):
        # Signal end of stream (triggers looping if enabled)
        try:
            self.frame_queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        # Don't set running=False immediately to allow looping
    
    def stop(self):
        """Completely stop the stream (no looping)"""
        self.running = False
        try:
            self.frame_queue.put(None, timeout=1.0)
        except queue.Full:
            pass
    
    def get_ui(self):
        """Get the Gradio UI instance for customization
        
        Returns:
            Gradio Blocks instance that can be extended with additional components
            
        Example:
            >>> streamer = GradioStreamingBackend()
            >>> streamer.connect(...)
            >>> ui = streamer.get_ui()
            >>> # Add custom components to ui
            >>> with ui:
            >>>     gr.Textbox(label="Prompt")
        """
        if self.stream is None:
            raise RuntimeError("Stream not connected. Call connect() first.")
        return self.stream.ui
    
    def set_additional_inputs(self, components):
        """Add custom input components to the Gradio UI
        
        Args:
            components: List of Gradio components to add as inputs
            
        Example:
            >>> import gradio as gr
            >>> streamer.set_additional_inputs([
            >>>     gr.Textbox(label="Prompt"),
            >>>     gr.Slider(0, 1, label="Strength")
            >>> ])
        """
        if self.stream is None:
            raise RuntimeError("Stream not connected. Call connect() first.")
        # This would require recreating the Stream with additional_inputs
        # For now, just document the pattern
        raise NotImplementedError(
            "To add custom inputs, pass them when creating Stream.\n"
            "Example: Stream(handler=..., additional_inputs=[gr.Textbox(), ...])"
        )