"""Interactive Gradio Backend for Inferix framework.

This module provides an interactive Gradio interface for video generation,
supporting:
- Real-time video preview (streaming)
- User input collection (prompt, guidance)
- Control buttons (Pause/Resume/Stop)
- Status display (progress, ETA, queued inputs)

Usage:
    from inferix.core.interactive import InteractiveSession
    from inferix.core.media.interactive_gradio import InteractiveGradioBackend
    
    session = InteractiveSession()
    session.set_initial_prompt("A cat walking")
    
    backend = InteractiveGradioBackend(session)
    backend.connect(width=832, height=480, fps=16, port=8000)
    
    # Use session in pipeline
    pipeline.run_interactive_generation(session, ...)
"""

import gradio as gr
import numpy as np
import torch
import threading
import time
import queue
from typing import Optional, Callable

from inferix.core.types.interactive import (
    GenerationStatus,
    ControlCommand,
    SessionState,
)


class InteractiveGradioBackend:
    """Gradio backend with interactive controls for video generation.
    
    This backend provides a web UI for interactive video generation:
    - Real-time video preview via streaming
    - Prompt input box for parameter changes
    - Guidance scale slider
    - Control buttons (Pause/Resume/Stop)
    - Status display (progress, ETA, queued inputs)
    
    Thread Safety:
        - All public methods are thread-safe
        - UI runs in separate thread from generation
    
    Attributes:
        session: InteractiveSession instance
        frame_queue: Queue for streaming frames
        status_queue: Queue for status updates
    """
    
    def __init__(self, session):
        """Initialize interactive Gradio backend.
        
        Args:
            session: InteractiveSession instance to sync with
        """
        self.session = session
        self.demo: Optional[gr.Blocks] = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=100)
        self.status_queue: queue.Queue = queue.Queue(maxsize=10)
        self.server_thread: Optional[threading.Thread] = None
        self.running = False
        self._current_status = {}  # Shared status state
        
    def connect(
        self,
        width: int,
        height: int,
        fps: int = 16,
        host: str = "0.0.0.0",
        port: int = 8000,
        frame_timeout: float = 10.0,
    ) -> bool:
        """Start interactive Gradio interface.
        
        Args:
            width: Video width
            height: Video height
            fps: Frames per second
            host: Server host
            port: Server port
            frame_timeout: Timeout for waiting frames (seconds)
            
        Returns:
            True if server started successfully
        """
        self.running = True
        self._width = width
        self._height = height
        self._fps = fps
        self._frame_timeout = frame_timeout
        
        # Set session status callback
        self.session.set_status_callback(self._on_status_update)
        
        # Create Gradio interface
        with gr.Blocks(title="Interactive Video Generation") as self.demo:
            gr.Markdown("# Interactive Video Generation")
            gr.Markdown(
                "Generate videos with real-time preview and interactive control. "
                "Submit new prompts to change generation direction."
            )
            
            # Main display area
            with gr.Row():
                # Video preview
                with gr.Column(scale=3):
                    video_display = gr.Image(
                        label="Live Preview",
                        type="numpy",
                        height=480,
                        streaming=True,
                    )
                
                # Status panel
                with gr.Column(scale=2):
                    status_display = gr.JSON(label="Status", value={})
            
            # User input area
            gr.Markdown("### Modify Generation")
            with gr.Row():
                prompt_input = gr.Textbox(
                    label="New Prompt",
                    placeholder="Enter new prompt to apply at next segment...",
                    scale=3,
                )
                guidance_slider = gr.Slider(
                    label="Guidance Scale",
                    minimum=1.0,
                    maximum=20.0,
                    value=7.5,
                    step=0.5,
                    scale=1,
                )
            
            # Control buttons
            with gr.Row():
                submit_btn = gr.Button("Submit Changes", variant="primary")
                pause_btn = gr.Button("Pause", variant="secondary")
                resume_btn = gr.Button("Resume", variant="secondary")
                stop_btn = gr.Button("Stop", variant="stop")
            
            # Result message
            result_msg = gr.Textbox(label="Result", interactive=False)
            
            # Frame generator for streaming (same pattern as GradioStreamingBackend)
            def frame_generator():
                """Generator that yields frames continuously with proper keep-alive."""
                print("[Gradio] Frame generator started, waiting for first frame...")
                
                # Block and wait for first frame
                try:
                    first_frame = self.frame_queue.get(block=True)
                    if first_frame is None:
                        print("[Gradio] Received stop signal before first frame.")
                        return
                    
                    # Format conversion
                    if first_frame.dtype != np.uint8:
                        first_frame = (np.clip(first_frame, 0, 1) * 255).astype(np.uint8)
                    
                    print(f"[Gradio] First frame: shape={first_frame.shape}")
                    yield first_frame
                    
                except Exception as e:
                    print(f"[Gradio] Error waiting for first frame: {e}")
                    return

                last_frame = first_frame
                frame_count = 1

                while self.running:
                    try:
                        frame = self.frame_queue.get(timeout=frame_timeout)
                        if frame is None:
                            print(f"[Gradio] Stream ended after {frame_count} frames.")
                            break
                        
                        # Format conversion
                        if frame.dtype != np.uint8:
                            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                        
                        last_frame = frame
                        frame_count += 1
                        
                        # Update status from queue
                        try:
                            while True:
                                status = self.status_queue.get_nowait()
                                self._current_status = status.to_dict()
                        except queue.Empty:
                            pass
                        
                        yield frame
                        
                    except queue.Empty:
                        # No new frame, yield last frame for keep-alive
                        yield last_frame
                        time.sleep(1.0 / fps)
            
            # Button callbacks
            def on_submit(prompt, guidance):
                """Handle submit button click."""
                if not prompt and guidance == 7.5:
                    return "No changes to submit"
                
                queued = self.session.submit_input(
                    prompt=prompt if prompt else None,
                    guidance_scale=guidance if guidance != 7.5 else None,
                    control=ControlCommand.CONTINUE,
                )
                
                return f"Changes queued! Will apply at {queued.will_apply_at} (ETA: {queued.estimated_wait_seconds:.1f}s)"
            
            def on_pause():
                """Handle pause button click."""
                self.session.submit_input(control=ControlCommand.PAUSE)
                return "Generation paused"
            
            def on_resume():
                """Handle resume button click."""
                self.session.submit_input(control=ControlCommand.RESUME)
                return "Generation resumed"
            
            def on_stop():
                """Handle stop button click."""
                self.session.submit_input(control=ControlCommand.STOP)
                return "Generation stopped"
            
            # Bind button events
            submit_btn.click(
                on_submit,
                inputs=[prompt_input, guidance_slider],
                outputs=[result_msg],
            )
            pause_btn.click(on_pause, outputs=[result_msg])
            resume_btn.click(on_resume, outputs=[result_msg])
            stop_btn.click(on_stop, outputs=[result_msg])
            
            # Auto-start streaming on page load (same pattern as GradioStreamingBackend)
            self.demo.load(fn=frame_generator, outputs=video_display)
        
        # Start server in background thread
        def run_server():
            try:
                print(f"[Gradio] Starting server on {host}:{port}...")
                self.demo.launch(
                    server_name=host,
                    server_port=port,
                    share=False,
                    quiet=False,
                )
            except Exception as e:
                print(f"[Gradio] Server error: {e}")
                import traceback
                traceback.print_exc()
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        # Wait for server to start
        time.sleep(2.0)
        
        if self.server_thread.is_alive():
            print(f"[Gradio] Server ready!")
            print(f"   Local access: http://localhost:{port}")
            
            # Try to detect WSL and provide Windows access URL
            try:
                import subprocess
                result = subprocess.run(
                    ['hostname', '-I'],
                    capture_output=True,
                    text=True,
                    timeout=1,
                )
                if result.returncode == 0:
                    wsl_ip = result.stdout.strip().split()[0]
                    print(f"   WSL access (from Windows): http://{wsl_ip}:{port}")
            except Exception:
                pass
            
            return True
        else:
            print("[Gradio] Server failed to start")
            return False
    
    def stream_batch(self, frames: torch.Tensor) -> bool:
        """Stream a batch of frames to the UI.
        
        This method is intended to be used as stream_callback
        in run_interactive_generation().
        
        Args:
            frames: Tensor of shape [T, H, W, C], dtype uint8, range [0, 255]
            
        Returns:
            True if streaming successful
        """
        if not self.running:
            return False
        
        frames_np = frames.cpu().numpy()
        T = frames_np.shape[0]
        
        for i in range(T):
            frame = frames_np[i]
            
            # Ensure uint8
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            
            try:
                self.frame_queue.put(frame, timeout=1.0)
            except queue.Full:
                print("[Gradio] Frame queue full, dropping frame")
                return False
        
        return True
    
    def broadcast_status(self, status: GenerationStatus):
        """Broadcast generation status to UI.
        
        This method can be set as session's status callback.
        
        Args:
            status: GenerationStatus to broadcast
        """
        if not self.running:
            return
        
        # Update local status
        self._current_status = status.to_dict()
        
        # Also push to queue for frame generator to pick up
        try:
            # Clear old status first
            while not self.status_queue.empty():
                try:
                    self.status_queue.get_nowait()
                except queue.Empty:
                    break
            self.status_queue.put(status, timeout=0.5)
        except queue.Full:
            pass  # Drop status if queue is full
    
    def _on_status_update(self, status: GenerationStatus):
        """Internal callback for session status updates."""
        self.broadcast_status(status)
    
    def disconnect(self):
        """Stop the Gradio server and cleanup."""
        self.running = False
        
        # Send stop signals
        try:
            self.frame_queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        
        print("[Gradio] Backend disconnected")
    
    def wait_for_user(self):
        """Block until user interacts (useful for demo mode)."""
        print("[Gradio] Waiting for user interaction...")
        print("   Press Ctrl+C to stop")
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Gradio] User interrupted")
            self.disconnect()
