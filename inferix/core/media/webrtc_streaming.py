from fastrtc import Stream
import numpy as np
import torch
import time
from typing import Optional
import queue
import threading

class PersistentWebRTCStreamer:
    def __init__(self):
        self.stream: Optional[Stream] = None
        self.frame_queue = queue.Queue()
        self.running = False
        self.server_thread = None

    def connect(self, width: int, height: int, fps: int = 16, host="0.0.0.0", port=8000):
        def frame_generator():
            while self.running:
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                    if frame is None:
                        break
                    yield frame
                except queue.Empty:
                    continue

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
        return True

    def stream_batch(self, x: torch.Tensor):
        if not self.running:
            return False

        x = x.cpu().numpy()
        T = x.shape[0]

        for i in range(T):
            frame = x[i]
            try:
                self.frame_queue.put(frame, timeout=1.0)
            except queue.Full:
                return False
        return True

    def disconnect(self):
        self.running = False
        try:
            self.frame_queue.put(None, timeout=1.0)
        except queue.Full:
            pass