"""Media processing and streaming utilities.

Streaming backends (priority order):
1. Gradio - Default, best for development and interaction
2. WebRTC - Optional, for real-time P2P communication (experimental)
3. RTMP - Optional, for production streaming

Interactive backend:
- InteractiveGradioBackend - Full interactive UI with controls
"""
from inferix.core.media.streaming_backend import (
    StreamingBackend,
    create_streaming_backend
)
from inferix.core.media.gradio_streaming import GradioStreamingBackend
from inferix.core.media.webrtc_streaming import WebRTCStreamingBackend
from inferix.core.media.rtmp_streaming import RTMPStreamingBackend
from inferix.core.media.interactive_gradio import InteractiveGradioBackend

__all__ = [
    'StreamingBackend',
    'create_streaming_backend',
    'GradioStreamingBackend',
    'WebRTCStreamingBackend',
    'RTMPStreamingBackend',
    'InteractiveGradioBackend',
]