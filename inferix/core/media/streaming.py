# rtmp_streamer.py
import torch
import numpy as np
import av
import fractions
import time
from typing import Optional, Any

# Try to import profiling components, but make them optional
try:
    from inferix.profiling.decorators import profile_streaming
    PROFILING_AVAILABLE = True
except ImportError:
    profile_streaming = lambda x=None: lambda y: y
    PROFILING_AVAILABLE = False


class PersistentRTMPStreamer:
    def __init__(self):
        self.container = None
        self.stream = None
        self.is_connected = False
        self.fps = 16
        self.frame_count = 0
        self.start_time = None

    def connect(self, rtmp_url: str, width: int, height: int, fps: int = 16):
        """Establish a persistent RTMP connection."""
        try:
            self.fps = fps

            # Set basic container parameters; encoding parameters are configured in codec_context
            self.container = av.open(rtmp_url, mode='w', format='flv')
            self.stream = self.container.add_stream('libx264', rate=fps)
            self.stream.width = width
            self.stream.height = height
            self.stream.pix_fmt = 'yuv420p'

            codec_context = self.stream.codec_context
            codec_context.width = width
            codec_context.height = height
            codec_context.pix_fmt = 'yuv420p'
            codec_context.time_base = fractions.Fraction(1, fps)  # Critical: set time base
            codec_context.framerate = fps

            # Set encoding options
            codec_context.options = {
                'preset': 'ultrafast',
                'tune': 'zerolatency',
                'profile': 'baseline',
                'level': '3.1',  # Lower level for better compatibility
                'g': '16',  # GOP size
                'keyint_min': '16',  # Minimum keyframe interval
                'sc_threshold': '0',
                'b_strategy': '0',  # Disable B-frames
                'crf': '23'
            }

            # Open the encoder
            codec_context.open()

            self.is_connected = True
            self.frame_count = 0
            self.start_time = time.time()
            print(f"✅ RTMP connection established: {rtmp_url}")
            print(f"   Settings: {width}x{height}@{fps}fps")
            return True
        except Exception as e:
            print(f"❌ RTMP connection failed: {e}")
            import traceback
            traceback.print_exc()
            self.is_connected = False
            return False

    @profile_streaming(lambda args, kwargs: args[1].shape[0] if len(args) > 1 and hasattr(args[1], 'shape') else 1)
    def stream_batch(self, x: torch.Tensor):
        """Stream a batch of frames — expects uint8 tensor of shape [T, H, W, C]."""
        if not self.is_connected:
            return False

        try:
            x = x.cpu().numpy()  # [T, H, W, C] uint8, range [0, 255]
            T, H, W, C = x.shape
            success_count = 0

            print(f"📊 Starting to stream batch: {T} frames, shape: {x.shape}")

            for i in range(T):
                frame = x[i]  # [H, W, C] — RGB

                # Create AVFrame
                av_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')

                # Set timestamp — use the same time base as the encoder
                av_frame.pts = self.frame_count  # Frame sequence number
                av_frame.time_base = fractions.Fraction(1, self.fps)

                try:
                    # Encode and mux packets
                    packets = self.stream.encode(av_frame)
                    for packet in packets:
                        if packet is not None:
                            self.container.mux(packet)
                    success_count += 1
                    self.frame_count += 1
                except Exception as e:
                    print(f"❌ Failed to encode frame {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # Force flush the encoder
            try:
                packets = self.stream.encode(None)
                for packet in packets:
                    if packet is not None:
                        self.container.mux(packet)
            except Exception as e:
                print(f"⚠️  Error during encoder flush: {e}")

            print(f"📤 Streaming complete: {success_count}/{T} frames, total frames: {self.frame_count}")
            return True

        except Exception as e:
            print(f"❌ Streaming error: {e}")
            import traceback
            traceback.print_exc()
            self.is_connected = False
            return False

    def flush(self):
        """Force flush the encoder buffer."""
        if self.is_connected and self.container:
            try:
                packets = self.stream.encode(None)
                for packet in packets:
                    if packet is not None:
                        self.container.mux(packet)
            except Exception as e:
                print(f"❌ Error during flush: {e}")

    def disconnect(self):
        try:
            if self.is_connected and self.stream and self.container:
                print("🔌 Disconnecting RTMP stream...")

                # Flush encoder first
                try:
                    packets = self.stream.encode(None)
                    for packet in packets:
                        if packet is not None:
                            self.container.mux(packet)
                except Exception as e:
                    print(f"⚠️  Error flushing encoder: {e}")

                time.sleep(0.1)

                # Close container
                try:
                    self.container.close()
                    print("✅ RTMP connection closed")
                except Exception as e:
                    print(f"⚠️  Error closing container: {e}")

        except Exception as e:
            print(f"❌ Error during disconnection: {e}")
            import traceback
            traceback.print_exc()

        self.is_connected = False
        self.container = None
        self.stream = None