"""
Progressive Streaming Generation Example for Self-Forcing

This script demonstrates the progressive streaming feature where:
1. Video is generated block-by-block (3 frames per block)
2. Each block is decoded and streamed immediately
3. Multiple segments can be chained for long videos
4. Memory is automatically cleaned up between segments

Usage:
    # Gradio streaming (default, recommended)
    python example/streaming/run_progressive_streaming.py \
        --config_path example/self_forcing/configs/self_forcing_dmd.yaml \
        --checkpoint_path ./weights/self_forcing/checkpoints/self_forcing_dmd.pt \
        --prompt "A cat walking" \
        --num_segments 1

    # RTMP streaming (for production)
    python example/streaming/run_progressive_streaming.py \
        --config_path example/self_forcing/configs/self_forcing_dmd.yaml \
        --checkpoint_path ./weights/self_forcing/checkpoints/self_forcing_dmd.pt \
        --prompt "A cat walking" \
        --streaming_backend rtmp \
        --rtmp_url rtmp://localhost:1935/live/stream
"""

import argparse
import os
import sys
import torch

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from inferix.pipeline.self_forcing.pipeline import SelfForcingPipeline
from inferix.core.media import create_streaming_backend
from inferix.profiling import ProfilingConfig


def main():
    parser = argparse.ArgumentParser(description='Progressive Streaming Generation')
    
    # Model configuration
    parser.add_argument('--config_path', type=str, required=True,
                       help='Path to model configuration YAML')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--default_config_path', type=str, default=None,
                       help='Path to default configuration YAML')
    
    # Generation parameters
    parser.add_argument('--prompt', type=str, required=True,
                       help='Text prompt for video generation')
    parser.add_argument('--num_segments', type=int, default=1,
                       help='Number of segments to generate (1=short, 10+=long)')
    parser.add_argument('--segment_length', type=int, default=21,
                       help='Frames per segment (must be multiple of 3)')
    parser.add_argument('--overlap_frames', type=int, default=3,
                       help='Overlapping frames between segments')
    parser.add_argument('--num_samples', type=int, default=1,
                       help='Batch size (number of videos to generate)')
    
    # Streaming options
    parser.add_argument('--streaming_backend', type=str, default='gradio',
                       choices=['gradio', 'webrtc', 'rtmp'],
                       help='Streaming backend (gradio=default/recommended, webrtc=experimental, rtmp=production)')
    parser.add_argument('--port', type=int, default=8000,
                       help='Server port (for gradio/webrtc)')
    parser.add_argument('--rtmp_url', type=str, default=None,
                       help='RTMP URL (required for rtmp backend)')
    
    # Optimization options
    parser.add_argument('--low_memory', action='store_true',
                       help='Enable memory optimization mode')
    parser.add_argument('--use_ema', action='store_true',
                       help='Use EMA weights')
    
    # Output options
    parser.add_argument('--output_folder', type=str, default=None,
                       help='Folder to save generated videos (optional)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    
    # Profiling options
    parser.add_argument('--enable_profiling', action='store_true',
                       help='Enable performance profiling')
    parser.add_argument('--profile_output_dir', type=str, default='./profiling_results',
                       help='Directory to save profiling reports')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("=" * 70)
    print("Progressive Streaming Generation")
    print("=" * 70)
    print(f"Prompt: {args.prompt}")
    print(f"Backend: {args.streaming_backend.upper()}")
    print(f"Segments: {args.num_segments}")
    print(f"Segment length: {args.segment_length} frames")
    print(f"Overlap: {args.overlap_frames} frames")
    print(f"Total frames: {args.num_segments * args.segment_length - (args.num_segments - 1) * args.overlap_frames}")
    print("=" * 70)
    print()
    
    # Initialize profiling configuration
    profiling_config = None
    if args.enable_profiling:
        print("Enabling performance profiling...")
        profiling_config = ProfilingConfig(
            enabled=True,
            real_time_display=True,
            generate_final_report=True,
            report_format="both",
            output_dir=args.profile_output_dir,
            profile_diffusion_steps=True,
            profile_block_computation=True
        )
    
    # Initialize pipeline
    pipeline = SelfForcingPipeline(
        config_path=args.config_path,
        default_config_path=args.default_config_path,
        profiling_config=profiling_config
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint_path}")
    pipeline.load_checkpoint(args.checkpoint_path, use_ema=args.use_ema)
    
    # Setup devices
    print("Setting up devices...")
    pipeline.setup_devices(low_memory=args.low_memory)
    
    # Initialize streaming backend
    streamer = None
    stream_callback = None
    
    try:
        streamer = create_streaming_backend(args.streaming_backend)
        
        # Configure backend-specific parameters
        if args.streaming_backend == 'rtmp' and not args.rtmp_url:
            raise ValueError("--rtmp_url required for RTMP backend")
        
        connect_params = {
            'width': 832,
            'height': 480,
            'fps': 16
        }
        
        if args.streaming_backend in ['gradio', 'webrtc']:
            connect_params['port'] = args.port
        elif args.streaming_backend == 'rtmp':
            connect_params['rtmp_url'] = args.rtmp_url
        
        print(f"\nInitializing {args.streaming_backend.upper()} streaming...")
        if streamer.connect(**connect_params):
            print(f"✅ {args.streaming_backend.upper()} ready")
            stream_callback = streamer.stream_batch
        else:
            print(f"❌ Failed to initialize {args.streaming_backend}")
            print(f"\n💥 Streaming backend initialization failed!")
            print("   This example requires working streaming backend.")
            return 1
    except Exception as e:
        print(f"❌ Streaming initialization error: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n💥 Streaming backend initialization failed!")
        print("   This example requires working streaming backend.")
        return 1
    
    # Generate with progressive streaming
    print("\n" + "=" * 70)
    print("Starting progressive generation...")
    print("=" * 70)
    print()
    
    try:
        video = pipeline.run_streaming_generation(
            prompts=[args.prompt],
            stream_callback=stream_callback,
            num_segments=args.num_segments,
            segment_length=args.segment_length,
            overlap_frames=args.overlap_frames,
            num_samples=args.num_samples,
            low_memory=args.low_memory
        )
        
        print("\n" + "=" * 70)
        print("✅ Generation completed!")
        print("=" * 70)
        print(f"Generated video shape: {video.shape if video is not None else 'N/A'}")
                
        # Display profiling summary if enabled
        if args.enable_profiling and hasattr(pipeline, '_profiler') and pipeline._profiler is not None:
            if pipeline._profiler.diffusion_analyzer is not None:
                diffusion_analysis = pipeline._profiler.diffusion_analyzer.get_diffusion_analysis()
                block_analysis = pipeline._profiler.diffusion_analyzer.get_block_analysis()
                        
                if diffusion_analysis:
                    print("\n📊 Streaming Performance Summary:")
                    print(f"   - Total Diffusion Steps: {diffusion_analysis['total_steps']}")
                    print(f"   - Generation FPS: {diffusion_analysis['fps']:.2f}")
                    print(f"   - Blocks Per Second: {diffusion_analysis['bps']:.2f}")
                    print(f"   - Average Block Size: {diffusion_analysis['avg_block_size']:.1f} frames")
                        
                if block_analysis:
                    print(f"\n📦 Block-Level Performance:")
                    print(f"   - Total Blocks: {block_analysis['total_blocks']}")
                    print(f"   - Block FPS: {block_analysis['fps']:.2f}")
                    print(f"   - Peak Memory: {block_analysis['peak_memory_usage_mb']:.0f} MB")
                        
                print(f"\n💾 Profiling reports saved to: {args.profile_output_dir}")
                print("   Use extract_streaming_metrics.py to extract data for documentation")
        
        # Optionally save to disk
        if args.output_folder and video is not None:
            os.makedirs(args.output_folder, exist_ok=True)
            from torchvision.io import write_video
            
            output_path = os.path.join(args.output_folder, "progressive_streaming.mp4")
            video_255 = torch.clamp(video[0] * 255.0, 0, 255).to(torch.uint8)
            write_video(output_path, video_255, fps=16)
            print(f"Video saved to: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup streaming
        if streamer:
            if args.streaming_backend == 'gradio':
                print(f"\n🔁 Gradio server still running (loop playback active)")
                print(f"   Access: http://localhost:{args.port}")
                print("   Press Ctrl+C to stop...")
                try:
                    import time
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print(f"\nStopping {args.streaming_backend}...")
                    streamer.disconnect()
            else:
                print(f"\nDisconnecting {args.streaming_backend}...")
                streamer.disconnect()
    
    print("\nDone!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
