"""
Interactive Streaming Generation Showcase for Self-Forcing

This script demonstrates the interactive generation capability:
1. Real-time video preview with Gradio UI
2. User input queue for prompt/guidance changes
3. Pause/Resume/Stop controls
4. Status display with progress and ETA
5. Multi-segment long video generation with overlap

Usage:
    # Basic interactive generation
    python example/streaming/run_interactive_streaming.py \
        --config_path example/self_forcing/configs/self_forcing_dmd.yaml \
        --checkpoint_path ./weights/self_forcing/checkpoints/self_forcing_dmd.pt \
        --prompt "A cat walking in a garden" \
        --num_segments 5

Features:
    - Open http://localhost:8000 to see interactive UI
    - Submit new prompts to change generation direction
    - Adjust guidance scale in real-time
    - Pause/Resume/Stop generation at any time
    - See progress and ETA updates
"""

import argparse
import os
import sys
import torch

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from inferix.pipeline.self_forcing.pipeline import SelfForcingPipeline
from inferix.core.interactive import InteractiveSession
from inferix.core.media import InteractiveGradioBackend, create_streaming_backend
from inferix.core.types import InputApplyPolicy
from inferix.profiling import ProfilingConfig


def parse_args():
    parser = argparse.ArgumentParser(description='Interactive Streaming Generation Showcase')
    
    # Model configuration
    parser.add_argument('--config_path', type=str, required=True,
                       help='Path to model configuration YAML')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--default_config_path', type=str, default=None,
                       help='Path to default configuration YAML')
    
    # Generation parameters
    parser.add_argument('--prompt', type=str, required=True,
                       help='Initial text prompt for video generation')
    parser.add_argument('--num_segments', type=int, default=5,
                       help='Number of segments to generate (1=short, 10+=long)')
    parser.add_argument('--segment_length', type=int, default=21,
                       help='Frames per segment (must be multiple of 3)')
    parser.add_argument('--overlap_frames', type=int, default=3,
                       help='Overlapping frames between segments')
    parser.add_argument('--num_samples', type=int, default=1,
                       help='Batch size (number of videos to generate)')
    
    # Interactive options
    parser.add_argument('--apply_policy', type=str, default='next_segment',
                       choices=['next_segment', 'next_block'],
                       help='When to apply user input (next_segment recommended)')
    parser.add_argument('--port', type=int, default=8000,
                       help='Server port for Gradio UI')
    
    # Streaming options
    parser.add_argument('--streaming_backend', type=str, default='gradio',
                       choices=['gradio', 'interactive'],
                       help='Streaming backend type')
    
    # Optimization options
    parser.add_argument('--low_memory', action='store_true',
                       help='Enable memory optimization mode')
    parser.add_argument('--use_memory_manager', action='store_true',
                       help='Use AsyncMemoryManager for component-level offload')
    parser.add_argument('--use_ema', action='store_true',
                       help='Use EMA weights')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    
    # Output options
    parser.add_argument('--output_folder', type=str, default=None,
                       help='Folder to save generated videos (optional)')
    
    # Profiling options
    parser.add_argument('--enable_profiling', action='store_true',
                       help='Enable performance profiling')
    parser.add_argument('--profile_output_dir', type=str, default='./profiling_results',
                       help='Directory to save profiling reports')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Print header
    print("=" * 70)
    print("Interactive Streaming Generation Showcase")
    print("=" * 70)
    print(f"Initial Prompt: {args.prompt}")
    print(f"Apply Policy:   {args.apply_policy.upper()}")
    print(f"Segments:       {args.num_segments}")
    print(f"Segment length: {args.segment_length} frames")
    print(f"Overlap:        {args.overlap_frames} frames")
    print(f"Backend:        {args.streaming_backend.upper()}")
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
        )
    
    # Initialize pipeline
    pipeline = SelfForcingPipeline(
        config_path=args.config_path,
        default_config_path=args.default_config_path,
        profiling_config=profiling_config,
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint_path}")
    pipeline.load_checkpoint(args.checkpoint_path, use_ema=args.use_ema)
    
    # Setup devices
    print("Setting up devices...")
    pipeline.setup_devices(
        low_memory=args.low_memory,
        use_memory_manager=args.use_memory_manager,
    )
    
    # Create interactive session
    apply_policy = (
        InputApplyPolicy.NEXT_SEGMENT 
        if args.apply_policy == 'next_segment' 
        else InputApplyPolicy.NEXT_BLOCK
    )
    
    session = InteractiveSession(apply_policy=apply_policy)
    session.set_device(torch.device(f"cuda:{torch.cuda.current_device()}"))
    
    # Create streaming backend
    if args.streaming_backend == 'interactive':
        # Full interactive backend with UI controls
        backend = InteractiveGradioBackend(session)
        stream_callback = backend.stream_batch
    else:
        # Standard Gradio streaming (no interactive controls)
        backend = create_streaming_backend("gradio")
        stream_callback = backend.stream_batch
    
    # Connect backend
    print(f"\nInitializing {args.streaming_backend} streaming...")
    if backend.connect(width=832, height=480, fps=16, port=args.port):
        print(f"Backend ready: http://localhost:{args.port}")
    else:
        print("Failed to initialize streaming backend!")
        return 1
    
    # Run interactive generation
    print("\n" + "=" * 70)
    print("Starting interactive generation...")
    print("=" * 70)
    print("\nINSTRUCTIONS:")
    print("  1. Open http://localhost:8000 in your browser")
    print("  2. Watch real-time video preview")
    print("  3. Submit new prompts to change generation direction")
    print("  4. Use Pause/Resume/Stop buttons to control generation")
    print("  5. See progress and ETA in status panel")
    print("\n" + "=" * 70)
    print()
    
    try:
        video = pipeline.run_interactive_generation(
            session=session,
            initial_prompt=args.prompt,
            num_segments=args.num_segments,
            segment_length=args.segment_length,
            overlap_frames=args.overlap_frames,
            stream_callback=stream_callback,
            num_samples=args.num_samples,
            low_memory=args.low_memory,
        )
        
        print("\n" + "=" * 70)
        print("Generation completed!")
        print("=" * 70)
        
        if video is not None:
            print(f"Generated video shape: {video.shape}")
        
        # Save if requested
        if args.output_folder and video is not None:
            os.makedirs(args.output_folder, exist_ok=True)
            from torchvision.io import write_video
            
            output_path = os.path.join(args.output_folder, "interactive_streaming.mp4")
            video_255 = torch.clamp(video[0] * 255.0, 0, 255).to(torch.uint8)
            write_video(output_path, video_255, fps=16)
            print(f"Video saved to: {output_path}")
    
    except Exception as e:
        print(f"\nGeneration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Keep Gradio running for loop playback
        if args.streaming_backend == 'gradio':
            print(f"\nGradio server still running (loop playback active)")
            print(f"Access: http://localhost:{args.port}")
            print("Press Ctrl+C to stop...")
            try:
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping...")
        
        backend.disconnect()
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
