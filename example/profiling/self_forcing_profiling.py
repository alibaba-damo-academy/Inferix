"""Example of how to use profiling with the Self Forcing pipeline.

This example shows how to use the built-in profiling capabilities of the SelfForcingPipeline
and how to extend it with custom profiling decorators for specific needs.
"""

import argparse
import os
import sys
from typing import List, Optional

# Note: These imports may show as errors in some IDEs but will work at runtime
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from inferix.pipeline.self_forcing.pipeline import SelfForcingPipeline
from inferix.models.wan_base.utils.parallel_config import ParallelConfig
from inferix.core.utils import set_random_seed
from inferix.core.memory.utils import gpu as get_gpu, get_cuda_free_memory_gb
from inferix.profiling.config import ProfilingConfig


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run Self Forcing Pipeline with profiling.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
    parser.add_argument("--default_config_path", type=str, 
                        default="example/self_forcing/configs/default_config.yaml",
                        help="Path to the default config file")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint folder")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt(s) for generation. Separate multiple prompts with semicolon ';'")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Path to input image for Image-to-Video (I2V). Required if --i2v is set.")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder")
    parser.add_argument("--num_output_frames", type=int, default=21,
                        help="Number of frames to generate")
    parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
    parser.add_argument("--save_with_index", action="store_true",
                        help="Whether to save the video using the index or prompt as the filename")
    parser.add_argument("--ulysses_size", type=int, default=1, help="Size of Ulysses Parallel")
    parser.add_argument("--ring_size", type=int, default=1, help="Size of Ring Sequence Parallel")
    parser.add_argument("--rtmp_url", type=str, default=None,
                        help="RTMP streaming URL, e.g., rtmp://localhost:1935/live/livestream. "
                             "If specified, stream while generating. Uses SRS by default; "
                             "you can replace it with your own RTMP-compatible player/server.")
    parser.add_argument("--rtmp_fps", type=int, default=16,
                        help="RTMP streaming frame rate, default is 16")
    
    # Profiling arguments
    parser.add_argument("--enable_profiling", action="store_true", help="Enable profiling")
    parser.add_argument("--profiling_config", type=str, default="example/profiling/configs/profiling_config.yaml",
                        help="Path to profiling configuration file")
    parser.add_argument("--profiling_output_dir", type=str, help="Directory for profiling reports")
    
    return parser.parse_args()


def setup_distributed_environment(args):
    """Setup distributed environment"""
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        rank = int(os.environ["RANK"])
        set_random_seed(args.seed)
        ulysses_size = args.ulysses_size
        ring_size = args.ring_size
        
        parallel_config = ParallelConfig(
            local_rank=local_rank,
            rank=rank,
            world_size=world_size,
            ulysses_size=ulysses_size,
            ring_size=ring_size,
        )
    else:
        device = torch.device("cuda:0")
        local_rank = 0
        world_size = 1
        rank = 0
        set_random_seed(args.seed)
        parallel_config = ParallelConfig()
    
    torch.cuda.set_device(local_rank)
    
    return parallel_config, rank


def load_profiling_config(config_path: str, args, rank: int) -> Optional[ProfilingConfig]:
    """Load profiling configuration from file and command line arguments."""
    if not args.enable_profiling:
        return None
    
    # Load config from file
    config_dict = OmegaConf.load(config_path)
    profiling_config_dict = config_dict.get('profiling', {})
    
    # Override with command line arguments
    if args.profiling_output_dir:
        # Add rank to output directory to avoid conflicts in multi-process scenarios
        output_dir = os.path.join(args.profiling_output_dir, f"rank_{rank}")
        profiling_config_dict['output_dir'] = output_dir
    
    # Create ProfilingConfig object
    try:
        profiling_config = ProfilingConfig(**profiling_config_dict)
        profiling_config.enabled = True  # Explicitly enable since we checked the flag
        return profiling_config
    except Exception as e:
        print(f"Warning: Failed to create profiling config: {e}")
        return None


def main():
    """Main function"""
    args = parse_arguments()
    
    # Setup distributed environment
    parallel_config, rank = setup_distributed_environment(args)
    
    # Load profiling configuration
    profiling_config = load_profiling_config(args.profiling_config, args, rank)
    
    # Display memory information
    gpu = get_gpu()
    print(f'[Rank {rank}] Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
    low_memory = get_cuda_free_memory_gb(gpu) < 40
    
    # Create pipeline with built-in profiling support
    pipeline = SelfForcingPipeline(
        config_path=args.config_path,
        default_config_path=args.default_config_path,
        parallel_config=parallel_config,
        profiling_config=profiling_config
    )
    
    try:
        # Load checkpoint
        pipeline.load_checkpoint(args.checkpoint_path, use_ema=args.use_ema)
        
        # Set device
        pipeline.setup_devices(low_memory=low_memory)
        
        # Parse prompts
        prompts = [p.strip() for p in args.prompt.split(';') if p.strip()]
        if not prompts:
            raise ValueError("No valid prompts provided.")
        
        # Execute inference based on mode
        if args.i2v:
            if not args.image_path:
                print("Error: --image_path is required for i2v mode.")
                sys.exit(1)
            if dist.is_initialized():
                print("Error: I2V does not support distributed inference yet.")
                sys.exit(1)
                
            pipeline.run_image_to_video(
                prompts=prompts,
                image_path=args.image_path,
                num_output_frames=args.num_output_frames,
                num_samples=args.num_samples,
                output_folder=args.output_folder,
                save_with_index=args.save_with_index,
                use_ema=args.use_ema,
                low_memory=low_memory
            )
        else:
            pipeline.run_text_to_video(
                prompts=prompts,
                num_output_frames=args.num_output_frames,
                num_samples=args.num_samples,
                output_folder=args.output_folder,
                save_with_index=args.save_with_index,
                use_ema=args.use_ema,
                rtmp_url=args.rtmp_url,
                rtmp_fps=args.rtmp_fps,
                low_memory=low_memory
            )
        
        # Cleanup distributed environment
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        
        if rank == 0:
            print("✅ All videos generated successfully.")
            
            # Print additional profiling information if available
            if pipeline._profiling_enabled and pipeline._profiler is not None and pipeline._profiler.diffusion_analyzer is not None:
                diffusion_analysis = pipeline._profiler.diffusion_analyzer.get_diffusion_analysis()
                block_analysis = pipeline._profiler.diffusion_analyzer.get_block_analysis()
                
                if diffusion_analysis:
                    print(f"📊 Diffusion Analysis:")
                    print(f"   - Total Steps: {diffusion_analysis['total_steps']}")
                    print(f"   - FPS (Frames Per Second): {diffusion_analysis['fps']:.2f}")
                    print(f"   - BPS (Blocks Per Second): {diffusion_analysis['bps']:.2f}")
                    print(f"   - Average Block Size: {diffusion_analysis['avg_block_size']:.1f} frames")
                
                if block_analysis:
                    print(f"📦 Block Analysis:")
                    print(f"   - Total Blocks: {block_analysis['total_blocks']}")
                    print(f"   - Block FPS: {block_analysis['fps']:.2f}")
                    print(f"   - Block BPS: {block_analysis['bps']:.2f}")
                    print(f"   - Peak Memory Usage: {block_analysis['peak_memory_usage_mb']:.0f} MB")
            
    finally:
        # Clean up profiling resources
        if hasattr(pipeline, 'cleanup_profiling'):
            pipeline.cleanup_profiling()


if __name__ == "__main__":
    main()