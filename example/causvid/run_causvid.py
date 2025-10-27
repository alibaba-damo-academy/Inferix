import argparse
import torch
import os
import torch.distributed as dist

from inferix.pipeline.causvid.pipeline import CausVidPipeline
from inferix.models.wan_base.utils.parallel_config import ParallelConfig
from inferix.core.utils import set_random_seed
from inferix.core.memory.utils import gpu as get_gpu, get_cuda_free_memory_gb


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run CausVid Pipeline for text-to-video generation.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
    parser.add_argument("--default_config_path", type=str, 
                        default="example/causvid/configs/default_config.yaml",
                        help="Path to the default config file")
    parser.add_argument("--checkpoint_folder", type=str, required=True, help="Path to the checkpoint folder")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt(s) for generation. Separate multiple prompts with semicolon ';'")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder for generated videos")
    parser.add_argument("--wan_base_model_path", type=str, default=None, help="Path to WAN base model folder")
    parser.add_argument("--num_rollout", type=int, default=3, help="Number of rollout iterations")
    parser.add_argument("--num_overlap_frames", type=int, default=3, help="Number of overlap frames")
    parser.add_argument("--ulysses_size", type=int, default=1, help="Size of Ulysses Parallel")
    parser.add_argument("--ring_size", type=int, default=1, help="Size of Ring Parallel")
    parser.add_argument("--enable_kv_offload", type=bool, default=True, help="Enable KV Cache offload to CPU")
    parser.add_argument("--diff_prompt", type=bool, default=False, help="Different chunk have different prompt")
    parser.add_argument("--is_interactive", type=bool, default=False, help="Get prompt from shell when diff_prompt is true")
    parser.add_argument("--diff_prompt_file", type=str, default=None, help="Load prompts from file when diff_prompt is true")

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
        device = torch.device("cuda")
        local_rank = 0
        world_size = 1
        rank = 0
        set_random_seed(args.seed)
        parallel_config = ParallelConfig()
    
    torch.cuda.set_device(device)
    
    return parallel_config, rank

def parse_prompt(args):
    rank = int(os.environ["RANK"])

    if args.diff_prompt:
        if args.is_interactive:
            prompts = []
            if rank == 0:
                if args.diff_prompt_file is not None or args.prompt is not None:
                    print("is_interactive has been set to True, the prompts you specified by --prompt and --diff_prompt_file will be ignored.")
        else:
            if args.diff_prompt_file is not None:
                    from omegaconf import OmegaConf
                    prompts = OmegaConf.load(args.diff_prompt_file).prompts
                    print(prompts)
                    if rank == 0:
                        if args.prompt is not None:
                            print("Prompts have been specified by --diff_prompt_file, --prompt will be ignored.")
            else:
                prompts = [p.strip() for p in args.prompt.split(';') if p.strip()]
    else:
        prompts = [p.strip() for p in args.prompt.split(';') if p.strip()]
    
    return prompts


def main():
    """Main function"""
    args = parse_arguments()
    
    # Setup distributed environment
    parallel_config, rank = setup_distributed_environment(args)

    gpu = get_gpu()
    print(f'[Rank {rank}] Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
    low_memory = get_cuda_free_memory_gb(gpu) < 40
    
    # Initialize pipeline
    pipeline = CausVidPipeline(
        config_path=args.config_path,
        default_config_path=args.default_config_path,
        wan_base_model_path=args.wan_base_model_path,
        enable_kv_offload=args.enable_kv_offload,
        parallel_config = parallel_config
    )
    
    # Load checkpoint
    pipeline.load_checkpoint(args.checkpoint_folder)
    
    # Set device
    pipeline.setup_devices(low_memory=low_memory)

    # Parse prompts
    prompts = parse_prompt(args)

    # Run text-to-video generation
    pipeline.run_text_to_video(
        prompts=prompts,
        output_folder=args.output_folder,
        num_rollout=args.num_rollout,
        num_overlap_frames=args.num_overlap_frames,
        is_diff_prompt=args.diff_prompt,
        is_interactive=args.is_interactive
    )

    # Cleanup distributed environment
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    
    if rank == 0:
        print("✅ All videos generated successfully.")


if __name__ == "__main__":
    main()