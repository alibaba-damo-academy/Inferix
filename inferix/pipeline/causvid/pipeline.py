import torch
import os
import numpy as np
from omegaconf import OmegaConf
from typing import Optional, List, Tuple, Callable
import torch.distributed as dist
from diffusers.utils import export_to_video

from inferix.pipeline.causvid.CausalInferencePipeline import CausalInferencePipeline
from inferix.models.wan_base.utils.parallel_config import ParallelConfig
from inferix.kvcache_manager.kvcache_manager import KVCacheRequest, KVCacheManager
from inferix.core.memory.utils import gpu as get_gpu, DynamicSwapInstaller
from inferix.pipeline.base_pipeline import AbstractInferencePipeline


class CausVidPipeline(AbstractInferencePipeline):
    """CausVid video generation pipeline, responsible for core inference business logic"""
    
    def __init__(
        self, 
        config_path: str,
        default_config_path: str,
        wan_base_model_path: str,
        enable_kv_offload: bool,
        parallel_config: Optional[ParallelConfig] = None,
    ):
        """
        Initialize the CausVid pipeline
        
        Args:
            config_path: Configuration file path
            wan_base_model_folder: WAN base model folder path
            device_id: Device ID
            rank: Rank for distributed training
            ulysses_size: Ulysses parallel size
            ring_size: Ring parallel size
        """
        # self.config = OmegaConf.load(config_path)
        self.config = OmegaConf.load(config_path)
        if default_config_path:
            default_config = OmegaConf.load(default_config_path)
            self.config = OmegaConf.merge(default_config, self.config)
        
        # Call parent class initialization
        super().__init__(self.config)
        
        
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.parallel_config = parallel_config or ParallelConfig()
        self.wan_base_model_path = wan_base_model_path
        self.enable_kv_offload = enable_kv_offload
        
        # Read memory optimization config
        self._memory_mode = getattr(self.config, 'memory_mode', 'balanced')
        self._vae_chunk_size = getattr(self.config, 'vae_chunk_size', None)  # None = use memory_mode preset
        
        # Initialize pipeline
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the inference pipeline"""
        torch.set_grad_enabled(False)
            
        # Initialize CausVid pipeline
        self.pipeline = CausalInferencePipeline(
            self.config,
            wan_base_model_path = self.wan_base_model_path,
            device = self.device,
            enable_kv_offload = self.enable_kv_offload,
            parallel_config = self.parallel_config
        )
        self.pipeline = self.pipeline.to(dtype=torch.bfloat16)
    
    def load_checkpoint(self, checkpoint_path: str, **kwargs) -> None:
        """[Implementation of abstract method] Load model checkpoint weights"""
        ckpt_file = os.path.join(checkpoint_path, "model.pt")
        state_dict = torch.load(ckpt_file, map_location="cpu")
        
        # Try to find the correct key
        key = 'generator'
        if key not in state_dict:
            available_keys = list(state_dict.keys())
            print(f"Warning: Key '{key}' not found in checkpoint. Available keys: {available_keys}")
            
            # Try fallback key
            fallback_key = 'generator_ema'
            if fallback_key in state_dict:
                print(f"Using fallback key: '{fallback_key}'")
                key = fallback_key
            else:
                # Checkpoint might be the model state_dict directly
                print("Attempting to load state_dict directly...")
                self.pipeline.generator.load_state_dict(state_dict, strict=True)
                return
        
        self.pipeline.generator.load_state_dict(state_dict[key], strict=True)
    
    def setup_devices(self, low_memory: bool = False):
        """Setup devices and memory management"""
        gpu = get_gpu()
            
        if low_memory:
            DynamicSwapInstaller.install_model(self.pipeline.text_encoder, device=gpu)
        else:
            self.pipeline.text_encoder.to(device=gpu)
                
        self.pipeline.vae.to(device=gpu)
    
    def _print_model_specific_config(self, **kwargs):
        """
        Override base method to add CausVid specific configuration.
        
        Args:
            **kwargs: Model-specific parameters
        """
        # Print CausVid method info
        print(f"Method:          CausVid")
        print(f"Block Size:      {self.pipeline.num_frame_per_block} frames/block")
        
        # Print segment info
        num_segments = kwargs.get('num_rollout', 3)
        frames_per_segment = 21
        print(f"Segment Size:    {frames_per_segment} frames/segment (7 blocks)")
        print(f"Num Segments:    {num_segments}")

    def run_text_to_video(
            self,
            prompts: List,
            output_folder: Optional[str] = None,
            num_rollout: int = 3,
            num_overlap_frames: int = 3,
            is_diff_prompt: bool = False,
            is_interactive: bool = False,
            **kwargs
        ) -> Optional[str]:
        """[Implementation of abstract method] Run text-to-video generation"""
        
        # Print generation configuration summary
        rank = self.parallel_config.rank if self.parallel_config else 0
        if rank == 0:
            # Calculate total frames: num_rollout segments × 21 frames, with overlap
            frames_per_segment = 21
            total_frames = num_rollout * frames_per_segment - (num_rollout - 1) * num_overlap_frames
            self._print_generation_config(
                num_prompts=len(prompts),
                num_segments=num_rollout,
                segment_length=frames_per_segment,
                overlap_frames=num_overlap_frames,
                num_samples=1,
                memory_mode=self._memory_mode,
                chunk_size=self._vae_chunk_size,
                num_rollout=num_rollout,
            )
        
        # If output_folder is not provided, use the default value
        if output_folder is None:
            output_folder = "./output"
            
        # Validate overlap frames
        assert num_overlap_frames % self.pipeline.num_frame_per_block == 0, \
            "num_overlap_frames must be divisible by num_frame_per_block"
            
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Prepare kv cache manager
        kv_cache_manager = KVCacheManager(device=self.device)
        
        if is_diff_prompt:
            result = self._run_inference_diff_prompt(
                prompts=prompts,
                output_folder=output_folder,
                num_overlap_frames=num_overlap_frames,
                kv_cache_manager=kv_cache_manager,
                is_interactive=is_interactive
            )
        else:
            result = self._run_inference_same_prompt(
                prompts=prompts,
                output_folder=output_folder,
                num_rollout=num_rollout,
                num_overlap_frames=num_overlap_frames,
                kv_cache_manager=kv_cache_manager
            )
            
        return result
    
    def _run_inference_same_prompt(
            self,
            prompts: List,
            output_folder: str,
            num_rollout: int = 3,
            num_overlap_frames: int = 3,
            kv_cache_manager: Optional[KVCacheManager] = None,
            **kwargs
        ):
        """Inference when all segments share same prompt"""

        for prompt_idx, prompt in enumerate(prompts):
            kv_cache_request = KVCacheRequest(prompt)
            kv_cache_requests = [kv_cache_request]
            start_latents = None
            all_video = []
            # Execute multi-segment generation
            for segment_index in range(num_rollout):
                start_latents = self._generate_one_segment(
                    prompt, 
                    start_latents, 
                    all_video,
                    num_overlap_frames,
                    kv_cache_requests,
                    kv_cache_manager,
                )
            
            # Clear KV Cache for current prompt
            self.pipeline.clear_cache(kv_cache_manager, kv_cache_requests)

            # Save video
            video_name = f"prompt_{prompt_idx}"
            result = self._save_video(all_video, output_folder, video_name)
        
        return result

    def _run_inference_diff_prompt(
            self,
            prompts: List,
            output_folder: str,
            num_overlap_frames: int = 3,
            kv_cache_manager: Optional[KVCacheManager] = None,
            is_interactive: bool = False,
            **kwargs
        ) -> Optional[str]:
        """Inference when different segment has different prompt"""

        segment_id = 0
        start_latents = None
        all_video = []
        while True:
            if is_interactive:
                prompt = get_prompt_from_shell(segment_id)
            else:
                prompt = prompts[segment_id] if segment_id < len(prompts) else "Quit"
            
            if prompt == "Quit":
                break
            segment_id += 1
            kv_cache_request = KVCacheRequest(prompt)
            kv_cache_requests = [kv_cache_request]
            
            start_latents = self._generate_one_segment(
                prompt,
                start_latents,
                all_video,
                num_overlap_frames,
                kv_cache_requests,
                kv_cache_manager
            )
            
            # Clear KV Cache for current prompt
            self.pipeline.clear_cache(kv_cache_manager, kv_cache_requests)
        
        # Save video
        import uuid
        video_name = str(uuid.uuid4())
        result = self._save_video(all_video, output_folder, video_name)

        return result
    
    def _generate_one_segment(
            self,
            prompt: str,
            start_latents: torch.tensor,
            all_video: List,
            num_overlap_frames: int,
            kv_cache_requests: List,
            kv_cache_manager: Optional[KVCacheManager] = None
        ):
        """Generate video for one segment (21 frames = 7 blocks)"""

        sampled_noise = torch.randn(
            [1, 21, 16, 60, 104], 
            device=self.device, 
            dtype=torch.bfloat16
        )
        
        video, latents = self.pipeline.inference(
            noise=sampled_noise,
            text_prompts=[prompt],
            return_latents=True,
            start_latents=start_latents,
            kv_cache_manager=kv_cache_manager,
            kv_cache_requests=kv_cache_requests,
            vae_chunk_size=self._vae_chunk_size,
        )
        
        current_video = video[0].permute(0, 2, 3, 1).cpu().numpy()
        
        # Encode start frame
        start_frame = self._encode_start_frame(video, num_overlap_frames)
        
        start_latents = torch.cat(
            [start_frame, latents[:, -(num_overlap_frames - 1):]], 
            dim=1
        )
        
        all_video.append(current_video[:-(4 * (num_overlap_frames - 1) + 1)])

        return start_latents

        
    def run_image_to_video(self, prompt: str, image_path: str, **kwargs) -> Optional[str]:
        """[Implementation of abstract method] Run image-to-video generation (CausVid does not support)"""
        raise NotImplementedError("CausVid does not support image to video generation")
        
    def _encode_start_frame(self, video: torch.Tensor, num_overlap_frames: int) -> torch.Tensor:
        """Encode the start frame"""
        def encode(vae, videos: torch.Tensor) -> torch.Tensor:
            device, dtype = videos[0].device, videos[0].dtype
            scale = [vae.mean.to(device=device, dtype=dtype),
                     1.0 / vae.std.to(device=device, dtype=dtype)]
            output = [
                vae.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
                for u in videos
            ]
            output = torch.stack(output, dim=0)
            return output
            
        start_frame = encode(self.pipeline.vae, (
            video[:, -4 * (num_overlap_frames - 1) - 1:-4 * (num_overlap_frames - 1), :] * 2.0 - 1.0
        ).transpose(2, 1).to(torch.bfloat16)).transpose(2, 1).to(torch.bfloat16)
        
        return start_frame
    
    def _save_video(self, all_video: List, output_folder: str, video_name: str):
        output_path = None
        if self.parallel_config.rank == 0:
            video = np.concatenate(all_video, axis=0)
            output_path = os.path.join(output_folder, f"{video_name}.mp4")
            print(f"Save videos for prompt to {output_path}")
            export_to_video(video, output_path, fps=16)
    
    def _generate_segment_with_streaming(
        self,
        prompt: str,
        initial_latent: Optional[torch.Tensor],
        stream_callback: Optional[Callable[[torch.Tensor], None]],
        segment_length: int = 21,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [Implementation of abstract method] Generate segment with streaming for CausVid.
        
        Note: This is a placeholder implementation. Full streaming support for CausVid
        requires integration with its rollout-based generation architecture.
        """
        raise NotImplementedError(
            "Streaming generation is not yet implemented for CausVidPipeline. "
            "Please use the standard run_text_to_video method with num_rollout parameter."
        )


def get_prompt_from_shell(segment_id):
    """Input prompt from shell"""
    rank = int(os.environ["RANK"])
    gpu = get_gpu()
    # Only input in rank 0
    if rank == 0:
        prompt = input(f"> Please give prompt for segment {segment_id}. [You can input Quit to abort]:")
        prompt_tensor = torch.tensor([ord(c) for c in prompt], dtype=torch.uint8, device=gpu)
        prompt_len = torch.tensor(len(prompt_tensor), dtype=torch.long, device=gpu)
    else:
        prompt_len = torch.tensor(0, dtype=torch.long, device=gpu)
    
    # Broadcast prompt length from rank 0
    dist.broadcast(prompt_len, src=0)

    # Prepare tensor to receive prompt tensor in rank != 0
    if rank != 0:
        prompt_tensor = torch.zeros(prompt_len.item(), dtype=torch.uint8, device=gpu)
    
    # Broadcast prompt tensor from rank 0
    dist.broadcast(prompt_tensor, src=0)

    # Recover prompt from prompt tensor
    if rank != 0:
        prompt = ''.join(chr(x) for x in prompt_tensor.tolist())
    
    return prompt