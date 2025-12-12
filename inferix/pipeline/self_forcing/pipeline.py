import torch
import os
from omegaconf import OmegaConf
from typing import List, Optional, Dict, Any, Callable, Tuple
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from PIL import Image

from inferix.core.memory.utils import gpu as get_gpu, DynamicSwapInstaller
from inferix.pipeline.self_forcing.CausalInferencePipeline import CausalInferencePipeline
from inferix.pipeline.self_forcing.CausalDiffusionInferencePipeline import CausalDiffusionInferencePipeline
from inferix.kvcache_manager.kvcache_manager import KVCacheRequest, KVCacheManager
from inferix.models.wan_base.utils.parallel_config import ParallelConfig
from inferix.core.media.streaming import PersistentRTMPStreamer
from inferix.core.media.webrtc_streaming import PersistentWebRTCStreamer

from inferix.pipeline.base_pipeline import AbstractInferencePipeline
from inferix.profiling.config import ProfilingConfig
# Import the profiling decorators directly from the decorators module
from inferix.profiling.decorators import profile_method, profile_session, add_profiling_event


class SelfForcingPipeline(AbstractInferencePipeline):
    """Self Forcing video generation pipeline, responsible for core inference business logic"""
    
    def __init__(self, config_path: str, default_config_path: Optional[str] = None, 
                 parallel_config: Optional[ParallelConfig] = None, profiling_config: Optional[ProfilingConfig] = None):
        """
        Initialize the Self Forcing pipeline
        
        Args:
            config_path: Configuration file path
            default_config_path: Default configuration file path
            parallel_config: Parallel configuration
            profiling_config: Profiling configuration (optional)
        """
        # Load configuration
        config = OmegaConf.load(config_path)
        if default_config_path:
            default_config = OmegaConf.load(default_config_path)
            config = OmegaConf.merge(default_config, config)
        
        # Call parent class initialization with profiling config
        super().__init__(config, profiling_config)
        
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.parallel_config = parallel_config or ParallelConfig()
        
        # Initialize pipeline
        self._initialize_pipeline()
        
    @profile_method("pipeline_initialization")
    def _initialize_pipeline(self, *args, **kwargs):
        """Initialize the inference pipeline on meta device to avoid CPU OOM"""
        torch.set_grad_enabled(False)
        print(f"Initializing {self.__class__.__name__} (meta device, 0 MB)...")
        
        # Create pipeline on meta device (zero memory footprint)
        meta_device = torch.device("meta")
        pipeline_class = CausalInferencePipeline if hasattr(self.config, 'denoising_step_list') else CausalDiffusionInferencePipeline
        
        self.pipeline = pipeline_class(
            self.config,
            device=meta_device,
            parallel_config=self.parallel_config,
            profiler=self._profiler if pipeline_class == CausalInferencePipeline else None
        )
        self._pipeline_type = "causal" if pipeline_class == CausalInferencePipeline else "diffusion"
        
    def _init_model(self) -> Any:
        """[Implementation of abstract method] Initialize and return the specific model instance"""
        # Model architecture has been initialized in _initialize_pipeline
        return self.pipeline
        
    @profile_method("load_checkpoint")
    @add_profiling_event("checkpoint_loaded", lambda result, *args, **kwargs: {
        "checkpoint_path": args[1] if len(args) > 1 else None,
        "use_ema": kwargs.get('use_ema', False)
    })
    def load_checkpoint(self, checkpoint_path: str, **kwargs) -> None:
        """Load checkpoint to CPU, defer GPU loading to setup_devices"""
        if not checkpoint_path:
            return
            
        use_ema = kwargs.get('use_ema', False)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        key = 'generator_ema' if use_ema else 'generator'
        
        if key not in state_dict:
            available_keys = list(state_dict.keys())
            print(f"Warning: Key '{key}' not found. Available: {available_keys}")
            fallback_key = 'generator_ema' if key == 'generator' else 'generator'
            key = fallback_key if fallback_key in state_dict else None
        
        # Store checkpoint for GPU loading in setup_devices
        self._checkpoint_state_dict = state_dict[key] if key else state_dict
        print("✅ Checkpoint loaded to CPU")

    def _get_model_components(self) -> Dict[str, Any]:
        """Return Self-Forcing model components in loading order"""
        return {
            'text_encoder': self.pipeline.text_encoder,
            'generator': self.pipeline.generator,
            'vae': self.pipeline.vae
        }
    
    def _materialize_and_load(self, model, name: str, gpu, checkpoint, low_memory: bool):
        """Override to add layer-by-layer loading for generator"""
        if name == 'generator' and low_memory and checkpoint and hasattr(model, 'transformer_blocks'):
            # Special handling: layer-by-layer loading for generator
            self._load_generator_layered(model, gpu, checkpoint)
        else:
            # Use base class implementation for other models
            super()._materialize_and_load(model, name, gpu, checkpoint, low_memory)
    
    def _load_generator_layered(self, generator, gpu, checkpoint):
        """Load generator transformer blocks layer-by-layer"""
        from inferix.core.memory.utils import get_cuda_free_memory_gb
        import torch
        
        num_blocks = len(generator.transformer_blocks)
        print(f"  Materializing {num_blocks} blocks from meta device...")
        
        # Load transformer blocks one by one
        for i, block in enumerate(generator.transformer_blocks):
            block.to_empty(device='cpu')
            block_prefix = f"transformer_blocks.{i}."
            block_state = {k[len(block_prefix):]: v for k, v in checkpoint.items() if k.startswith(block_prefix)}
            if block_state:
                block.load_state_dict(block_state, assign=True)
            block.to(device=gpu, dtype=torch.bfloat16)
            
            if (i + 1) % 5 == 0:
                torch.cuda.empty_cache()
                print(f"  Loaded {i+1}/{num_blocks} blocks, {get_cuda_free_memory_gb(gpu):.2f} GB free")
        
        # Load other components (conv_in, conv_out, norm_out, time_embed)
        for name in ['conv_in', 'conv_out', 'norm_out', 'time_embed']:
            if hasattr(generator, name):
                module = getattr(generator, name)
                module.to_empty(device='cpu')
                module_state = {k[len(name)+1:]: v for k, v in checkpoint.items() if k.startswith(f"{name}.")}
                if module_state:
                    module.load_state_dict(module_state, assign=True)
                module.to(device=gpu, dtype=torch.bfloat16)
        
    def load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image for I2V"""
        transform = transforms.Compose([
            transforms.Resize((480, 832)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).unsqueeze(2).to(device=self.device, dtype=torch.bfloat16)
        return image
        
    @profile_session("t2v_generation", {
        'mode': 'text_to_video'
    })
    def run_text_to_video(
        self, 
        prompts: List[str], 
        num_output_frames: int = 21, 
        num_samples: int = 1,
        output_folder: Optional[str] = None,
        save_with_index: bool = False,
        use_ema: bool = False,
        rtmp_url: Optional[str] = None,
        rtmp_fps: int = 16,
        low_memory: bool = False,
        enable_webrtc: Optional[bool] = False,
        **kwargs
    ) -> torch.Tensor:
        """[Implementation of abstract method] Run text-to-video generation"""
        session_tags = {
            'num_prompts': len(prompts),
            'num_output_frames': num_output_frames,
            'num_samples': num_samples,
            'rank': self.parallel_config.rank if self.parallel_config else 0,
            'local_rank': self.parallel_config.local_rank if self.parallel_config else 0,
            'world_size': self.parallel_config.world_size if self.parallel_config else 1,
            'ulysses_size': self.parallel_config.ulysses_size if self.parallel_config else 1,
            'ring_size': self.parallel_config.ring_size if self.parallel_config else 1
        }
        
        # Update session tags
        if self._profiling_enabled and self._profiler is not None and self._profiler.current_session is not None:
            self._profiler.current_session.tags.update(session_tags)
        
        result = self._run_inference(
            prompts=prompts,
            num_output_frames=num_output_frames,
            num_samples=num_samples,
            output_folder=output_folder,
            save_with_index=save_with_index,
            use_ema=use_ema,
            rtmp_url=rtmp_url,
            rtmp_fps=rtmp_fps,
            enable_webrtc=enable_webrtc,
            low_memory=low_memory
        )
        
        return result
        
    @profile_session("i2v_generation", {
        'mode': 'image_to_video'
    })
    def run_image_to_video(
        self,
        prompts: List[str], 
        image_path: str,
        num_output_frames: int = 21,
        num_samples: int = 1,
        output_folder: Optional[str] = None,
        save_with_index: bool = False,
        use_ema: bool = False,
        rtmp_url: Optional[str] = None,
        rtmp_fps: int = 16,
        low_memory: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """[Implementation of abstract method] Run image-to-video generation"""
        # Check distributed inference support
        if dist.is_initialized():
            raise NotImplementedError("I2V does not support distributed inference yet.")
            
        session_tags = {
            'num_prompts': len(prompts),
            'num_output_frames': num_output_frames,
            'num_samples': num_samples,
            'image_path': image_path,
            'rank': self.parallel_config.rank if self.parallel_config else 0,
            'local_rank': self.parallel_config.local_rank if self.parallel_config else 0,
            'world_size': self.parallel_config.world_size if self.parallel_config else 1,
            'ulysses_size': self.parallel_config.ulysses_size if self.parallel_config else 1,
            'ring_size': self.parallel_config.ring_size if self.parallel_config else 1
        }
        
        # Update session tags
        if self._profiling_enabled and self._profiler is not None and self._profiler.current_session is not None:
            self._profiler.current_session.tags.update(session_tags)
        
        initial_image = self.load_image(image_path)
        initial_latent = self.pipeline.vae.encode_to_latent(initial_image).to(
            device=self.device, dtype=torch.bfloat16
        )
        initial_latent = initial_latent.repeat(num_samples, 1, 1, 1, 1)
        noise_frames = num_output_frames - 1
        
        result = self._run_inference(
            prompts=prompts,
            num_output_frames=noise_frames,
            num_samples=num_samples,
            initial_latent=initial_latent,
            output_folder=output_folder,
            save_with_index=save_with_index,
            use_ema=use_ema,
            rtmp_url=rtmp_url,
            rtmp_fps=rtmp_fps,
            low_memory=low_memory
        )
        
        return result
        
    @profile_method("run_inference")
    @add_profiling_event("generation_completed", lambda result, *args, **kwargs: {
        "output_shape": list(result.shape) if result is not None else None
    })
    def _run_inference(
        self,
        prompts: List[str],
        num_output_frames: int,
        num_samples: int,
        initial_latent: Optional[torch.Tensor] = None,
        output_folder: Optional[str] = None,
        save_with_index: bool = False,
        use_ema: bool = False,
        rtmp_url: Optional[str] = None,
        rtmp_fps: int = 16,
        enable_webrtc: bool = False,
        low_memory: bool = False
    ) -> torch.Tensor:
        """Execute the core inference logic"""
        local_rank = self.parallel_config.local_rank
        rank = self.parallel_config.rank
        
        # Record model parameters if profiling is enabled
        if self._profiling_enabled and self._profiler is not None:
            # Record generator model parameters
            self._profiler.record_model_parameters(
                model_name="WanDiffusionGenerator",
                parameters_count=sum(p.numel() for p in self.pipeline.generator.parameters()),
                model_type="diffusion"
            )
            
            # Record text encoder parameters
            self._profiler.record_model_parameters(
                model_name="WanTextEncoder",
                parameters_count=sum(p.numel() for p in self.pipeline.text_encoder.parameters()),
                model_type="text_encoder"
            )
            
            # Record VAE parameters
            self._profiler.record_model_parameters(
                model_name="WanVAE",
                parameters_count=sum(p.numel() for p in self.pipeline.vae.parameters()),
                model_type="vae"
            )
        
        # Create output directory
        if output_folder and local_rank == 0:
            os.makedirs(output_folder, exist_ok=True)
                
        if dist.is_initialized():
            dist.barrier()
                
        # Initialize RTMP stream
        rtmp_streamer = None
        if rtmp_url and rank == 0:
            rtmp_streamer = PersistentRTMPStreamer()
            if not rtmp_streamer.connect(rtmp_url, width=832, height=480, fps=rtmp_fps):
                print("⚠️  RTMP streaming initialization failed; saving to local files only.")
                rtmp_streamer = None
                
        # Initialize WebRTC stream
        webrtc_streamer = None
        if enable_webrtc and rank == 0:  
            webrtc_streamer = PersistentWebRTCStreamer()
            if not webrtc_streamer.connect(width=832, height=480, fps=16, port=8000):
                print("⚠️  WebRTC streaming initialization failed")
                webrtc_streamer = None
                    
        all_videos = []

        if self._pipeline_type == "causal":
            kv_cache_manager = KVCacheManager(device=self.device)
        else:
            kv_cache_manager_pos = KVCacheManager(device=self.device)
            kv_cache_manager_neg = KVCacheManager(device=self.device)
            
        kv_cache_requests = [KVCacheRequest(f"req_{idx}") for idx in range(num_samples)]

        for prompt_idx, prompt in enumerate(prompts):
            # Prepare batch prompts
            batch_prompts = [prompt] * num_samples
                    
            # Sample noise
            sampled_noise = torch.randn(
                [num_samples, num_output_frames, 16, 60, 104],
                device=self.device,
                dtype=torch.bfloat16
            )
                    
            # Execute inference, decide whether to pass low_memory parameter based on pipeline type
            if self._pipeline_type == "causal":
                # CausalInferencePipeline supports low_memory parameter
                video, latents = self.pipeline.inference(
                    noise=sampled_noise,
                    text_prompts=batch_prompts,
                    return_latents=True,
                    initial_latent=initial_latent,
                    kv_cache_manager=kv_cache_manager,
                    kv_cache_requests=kv_cache_requests,
                    low_memory=low_memory,
                    profile=self._profiling_enabled
                )
            else:
                # CausalDiffusionInferencePipeline does not support low_memory parameter
                # Pass the parameters that are supported by this pipeline
                video, latents = self.pipeline.inference(
                    noise=sampled_noise,
                    text_prompts=batch_prompts,
                    return_latents=True,
                    initial_latent=initial_latent,
                    kv_cache_manager_pos=kv_cache_manager_pos,
                    kv_cache_manager_neg=kv_cache_manager_neg,
                    kv_cache_requests=kv_cache_requests,
                    profile=self._profiling_enabled,
                )
                    
            all_videos.append(video)
                    
            # Save and stream (only rank 0)
            if rank == 0:
                self._save_and_stream_video(
                    video, prompt_idx, prompt, num_samples,
                    output_folder, save_with_index, use_ema,
                    rtmp_streamer,
                    webrtc_streamer
                )
                        
            # Clear cache
            self.pipeline.vae.model.clear_cache()
                    
            if dist.is_initialized():
                dist.barrier()
                        
        # Clean up RTMP connection
        if rank == 0 and rtmp_streamer is not None:
            rtmp_streamer.disconnect()
            print("✅ RTMP streaming disconnected")
                
        return torch.cat(all_videos, dim=0) if all_videos else None

    @profile_method("save_and_stream_video")
    def _save_and_stream_video(
        self,
        video: torch.Tensor,
        prompt_idx: int,
        prompt: str,
        num_samples: int,
        output_folder: Optional[str],
        save_with_index: bool,
        use_ema: bool,
        rtmp_streamer,
        webrtc_streamer
    ):
        """Save video and stream it"""
        video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        video_255 = torch.clamp(video * 255.0, 0, 255).to(torch.uint8)
            
        # RTMP streaming
        if rtmp_streamer is not None:
            for sample_idx in range(num_samples):
                rtmp_streamer.stream_batch(video_255[sample_idx])     

        # WebRTC streaming
        if webrtc_streamer is not None:
            for sample_idx in range(num_samples):
                webrtc_streamer.stream_batch(video_255[sample_idx])
        # Save to file
        if output_folder:
            model_suffix = "ema" if use_ema else "regular"
            for sample_idx in range(num_samples):
                if save_with_index:
                    filename = f"{prompt_idx}-{sample_idx}_{model_suffix}.mp4"
                else:
                    safe_prompt = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt[:50]).strip("_")
                    filename = f"{safe_prompt}-{sample_idx}_{model_suffix}.mp4"
                        
                output_path = os.path.join(output_folder, filename)
                write_video(output_path, video_255[sample_idx], fps=16)
    
    def _generate_segment_with_streaming(
        self,
        prompt: str,
        initial_latent: Optional[torch.Tensor],
        stream_callback: Optional[Callable[[torch.Tensor], None]],
        segment_length: int = 21,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [Implementation of abstract method] Generate a single segment with block-wise streaming for Self-Forcing.
        
        SELF-FORCING ARCHITECTURE:
        -------------------------
        - BLOCK SIZE: 3 frames (num_frame_per_block=3)
          * Each block is generated using semi-autoregressive decoding
          * KV cache is updated after each block for autoregressive continuation
          * Block generation: ~500ms (depending on hardware)
        
        - SEGMENT SIZE: 21 frames (default) = 7 blocks
          * One complete generation cycle
          * Total time: ~3.5s (7 blocks × 500ms/block)
        
        BLOCK-WISE STREAMING FLOW:
        --------------------------
        For a 21-frame segment:
        
        Time    Block   Frames      Action
        ----    -----   ------      ------
        0.0s    0       [0,1,2]     Generate latent → Decode → Stream 3 frames
        0.5s    1       [3,4,5]     Generate latent → Decode → Stream 3 frames
        1.0s    2       [6,7,8]     Generate latent → Decode → Stream 3 frames
        1.5s    3       [9,10,11]   Generate latent → Decode → Stream 3 frames
        2.0s    4       [12,13,14]  Generate latent → Decode → Stream 3 frames
        2.5s    5       [15,16,17]  Generate latent → Decode → Stream 3 frames
        3.0s    6       [18,19,20]  Generate latent → Decode → Stream 3 frames
        3.5s    Done    21 frames   Return full segment + final latent
        
        BENEFIT: User sees first 3 frames after 0.5s instead of waiting 3.5s for all 21 frames!
        
        USAGE EXAMPLE:
        --------------
        # Example 1: Short video with real-time streaming
        pipeline.run_streaming_generation(
            prompts=['a cat walking'],
            stream_callback=webrtc_streamer.stream_batch,
            num_segments=1,
            segment_length=21  # 7 blocks of 3 frames each
        )
        # WebRTC receives frames progressively: 3 frames every ~500ms
        
        # Example 2: Long video with segment looping
        pipeline.run_streaming_generation(
            prompts=['a cat walking'],
            stream_callback=webrtc_streamer.stream_batch,
            num_segments=10,       # 10 segments
            segment_length=21,     # 21 frames per segment
            overlap_frames=3       # 3 frames overlap (1 block)
        )
        # Total: 10 segments × 21 frames - 9 overlaps × 3 frames = 183 unique frames
        # WebRTC receives: 3 frames every ~500ms for ~35 seconds of generation
        
        Args:
            prompt (str): Text prompt for current segment.
            initial_latent (Optional[torch.Tensor]): Initial latent for continuation.
                - None: T2V mode, start from noise
                - [B, 3, C, H, W]: I2V or continuation from previous segment
            stream_callback (Optional[Callable]): Callback for streaming decoded blocks.
                - Called 7 times for a 21-frame segment (once per block)
                - Receives: torch.Tensor [3, H, W, C], uint8, range [0, 255]
            segment_length (int): Number of frames to generate.
                - Must be multiple of 3 (block size)
                - Recommended: 21 (7 blocks), 24 (8 blocks), 30 (10 blocks)
            **kwargs:
                - num_samples (int): Batch size, default 1
                - low_memory (bool): Enable memory optimization, default False
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - decoded_video: [B, segment_length, H, W, C], float32, range [0, 1]
                - final_latent: [B, segment_length, C, H, W] for next segment continuation
        
        Raises:
            ValueError: If segment_length is not compatible with block size (3).
        
        Implementation Details:
            1. Validates segment_length is multiple of block_size (3)
            2. Creates block_decode_callback for progressive decoding
            3. Calls CausalInferencePipeline.inference() with block_callback
            4. Each block triggers: generate → decode → stream → continue
            5. Returns concatenated video and final latent for continuation
        """
        # Extract parameters from kwargs
        num_samples = kwargs.get('num_samples', 1)
        low_memory = kwargs.get('low_memory', False)
        
        rank = self.parallel_config.rank
        
        # Validate segment length
        num_frame_per_block = getattr(self.pipeline.args, 'num_frame_per_block', 3)
        independent_first_frame = getattr(self.pipeline.args, 'independent_first_frame', False)
        
        if initial_latent is None and independent_first_frame:
            # First segment without conditioning: 1 + N*block_size
            if (segment_length - 1) % num_frame_per_block != 0:
                valid_values = [1 + num_frame_per_block * i for i in range(1, 10)]
                raise ValueError(
                    f"For independent_first_frame mode, segment_length must be 1 + N*{num_frame_per_block}. "
                    f"Got {segment_length}. Valid values: {valid_values}"
                )
        else:
            # Continuation or non-independent mode: must be multiple of block_size
            if segment_length % num_frame_per_block != 0:
                valid_values = [num_frame_per_block * i for i in range(1, 10)]
                raise ValueError(
                    f"segment_length must be a multiple of {num_frame_per_block}. "
                    f"Got {segment_length}. Valid values: {valid_values}"
                )
        
        # Prepare inputs
        batch_prompts = [prompt] * num_samples
        noise_frames = segment_length - (initial_latent.shape[1] if initial_latent is not None else 0)
        
        sampled_noise = torch.randn(
            [num_samples, noise_frames, 16, 60, 104],
            device=self.device,
            dtype=torch.bfloat16
        )
        
        # Initialize KV cache manager
        kv_cache_manager = KVCacheManager(device=self.device)
        kv_cache_requests = [KVCacheRequest(f"stream_req_{idx}") for idx in range(num_samples)]
        
        # Create a block callback wrapper for streaming
        decoded_blocks = []
        
        def block_decode_callback(block_latent: torch.Tensor, block_index: int):
            """Decode and stream a single block immediately after generation"""
            # Only decode and stream on rank 0
            if rank != 0:
                return
            
            # Decode block to pixel space
            with torch.no_grad():
                block_video = self.pipeline.vae.decode_to_pixel(block_latent, use_cache=False)
                block_video = (block_video * 0.5 + 0.5).clamp(0, 1)
                
                # Convert to [B, T, H, W, C] format
                block_video = rearrange(block_video, 'b t c h w -> b t h w c')
                
                # Store for final concatenation
                decoded_blocks.append(block_video.cpu())
                
                # Stream via callback if provided
                if stream_callback is not None:
                    # Stream each sample in the batch
                    for sample_idx in range(block_video.shape[0]):
                        # Convert to uint8 for streaming
                        stream_frames = torch.clamp(block_video[sample_idx] * 255.0, 0, 255).to(torch.uint8)
                        stream_callback(stream_frames)
                        
                print(f"✅ Block {block_index} decoded and streamed ({block_latent.shape[1]} frames)")
        
        # Execute inference with block callback
        print(f"Generating segment: {segment_length} frames, {num_samples} sample(s)")
        
        video_latents, final_latents = self.pipeline.inference(
            noise=sampled_noise,
            text_prompts=batch_prompts,
            return_latents=True,
            initial_latent=initial_latent,
            kv_cache_manager=kv_cache_manager,
            kv_cache_requests=kv_cache_requests,
            low_memory=low_memory,
            profile=self._profiling_enabled,
            block_callback=block_decode_callback  # Pass block callback
        )
        
        # Concatenate all decoded blocks
        if rank == 0 and decoded_blocks:
            full_video = torch.cat(decoded_blocks, dim=1)  # [B, T, H, W, C]
        else:
            # Fallback: decode the entire video if block streaming wasn't used
            full_video = self.pipeline.vae.decode_to_pixel(final_latents, use_cache=False)
            full_video = (full_video * 0.5 + 0.5).clamp(0, 1)
            full_video = rearrange(full_video, 'b t c h w -> b t h w c').cpu()
        
        # Clear cache
        self.pipeline.vae.model.clear_cache()
        for req in kv_cache_requests:
            kv_cache_manager.free(req)
        
        return full_video, final_latents