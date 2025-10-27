import torch
import os
from omegaconf import OmegaConf
from typing import List, Optional, Dict, Any
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
        """Initialize the inference pipeline"""
        torch.set_grad_enabled(False)
            
        # Select pipeline type based on configuration
        if hasattr(self.config, 'denoising_step_list'):
            self.pipeline = CausalInferencePipeline(
                self.config, 
                device=self.device, 
                parallel_config=self.parallel_config,
                profiler=self._profiler  # Pass profiler to the pipeline
            )
            self._pipeline_type = "causal"
        else:
            self.pipeline = CausalDiffusionInferencePipeline(
                self.config, 
                device=self.device
            )
            self._pipeline_type = "diffusion"
                
        self.pipeline = self.pipeline.to(dtype=torch.bfloat16)
        
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
        """[Implementation of abstract method] Load model checkpoint weights"""
        use_ema = kwargs.get('use_ema', False)
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            key = 'generator_ema' if use_ema else 'generator'
            if hasattr(self.pipeline, 'generator'):
                self.pipeline.generator.load_state_dict(state_dict[key])
            else:
                # If pipeline doesn't have a generator attribute, load directly to pipeline
                self.pipeline.load_state_dict(state_dict[key])

    @profile_method("setup_devices")
    @add_profiling_event("devices_setup", lambda result, *args, **kwargs: {
        "low_memory": kwargs.get('low_memory', False)
    })
    def setup_devices(self, low_memory: bool = False):
        """Setup devices and memory management"""
        gpu = get_gpu()
            
        if low_memory:
            DynamicSwapInstaller.install_model(self.pipeline.text_encoder, device=gpu)
        else:
            self.pipeline.text_encoder.to(device=gpu)
                
        self.pipeline.generator.to(device=gpu)
        self.pipeline.vae.to(device=gpu)
        
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