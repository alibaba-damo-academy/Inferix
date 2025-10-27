from typing import List, Optional
from contextlib import contextmanager

import torch

from inferix.models.self_forcing import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from inferix.core.memory.utils import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation
from inferix.models.wan_base import ParallelConfig
from inferix.kvcache_manager.kvcache_manager import KVCacheRequest, KVCacheManager


class PerformanceProfiler:
    """Performance profiler using context managers for elegant timing"""
    
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        if enabled:
            self.events = {}
            self.current_stage = None
    
    def __bool__(self):
        return self.enabled
    
    @contextmanager
    def stage(self, name: str):
        """Context manager for timing a stage"""
        if not self.enabled:
            yield
            return
            
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        try:
            self.current_stage = name
            yield
        finally:
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            self.events[name] = elapsed_time
            self.current_stage = None
    
    def record_block_time(self, block_index: int, time_ms: float):
        """Record block computation time"""
        if not self.enabled:
            return
        self.events[f"block_{block_index}"] = time_ms
    
    def get_results(self):
        """Get profiling results"""
        return self.events if self.enabled else {}


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None,
            parallel_config: Optional[ParallelConfig] = None,
            profiler=None
    ):
        super().__init__()

        self.parallel_config = parallel_config
        self._profiler = profiler  # Store external profiler reference

        # Step 1: Initialize all models
        # Get model path from configuration (following README path convention)
        model_path = getattr(args, 'model_path', 'weights/Wan2.1-T2V-1.3B')
        
        self.generator = WanDiffusionWrapper(
            model_path=model_path,
            **getattr(args, "model_kwargs", {}), is_causal=True, parallel_config=parallel_config) if generator is None else generator
        
        self.text_encoder = WanTextEncoder(model_path=model_path) if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper(model_path=model_path) if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache_meta = None
        self.crossattn_cache_meta = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        if self.parallel_config.rank == 0:
            print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        kv_cache_manager: KVCacheManager,
        kv_cache_requests: List[KVCacheRequest],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        # Create performance profiler instance
        perf_profiler = PerformanceProfiler(enabled=profile)
        
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        with perf_profiler.stage("initialization"):
            if self.kv_cache_meta is None:
                self._initialize_kv_cache(
                    kv_cache_manager,
                    kv_cache_requests,
                    dtype=noise.dtype,
                )
            else:
                # reset kv cache
                for block_index in range(len(self.kv_cache_meta)):
                    self.kv_cache_meta[block_index]["global_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)
                    self.kv_cache_meta[block_index]["local_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)

            if self.crossattn_cache_meta is None:
                self._initialize_crossattn_cache(
                    kv_cache_manager,
                    kv_cache_requests,
                    dtype=noise.dtype,
                )
            else:
                # reset cross attn cache
                for block_index in range(self.num_transformer_blocks):
                    self.crossattn_cache_meta[block_index]["is_init"] = False

            # Step 2: Cache context feature
            current_start_frame = 0
            if initial_latent is not None:
                timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
                if self.independent_first_frame:
                    # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                    assert (num_input_frames - 1) % self.num_frame_per_block == 0
                    num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                    output[:, :1] = initial_latent[:, :1]
                    self.generator(
                        noisy_image_or_video=initial_latent[:, :1],
                        conditional_dict=conditional_dict,
                        timestep=timestep * 0,
                        kv_cache_meta=self.kv_cache_meta,
                        crossattn_cache_meta=self.crossattn_cache_meta,
                        current_start=current_start_frame * self.frame_seq_length,
                        kv_cache_manager=kv_cache_manager,
                        kv_cache_requests=kv_cache_requests,
                    )
                    current_start_frame += 1
                else:
                    # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                    assert num_input_frames % self.num_frame_per_block == 0
                    num_input_blocks = num_input_frames // self.num_frame_per_block

                for _ in range(num_input_blocks):
                    current_ref_latents = \
                        initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                    output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                    self.generator(
                        noisy_image_or_video=current_ref_latents,
                        conditional_dict=conditional_dict,
                        timestep=timestep * 0,
                        kv_cache_meta=self.kv_cache_meta,
                        crossattn_cache_meta=self.crossattn_cache_meta,
                        current_start=current_start_frame * self.frame_seq_length,
                        kv_cache_manager=kv_cache_manager,
                        kv_cache_requests=kv_cache_requests,
                    )
                    current_start_frame += self.num_frame_per_block

        # Step 3: Temporal denoising loop
        with perf_profiler.stage("diffusion_generation"):
            all_num_frames = [self.num_frame_per_block] * num_blocks
            if self.independent_first_frame and initial_latent is None:
                all_num_frames = [1] + all_num_frames
            
            block_times = []
            for block_index, current_num_frames in enumerate(all_num_frames):
                # Initialize block timing events
                block_start_time = None
                block_end_time = None
                block_time = 0.0
                
                if profile:
                    block_start_time = torch.cuda.Event(enable_timing=True)
                    block_end_time = torch.cuda.Event(enable_timing=True)
                    block_start_time.record()

                noisy_input = noise[
                    :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

                # Initialize variables that might be used later
                denoised_pred = None
                timestep = None

                # Step 3.1: Spatial denoising loop
                for index, current_timestep in enumerate(self.denoising_step_list):
                    # Record diffusion step with external profiler if available
                    diffusion_start_time = None
                    diffusion_end_time = None
                    if self._profiler is not None and hasattr(self._profiler, 'record_diffusion_step'):
                        diffusion_start_time = torch.cuda.Event(enable_timing=True)
                        diffusion_end_time = torch.cuda.Event(enable_timing=True)
                        diffusion_start_time.record()
                    
                    # print(f"current_timestep: {current_timestep}")
                    # set current timestep
                    timestep = torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64) * current_timestep

                    if index < len(self.denoising_step_list) - 1:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache_meta=self.kv_cache_meta,
                            crossattn_cache_meta=self.crossattn_cache_meta,
                            current_start=current_start_frame * self.frame_seq_length,
                            kv_cache_manager=kv_cache_manager,
                            kv_cache_requests=kv_cache_requests,
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                    else:
                        # for getting real output
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache_meta=self.kv_cache_meta,
                            crossattn_cache_meta=self.crossattn_cache_meta,
                            current_start=current_start_frame * self.frame_seq_length,
                            kv_cache_manager=kv_cache_manager,
                            kv_cache_requests=kv_cache_requests,
                        )
                    
                    # Record diffusion step completion with external profiler
                    if (self._profiler is not None and hasattr(self._profiler, 'record_diffusion_step') 
                        and diffusion_start_time is not None and diffusion_end_time is not None):
                        diffusion_end_time.record()
                        torch.cuda.synchronize()
                        diffusion_time_ms = diffusion_start_time.elapsed_time(diffusion_end_time)
                        
                        # Record the diffusion step in the external profiler
                        try:
                            self._profiler.record_diffusion_step(
                                step=index,
                                timestep=current_timestep.item() / 1000.0,  # Normalize timestep
                                block_size=current_num_frames,
                                computation_time_ms=diffusion_time_ms,
                                guidance_scale=getattr(self.args, 'guidance_scale', None)
                            )
                        except Exception as e:
                            # Silently ignore profiler errors to avoid breaking the pipeline
                            pass

                # Step 3.2: record the model's output
                if denoised_pred is not None:
                    output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

                # Step 3.3: rerun with timestep zero to update KV cache using clean context
                if timestep is not None:
                    context_timestep = torch.ones_like(timestep) * self.args.context_noise
                    if denoised_pred is not None:
                        self.generator(
                            noisy_image_or_video=denoised_pred,
                            conditional_dict=conditional_dict,
                            timestep=context_timestep,
                            kv_cache_meta=self.kv_cache_meta,
                            crossattn_cache_meta=self.crossattn_cache_meta,
                            current_start=current_start_frame * self.frame_seq_length,
                            kv_cache_manager=kv_cache_manager,
                            kv_cache_requests=kv_cache_requests,
                        )

                if profile and block_start_time is not None and block_end_time is not None:
                    block_end_time.record()
                    torch.cuda.synchronize()
                    block_time = block_start_time.elapsed_time(block_end_time)
                    block_times.append(block_time)
                    perf_profiler.record_block_time(block_index, block_time)
                    
                    # Record block computation with external profiler if available
                    if self._profiler is not None and hasattr(self._profiler, 'record_block_computation'):
                        try:
                            # Estimate memory usage during block computation
                            memory_used_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                            
                            self._profiler.record_block_computation(
                                block_index=block_index,
                                block_size=current_num_frames,
                                computation_time_ms=block_time,
                                memory_usage_mb=memory_used_mb
                            )
                        except Exception as e:
                            # Silently ignore profiler errors to avoid breaking the pipeline
                            pass

                # Step 3.4: update the start and end frame indices
                current_start_frame += current_num_frames

        # Step 4: Decode the output
        with perf_profiler.stage("vae_decoding"):
            video = self.vae.decode_to_pixel(output, use_cache=False)
            video = (video * 0.5 + 0.5).clamp(0, 1)

        # Print profiling results if enabled
        if profile:
            events = perf_profiler.get_results()
            init_time = events.get("initialization", 0)
            diffusion_time = events.get("diffusion_generation", 0)
            vae_time = events.get("vae_decoding", 0)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        else:
            return video

    def _initialize_kv_cache(self, 
                             kv_cache_manager: KVCacheManager,  
                             kv_cache_requests: List[KVCacheRequest], 
                             dtype):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache_meta = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 32760

        ulysses_size = self.parallel_config.ulysses_size if self.parallel_config is not None else 1
        ring_size = self.parallel_config.ring_size if self.parallel_config is not None else 1
        for layer_idx in range(self.num_transformer_blocks):
            for kv_cache_request in kv_cache_requests:
                self.generator.model.blocks[layer_idx].kv_cache_manager.allocate_kv_cache(kv_cache_manager=kv_cache_manager, kv_cache_request=kv_cache_request, sequence_length=kv_cache_size, dtype=dtype, ulysses_size=ulysses_size, ring_size=ring_size)

        device = kv_cache_manager.device
        for _ in range(self.num_transformer_blocks):
            kv_cache_meta.append({
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache_meta = kv_cache_meta  # always store the clean cache

    def _initialize_crossattn_cache(self, 
                                    kv_cache_manager: KVCacheManager, 
                                    kv_cache_requests: List[KVCacheRequest], 
                                    dtype):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache_meta = []

        for layer_idx in range(self.num_transformer_blocks):
            for kv_cache_request in kv_cache_requests:
                self.generator.model.blocks[layer_idx].kv_cache_manager.allocate_crossattn_cache(kv_cache_manager=kv_cache_manager, kv_cache_request=kv_cache_request, crossattn_length=512, dtype=dtype)

        for _ in range(self.num_transformer_blocks):
            crossattn_cache_meta.append({
                "is_init": False
            })

        self.crossattn_cache_meta = crossattn_cache_meta  # always store the clean cache
    
    def clear_cache(self, kv_cache_manager, kv_cache_requests):
        """
        Clear self-attention and cross-attention cache for the Wan model.
        """
        for layer_idx in range(self.num_transformer_blocks):
            for kv_cache_request in kv_cache_requests:
                self.generator.model.blocks[layer_idx].kv_cache_manager.clear_cache(kv_cache_manager=kv_cache_manager, kv_cache_request=kv_cache_request)
        self.kv_cache_meta = None
        self.crossattn_cache_meta = None