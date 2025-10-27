import torch
from typing import List, Optional
import torch.distributed as dist
from inferix.kvcache_manager.kvcache_manager import KVCacheManager
from inferix.models.causvid import WanDiffusionWrapper, WanVAEWrapper, WanTextEncoder
from inferix.models.wan_base import ParallelConfig

class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            wan_base_model_path,
            device,
            enable_kv_offload,
            parallel_config: Optional[ParallelConfig] = None
    ):
        super().__init__()

        self.parallel_config = parallel_config

        # Step 1: Initialize all models
        # Get model path from configuration (following README path convention)
        model_path = wan_base_model_path if wan_base_model_path is not None else 'weights/Wan2.1-T2V-1.3B'
        self.generator = WanDiffusionWrapper(
            model_path=model_path,
            **getattr(args, "model_kwargs", {}), enable_kv_offload=enable_kv_offload, parallel_config=parallel_config).to(device)
        self.text_encoder = WanTextEncoder(model_path=model_path)
        self.vae = WanVAEWrapper(model_path=model_path)

        if dist.is_initialized():
            dist.barrier()

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        self.denoising_step_list = self.denoising_step_list[:-1]
        if args.warp_denoising_step: 
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))).cuda()
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        if self.parallel_config is not None and self.parallel_config.world_size > 1:
            self.per_rank_frame_seq_length = self.frame_seq_length // self.parallel_config.world_size
        else:
            self.per_rank_frame_seq_length = self.frame_seq_length

        self.is_kv_cache_initialized = False
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)

        if self.parallel_config.rank == 0:
            print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def _initialize_kv_cache(self, kv_cache_manager, kv_cache_requests, dtype):
        """
        Initialize a Per-GPU self-attention cache for the Wan model.
        """
        ulysses_size = self.parallel_config.ulysses_size if self.parallel_config is not None else 1
        ring_size = self.parallel_config.ring_size if self.parallel_config is not None else 1
        for layer_idx in range(self.num_transformer_blocks):
            for kv_cache_request in kv_cache_requests:
                self.generator.model.blocks[layer_idx].kv_cache_manager.allocate_kv_cache(kv_cache_manager=kv_cache_manager, kv_cache_request=kv_cache_request, sequence_length=32760, dtype=dtype, ulysses_size=ulysses_size, ring_size=ring_size)

    def _initialize_crossattn_cache(self, kv_cache_manager, kv_cache_requests, dtype):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        for layer_idx in range(self.num_transformer_blocks):
            for kv_cache_request in kv_cache_requests:
                self.generator.model.blocks[layer_idx].kv_cache_manager.allocate_crossattn_cache(kv_cache_manager=kv_cache_manager, kv_cache_request=kv_cache_request, crossattn_length=512, dtype=dtype)

    def _reset_crossattn_cache(self):
        """
        Reset cross-attention cache for the Wan model.
        """
        for layer_idx in range(self.num_transformer_blocks):
            self.generator.model.blocks[layer_idx].is_cross_attn_init = False
    
    def clear_cache(self, kv_cache_manager, kv_cache_requests):
        """
        Clear self-attention and cross-attention cache for the Wan model.
        """
        for layer_idx in range(self.num_transformer_blocks):
            for kv_cache_request in kv_cache_requests:
                self.generator.model.blocks[layer_idx].kv_cache_manager.clear_cache(kv_cache_manager=kv_cache_manager, kv_cache_request=kv_cache_request)
        self.is_kv_cache_initialized = False

    
    def inference(self, noise: torch.Tensor, text_prompts: List[str], start_latents: Optional[torch.Tensor], return_latents: bool = True, kv_cache_manager: Optional[KVCacheManager] = None, kv_cache_requests: Optional[List] = None) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        output = torch.zeros(
            [batch_size, num_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        if not self.is_kv_cache_initialized:
            self._initialize_kv_cache(
                kv_cache_manager,
                kv_cache_requests,
                dtype=noise.dtype,
            )
            self._initialize_crossattn_cache(
                kv_cache_manager,
                kv_cache_requests,
                dtype=noise.dtype,
            )
            self.is_kv_cache_initialized = True
        else:
            self._reset_crossattn_cache()
            

        # Safety check: Ensure start_latents is not None before accessing its attributes
        if start_latents is not None:
            num_input_blocks = start_latents.shape[1] // self.num_frame_per_block
        else:
            num_input_blocks = 0

        num_blocks = num_frames // self.num_frame_per_block
        
        for block_index in range(num_blocks):
            noisy_input = noise[:, block_index *
                                self.num_frame_per_block:(block_index + 1) * self.num_frame_per_block]

            if start_latents is not None and block_index < num_input_blocks:
                timestep = torch.ones(
                    [batch_size, self.num_frame_per_block], device=noise.device, dtype=torch.int64) * 0

                current_ref_latents = start_latents[:, block_index * self.num_frame_per_block:(
                    block_index + 1) * self.num_frame_per_block]
                output[:, block_index * self.num_frame_per_block:(
                    block_index + 1) * self.num_frame_per_block] = current_ref_latents

                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                    current_end=(block_index + 1) * self.num_frame_per_block * self.frame_seq_length,
                    kv_start=block_index * self.num_frame_per_block * self.per_rank_frame_seq_length,
                    kv_end=(block_index + 1) * self.num_frame_per_block * self.per_rank_frame_seq_length,
                    kv_cache_manager=kv_cache_manager,
                    kv_cache_requests=kv_cache_requests,
                )
                continue

            denoised_pred = None  # Initialize variable to avoid unbound error
            timestep = None  # Initialize variable to avoid unbound error
            
            for index, current_timestep in enumerate(self.denoising_step_list):
                # set current timestep
                timestep = torch.ones(
                    [batch_size, self.num_frame_per_block], device=noise.device, dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                        current_end=(block_index + 1) * self.num_frame_per_block * self.frame_seq_length,
                        kv_start=block_index * self.num_frame_per_block * self.per_rank_frame_seq_length,
                        kv_end=(block_index + 1) * self.num_frame_per_block * self.per_rank_frame_seq_length,
                        kv_cache_manager=kv_cache_manager,
                        kv_cache_requests=kv_cache_requests,
                    )
                    
                    # Safety check: Ensure no index out of bounds
                    if index + 1 >= len(self.denoising_step_list):
                        raise IndexError(f"Index {index + 1} out of bounds for denoising_step_list of length {len(self.denoising_step_list)}")
                    
                    next_timestep = self.denoising_step_list[index + 1]
                    
                    # Fix device inconsistency issue: Use the same device as denoised_pred
                    timestep_tensor = next_timestep * torch.ones(
                        [batch_size], device=denoised_pred.device, dtype=torch.long
                    )
                    
                    # Safely handle add_noise return value and reshape
                    noisy_input_flat = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        timestep_tensor
                    )
                    # Ensure result is not None before view operation
                    if noisy_input_flat is not None:
                        noisy_input = noisy_input_flat.view(denoised_pred.shape)
                    else:
                        raise RuntimeError("scheduler.add_noise returned None")
                else:
                    # for getting real output
                    denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                        current_end=(block_index + 1) * self.num_frame_per_block * self.frame_seq_length,
                        kv_start=block_index * self.num_frame_per_block * self.per_rank_frame_seq_length,
                        kv_end=(block_index + 1) * self.num_frame_per_block * self.per_rank_frame_seq_length,
                        kv_cache_manager=kv_cache_manager,
                        kv_cache_requests=kv_cache_requests,
                    )

            if denoised_pred is not None and timestep is not None:
                output[:, block_index * self.num_frame_per_block:(
                    block_index + 1) * self.num_frame_per_block] = denoised_pred

                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                    current_end=(block_index + 1) * self.num_frame_per_block * self.frame_seq_length,
                    kv_start=block_index * self.num_frame_per_block * self.per_rank_frame_seq_length,
                    kv_end=(block_index + 1) * self.num_frame_per_block * self.per_rank_frame_seq_length,
                    kv_cache_manager=kv_cache_manager,
                    kv_cache_requests=kv_cache_requests,
                )
            else:
                raise RuntimeError(f"denoised_pred or timestep is None after denoising loop for block {block_index}")

        # Step 3: Decode the output
        video = self.vae.decode_to_pixel(output)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if dist.is_initialized():
            dist.barrier()

        if return_latents:
            return video, output
        else:
            return video