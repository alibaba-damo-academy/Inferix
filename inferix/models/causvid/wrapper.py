import types
from typing import List, Optional
import torch
import os

from inferix.models.schedulers.flow_match import FlowMatchScheduler
from inferix.kvcache_manager.kvcache_manager import KVCacheManager
from inferix.models.wan_base.text_encoder import HuggingfaceTokenizer, umt5_xxl
from inferix.models.wan_base import _video_vae, ParallelConfig
from inferix.models.causvid import CausalWanModel


class WanTextEncoder(torch.nn.Module):
    def __init__(self, model_path: str, tokenizer_path: Optional[str] = None, seq_len: int = 512) -> None:
        super().__init__()
        
        if tokenizer_path is None:
            tokenizer_path = os.path.join(model_path, "google/umt5-xxl/")

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        
        # Build model weight path
        model_weight_path = os.path.join(model_path, "models_t5_umt5-xxl-enc-bf16.pth")
        if not os.path.exists(model_weight_path):
            raise FileNotFoundError(f"Text encoder weights not found at: {model_weight_path}")
            
        self.text_encoder.load_state_dict(
            torch.load(model_weight_path, map_location='cpu', weights_only=False)
        )

        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=seq_len, clean='whitespace')

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }

class WanVAEWrapper(torch.nn.Module):
    def __init__(self, model_path: str, z_dim: int = 16):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # Build VAE model weight path
        vae_weight_path = os.path.join(model_path, "Wan2.1_VAE.pth")
        if not os.path.exists(vae_weight_path):
            raise FileNotFoundError(f"VAE weights not found at: {vae_weight_path}")

        # init model
        self.model = _video_vae(
            pretrained_path=vae_weight_path,
            z_dim=z_dim,
        ).eval().requires_grad_(False)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False, chunk_size: int = 2) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        # Decode in chunks to reduce peak memory usage
        # Process each batch independently
        for batch_idx, u in enumerate(zs):
            batch_chunks = []
            num_frames = u.shape[1]  # u: [C, T, H, W]
            
            # Split temporal dimension into chunks
            for chunk_start in range(0, num_frames, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_frames)
                # Extract chunk along temporal axis: [C, chunk_size, H, W]
                frame_chunk = u[:, chunk_start:chunk_end, :, :]
                
                # Decode this chunk
                decoded_chunk = decode_function(frame_chunk.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0)
                batch_chunks.append(decoded_chunk)
                
                # Clear VAE internal cache after each chunk to free memory
                self.model.clear_cache()
                
                # Clear GPU cache after each chunk (only affects GPU allocator, not data)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Concatenate chunks for this batch: [C, T, H, W]
            batch_output = torch.cat(batch_chunks, dim=1)
            output.append(batch_output)
        
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_path: str,
            timestep_shift: float = 8.0,
            enable_kv_offload: bool = True,
            parallel_config: Optional[ParallelConfig] = None
    ):
        super().__init__()

        self.parallel_config = parallel_config
        self.enable_kv_offload = enable_kv_offload
        if parallel_config is None:
            self.parallel_config = ParallelConfig()

        # init ulysses and ring
        if self.parallel_config.ulysses_size > 1 or self.parallel_config.ring_size > 1:
            from xfuser.core.distributed import (
                init_distributed_environment,
                initialize_model_parallel,
            )
            init_distributed_environment(
                rank=self.parallel_config.rank,
                world_size=self.parallel_config.world_size,
                local_rank=self.parallel_config.local_rank,
            )

            initialize_model_parallel(
                sequence_parallel_degree=self.parallel_config.world_size,
                ring_degree=self.parallel_config.ring_size,
                ulysses_degree=self.parallel_config.ulysses_size,
            )

        # Build model path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")

        self.model = CausalWanModel.from_pretrained(
            model_path,
            enable_kv_offload=self.enable_kv_offload,
            parallel_config=self.parallel_config)
        
        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = False

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 32760
        self.post_init()


    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                        scheduler.sigmas,
                                                        scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self, noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor,
        kv_start: Optional[int] = None,
        kv_end: Optional[int] = None,
        current_start: Optional[int] = None,
        current_end: Optional[int] = None,
        kv_cache_manager: Optional[KVCacheManager] = None,
        kv_cache_requests: Optional[List] = None,
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep


        flow_pred = self.model(
            noisy_image_or_video.permute(0, 2, 1, 3, 4),
            t=input_timestep, context=prompt_embeds,
            seq_len=self.seq_len,
            kv_start=kv_start,
            kv_end=kv_end,
            current_start=current_start,
            current_end=current_end,
            kv_cache_manager=kv_cache_manager,
            kv_cache_requests=kv_cache_requests,
        ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        return pred_x0
    
    def get_scheduler(self):
        """
        Update the current scheduler with the interface's static method
        """
        from inferix.models.schedulers.flow_match import SchedulerInterface

        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()