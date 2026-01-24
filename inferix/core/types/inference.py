from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np
import torch

from inferix.kvcache_manager.kvcache_manager import KVCacheManager, KVCacheRequest


class DecodeMode(Enum):
    """VAE decoding timing strategy."""
    AFTER_ALL = "after_all"      # Decode after all latents generated (default)
    PER_BLOCK = "per_block"      # Decode per block (streaming/low memory)
    NO_DECODE = "no_decode"      # Return latent only, no VAE decode


class MemoryMode(Enum):
    """Memory management strategy."""
    AGGRESSIVE = "aggressive"    # Free cache aggressively (16GB VRAM)
    BALANCED = "balanced"        # Balanced (default)
    RELAXED = "relaxed"          # Keep cache for reuse (24GB+ VRAM)

@dataclass(frozen=True)
class PackedCoreAttnParams:
    # Packed sequence parameters for core_attn
    q_range: torch.Tensor
    k_range: torch.Tensor
    np_q_range: np.ndarray
    np_k_range: np.ndarray
    max_seqlen_q: int
    max_seqlen_k: int


@dataclass(frozen=True)
class PackedCrossAttnParams:
    # Packed sequence parameters for cross_attn
    q_ranges: torch.Tensor = None
    kv_ranges: torch.Tensor = None
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_kv: torch.Tensor = None
    max_seqlen_q: Optional[int] = None
    max_seqlen_kv: Optional[int] = None


@dataclass(frozen=True)
class ModelMetaArgs:
    H: int
    W: int
    cp_pad_size: int
    cp_split_sizes: List[int]
    slice_point: int
    denoising_range_num: int
    range_num: int
    extract_prefix_video_feature: bool
    fwd_extra_1st_chunk: bool
    distill_nearly_clean_chunk: bool
    clip_token_nums: int
    enable_cuda_graph: bool
    core_attn_params: PackedCoreAttnParams
    cross_attn_params: PackedCrossAttnParams


class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    def __init__(self, max_batch_size, max_sequence_length):
        self.max_sequence_length = max_sequence_length
        self.max_batch_size = max_batch_size
        self.sequence_len_offset = 0
        self.kv_cache_request = KVCacheRequest(request_id=f"magi")
        self.kv_cache_manager = KVCacheManager(device=torch.cuda.current_device())
        self.key_value_memory_dict = {}
        self.update_kv_cache = False

    # def swap_key_value_dict(self, batch_idx):
    #     "swap between batches"
    #     if len(self.key_value_memory_dict) == 0:
    #         raise ValueError("should not swap when dict in empty")

    #     for layer_number in self.key_value_memory_dict.keys():
    #         inference_key_memory, inference_value_memory = self.key_value_memory_dict[layer_number]
    #         assert len(batch_idx) == inference_key_memory.shape[1]  # make sure batch size is the same
    #         new_inference_key_memory = inference_key_memory[:, batch_idx]
    #         new_inference_value_memory = inference_value_memory[:, batch_idx]
    #         self.key_value_memory_dict[layer_number] = (new_inference_key_memory, new_inference_value_memory)