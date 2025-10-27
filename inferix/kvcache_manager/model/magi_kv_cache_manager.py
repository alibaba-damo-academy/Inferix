# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from typing import Tuple, Optional

from inferix.core.config import EngineConfig
from inferix.core.types.inference import InferenceParams, ModelMetaArgs
from inferix.kvcache_manager.kvcache_manager import KVCacheRequestSpec, KVCacheSpec
from einops import rearrange


class MagiKVCacheManager:
    """
    Standalone KV Cache management module for transformer inference.

    This module handles the allocation, storage, and retrieval of key-value pairs
    during transformer inference to enable efficient autoregressive generation.
    """

    def __init__(self, layer_number: int, num_query_groups_per_partition: int,
                 hidden_size_per_attention_head: int, engine_config: EngineConfig):
        """
        Initialize the KV cache manager.

        Args:
            layer_number: The layer number this cache manager is associated with
            num_query_groups_per_partition: Number of query groups per partition
            hidden_size_per_attention_head: Hidden size per attention head
            engine_config: Engine configuration containing KV offload settings
        """
        self.layer_number = layer_number
        self.num_query_groups_per_partition = num_query_groups_per_partition
        self.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.engine_config = engine_config

    def allocate_key_value_memory(self, inference_params: InferenceParams, sequence_length: int, batch_size: int,
                                  dtype: torch.dtype) -> None:
        """
        Allocate memory to store KV cache during inference.

        Args:
            inference_params: Inference parameters object
            sequence_length: Maximum sequence length
            batch_size: Batch size
            dtype: Data type for the allocated tensor
        """
        kv_cache_request_spec = KVCacheRequestSpec(
            num_tokens=sequence_length,
            block_size=1,  # TODO(fty): Support block size > 1
            specs={
                f"layer_{self.layer_number}": KVCacheSpec(
                    num_kv_heads=self.num_query_groups_per_partition,
                    head_size=self.hidden_size_per_attention_head,
                    dtype=dtype,
                    kv_offload=self.engine_config.kv_offload,
                    use_mla=False,
                )
            },
        )
        inference_params.kv_cache_manager.allocate_slots(
            inference_params.kv_cache_request, kv_cache_request_spec
        )

    def _full_adjust_key_and_value(
            self,
            inference_params: InferenceParams,
            key_and_value: torch.Tensor,
            meta_args: ModelMetaArgs
    ) -> torch.Tensor:
        """
        Saves the generated key and value tensors to the end of the buffers in inference_params.
        Returns the full size keys and values from the provided inference_params.

        Args:
            inference_params: Inference parameters containing KV cache state
            key_and_value: Current key-value tensor to be cached
            meta_args: Model metadata arguments

        Returns:
            Concatenated tensor of cached and current key-value pairs
        """
        # TODO(fty): Support block size > 1
        # The input has already been reshaped from (b, sq) to (sq, b) outside the model
        with torch.cuda.nvtx.range(f"rearrange"):
            key_and_value = rearrange(key_and_value, "(nb bls) hn (coef d) -> coef nb bls hn d", coef=2, bls=1)
            key_and_value = key_and_value.contiguous()

        # Pre-allocate memory for key-values for inference
        inf_max_seq_length = inference_params.max_sequence_length
        inf_max_batch_size = inference_params.max_batch_size

        if f"layer_{self.layer_number}" not in inference_params.kv_cache_manager.layers(
                inference_params.kv_cache_request
        ):
            self.allocate_key_value_memory(
                inference_params,
                inf_max_seq_length,
                inf_max_batch_size,
                key_and_value.dtype,
            )

        inference_key_and_value_memory = inference_params.kv_cache_manager.get_raw(
            inference_params.kv_cache_request, f"layer_{self.layer_number}"
        )
        sequence_start = (
                meta_args.slice_point * meta_args.clip_token_nums * inf_max_batch_size
        )

        with torch.cuda.nvtx.range(f"get_range"):
            get_key_and_value = inference_params.kv_cache_manager.get_range(
                inference_params.kv_cache_request,
                f"layer_{self.layer_number}",
                0,
                sequence_start,
            )

        # Copy key and values
        if inference_params.update_kv_cache:
            key_and_value_total = key_and_value

            clip_size = (
                key_and_value_total.size(1) - meta_args.clip_token_nums * inf_max_batch_size
                if meta_args.distill_nearly_clean_chunk
                else key_and_value_total.size(1)
            )
            sequence_end = sequence_start + clip_size
            assert sequence_end <= inference_key_and_value_memory.size(1)
            # Update KV cache
            inference_params.kv_cache_manager.set(
                inference_params.kv_cache_request,
                f"layer_{self.layer_number}",
                sequence_start,
                clip_size,
                key_and_value_total[:, :clip_size],
            )

        with torch.cuda.nvtx.range(f"cat"):
            ret = torch.cat([get_key_and_value, key_and_value], dim=1)
        return ret

    def adjust_key_and_value_for_inference(
            self,
            key_and_value: torch.Tensor,
            inference_params: Optional[InferenceParams],
            meta_args: ModelMetaArgs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adjust key and value tensors for inference, handling KV cache management.

        Args:
            key_and_value: Combined key-value tensor
            inference_params: Inference parameters (can be None for no caching)
            meta_args: Model metadata arguments

        Returns:
            Tuple of (key, value) tensors
        """
        if inference_params is None:
            return torch.chunk(key_and_value, 2, dim=-1)

        # Only update KV cache when necessary, covering three conditions:
        # 1. Extract prefix video clean feature
        # 2. The first chunk of current KV is clean, and we need to save its features
        # 3. Previous chunk is clean and we need to save/load its features
        if (meta_args.extract_prefix_video_feature or
                meta_args.fwd_extra_1st_chunk or
                meta_args.slice_point > 0):
            key_and_value = self._full_adjust_key_and_value(inference_params, key_and_value, meta_args)
            with torch.cuda.nvtx.range(f"rearrange_back"):
                key_and_value = rearrange(key_and_value, "coef nb bls hn d -> coef (nb bls) hn d")
            key, value = key_and_value[0], key_and_value[1]
            return key.contiguous(), value.contiguous()

        key, value = torch.chunk(key_and_value, 2, dim=-1)
        return key.contiguous(), value.contiguous()

    def clear_cache(self, inference_params: InferenceParams) -> None:
        """
        Clear the KV cache for this layer.

        Args:
            inference_params: Inference parameters containing the cache to clear
        """
        inference_params.kv_cache_manager.free_layer(
            inference_params.kv_cache_request, f"layer_{self.layer_number}"
        )

    def get_cache_size(self, inference_params: InferenceParams) -> Optional[int]:
        """
        Get the current cache size for this layer.

        Args:
            inference_params: Inference parameters containing the cache

        Returns:
            Cache size in elements, or None if no cache exists
        """
        if self.layer_number in inference_params.kv_cache_manager.layers(
                inference_params.kv_cache_request
        ):
            cache = inference_params.kv_cache_manager.get_raw(
                inference_params.kv_cache_request, f"layer_{self.layer_number}"
            )
            return cache.numel()
        return None

    def is_cached(self, inference_params: InferenceParams) -> bool:
        """
        Check if this layer has cached key-value pairs.

        Args:
            inference_params: Inference parameters containing the cache

        Returns:
            True if cache exists for this layer, False otherwise
        """
        return self.layer_number in inference_params.kv_cache_manager.layers(
            inference_params.kv_cache_request
        )


class KVCacheManagerFactory:
    """
    Factory class for creating KV cache managers with consistent configuration.
    """

    @staticmethod
    def create_manager(
            layer_number: int,
            num_query_groups_per_partition: int,
            hidden_size_per_attention_head: int,
            engine_config: EngineConfig
    ) -> MagiKVCacheManager:
        """
        Create a new KV cache manager instance.

        Args:
            layer_number: The layer number for the cache manager
            num_query_groups_per_partition: Number of query groups per partition
            hidden_size_per_attention_head: Hidden size per attention head
            engine_config: Engine configuration

        Returns:
            Configured KV cache manager instance
        """
        return MagiKVCacheManager(
            layer_number=layer_number,
            num_query_groups_per_partition=num_query_groups_per_partition,
            hidden_size_per_attention_head=hidden_size_per_attention_head,
            engine_config=engine_config
        )