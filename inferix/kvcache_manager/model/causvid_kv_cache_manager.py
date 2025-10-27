import torch
from typing import Optional
from inferix.kvcache_manager.kvcache_manager import (
    KVCacheManager, KVCacheRequest, KVCacheRequestSpec, KVCacheSpec
)

class CausVidKVCacheManager:
    """
    Standalone KV Cache management module for transformer inference.
    
    This module handles the allocation, storage, and retrieval of key-value pairs
    during transformer inference to enable efficient autoregressive generation.
    """

    def __init__(self, layer_number: int, num_query_groups_per_partition: int, 
                hidden_size_per_attention_head: int, enable_kv_offload: bool = False):
        """
        Initialize the KV cache manager.
        
        Args:
            layer_number: The layer number this cache manager is associated with
            num_query_groups_per_partition: Number of query groups per partition (typically 12)
            hidden_size_per_attention_head: Hidden size per attention head (typically 128)
            enable_kv_offload: Whether to enable KV cache offloading to CPU
        """
        self.layer_number = layer_number
        self.num_query_groups_per_partition = num_query_groups_per_partition
        self.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.enable_kv_offload = enable_kv_offload

    def allocate_kv_cache(self, kv_cache_manager: KVCacheManager, kv_cache_request: KVCacheRequest, 
                         sequence_length: int, dtype: torch.dtype, ulysses_size: int = 1, ring_size: int = 1) -> None:
        """
        Allocate memory to store KV cache during inference.
        
        Args:
            kv_cache_manager: The underlying KV cache manager
            kv_cache_request: KV cache request object
            sequence_length: Maximum sequence length
            batch_size: Batch size
            dtype: Data type for the allocated tensor
        """
        kv_cache_request_spec = KVCacheRequestSpec(
            num_tokens=sequence_length // ring_size,
            block_size=1,  # Self-forcing uses block_size=1
            specs={
                f"layer_{self.layer_number}": KVCacheSpec(
                    num_kv_heads=self.num_query_groups_per_partition // ulysses_size,
                    head_size=self.hidden_size_per_attention_head,
                    dtype=dtype,
                    kv_offload=self.enable_kv_offload,
                    use_mla=False,
                )
            },
        )
        kv_cache_manager.allocate_slots(kv_cache_request, kv_cache_request_spec)

    def allocate_crossattn_cache(self, kv_cache_manager: KVCacheManager, kv_cache_request: KVCacheRequest, 
                                crossattn_length: int, dtype: torch.dtype) -> None:
        """
        Allocate memory to store cross-attention cache during inference.
        
        Args:
            kv_cache_manager: The underlying KV cache manager
            kv_cache_request: KV cache request object
            crossattn_length: Cross-attention sequence length (typically 512)
            batch_size: Batch size
            dtype: Data type for the allocated tensor
        """
        kv_cache_request_spec = KVCacheRequestSpec(
            num_tokens=crossattn_length,
            block_size=1,
            specs={
                f"crossattn_layer_{self.layer_number}": KVCacheSpec(
                    num_kv_heads=self.num_query_groups_per_partition,
                    head_size=self.hidden_size_per_attention_head,
                    dtype=dtype,
                    kv_offload=self.enable_kv_offload,
                    use_mla=False,
                )
            },
        )
        kv_cache_manager.allocate_slots(kv_cache_request, kv_cache_request_spec)

    def reset_kv_cache(self, kv_cache_manager: KVCacheManager, kv_cache_request: KVCacheRequest, device: torch.device) -> None:
        """
        Reset KV cache indices to zero (equivalent to clearing cache).
        
        Args:
            kv_cache_manager: The underlying KV cache manager
            kv_cache_request: KV cache request object
            device: Device for tensors
        """
        if f"layer_{self.layer_number}" in kv_cache_manager.layers(kv_cache_request):
            # Reset end indices (simulating original global_end_index and local_end_index reset)
            pass  # The underlying manager handles this through get_range parameters

    def reset_crossattn_cache(self, kv_cache_manager: KVCacheManager, kv_cache_request: KVCacheRequest) -> None:
        """
        Reset cross-attention cache initialization flags.
        
        Args:
            kv_cache_manager: The underlying KV cache manager
            kv_cache_request: KV cache request object
        """
        if f"crossattn_layer_{self.layer_number}" in kv_cache_manager.layers(kv_cache_request):
            # Reset cross-attention cache (simulating original is_init=False)
            pass  # The underlying manager handles this

    def get_kv_cache(self, kv_cache_manager: KVCacheManager, kv_cache_request: KVCacheRequest, 
                    start_index: int, length: int) -> torch.Tensor:
        """
        Get KV cache data for a specific range.
        
        Args:
            kv_cache_manager: The underlying KV cache manager
            kv_cache_request: KV cache request object
            start_index: Start index in the cache
            length: Length of data to retrieve
            
        Returns:
            KV cache tensor with shape compatible with original implementation
        """
        return kv_cache_manager.get_range(
            kv_cache_request, f"layer_{self.layer_number}", start_index, length
        ).squeeze(2)
    

    def set_kv_cache(self, kv_cache_manager: KVCacheManager, kv_cache_request: KVCacheRequest, 
                    start_index: int, k_data: torch.Tensor, v_data: torch.Tensor) -> None:
        """
        Set KV cache data at a specific range.
        
        Args:
            kv_cache_manager: The underlying KV cache manager
            kv_cache_request: KV cache request object
            start_index: Start index in the cache
            kv_data: KV data tensor to store
        """
        combined_kv = torch.stack([k_data, v_data], dim=0).unsqueeze(2)
        kv_cache_manager.set(
            kv_cache_request, f"layer_{self.layer_number}", 
            start_index, combined_kv.shape[1], combined_kv
        )

    def get_crossattn_cache(self, kv_cache_manager: KVCacheManager, kv_cache_request: KVCacheRequest) -> torch.Tensor:
        """
        Get cross-attention cache data.
        
        Args:
            kv_cache_manager: The underlying KV cache manager
            kv_cache_request: KV cache request object
            
        Returns:
            Cross-attention cache tensor
        """
        return kv_cache_manager.get(kv_cache_request, f"crossattn_layer_{self.layer_number}").squeeze(2)

    def set_crossattn_cache(self, kv_cache_manager: KVCacheManager, kv_cache_request: KVCacheRequest, 
                           k_data: torch.Tensor, v_data: torch.Tensor) -> None:
        """
        Set cross-attention cache data.
        
        Args:
            kv_cache_manager: The underlying KV cache manager
            kv_cache_request: KV cache request object
            kv_data: Cross-attention KV data tensor to store
        """
        combined_kv = torch.stack([k_data, v_data], dim=0).unsqueeze(2)

        kv_cache_manager.set(
            kv_cache_request, f"crossattn_layer_{self.layer_number}", 
            0, combined_kv.shape[1], combined_kv
        )

    def clear_cache(self, kv_cache_manager: KVCacheManager, kv_cache_request: KVCacheRequest) -> None:
        """
        Clear the KV cache for this layer.
        
        Args:
            kv_cache_manager: The underlying KV cache manager
            kv_cache_request: KV cache request object
        """
        if f"layer_{self.layer_number}" in kv_cache_manager.layers(kv_cache_request):
            kv_cache_manager.free_layer(kv_cache_request, f"layer_{self.layer_number}")
        if f"crossattn_layer_{self.layer_number}" in kv_cache_manager.layers(kv_cache_request):
            kv_cache_manager.free_layer(kv_cache_request, f"crossattn_layer_{self.layer_number}")

    def get_cache_size(self, kv_cache_manager: KVCacheManager, kv_cache_request: KVCacheRequest) -> Optional[int]:
        """
        Get the current cache size for this layer.
        
        Args:
            kv_cache_manager: The underlying KV cache manager
            kv_cache_request: KV cache request object
            
        Returns:
            Cache size in elements, or None if no cache exists
        """
        if f"layer_{self.layer_number}" in kv_cache_manager.layers(kv_cache_request):
            cache = kv_cache_manager.get_raw(kv_cache_request, f"layer_{self.layer_number}")
            return cache.numel()
        return None

    def is_cached(self, kv_cache_manager: KVCacheManager, kv_cache_request: KVCacheRequest) -> bool:
        """
        Check if this layer has cached key-value pairs.
        
        Args:
            kv_cache_manager: The underlying KV cache manager
            kv_cache_request: KV cache request object
            
        Returns:
            True if cache exists for this layer, False otherwise
        """
        return f"layer_{self.layer_number}" in kv_cache_manager.layers(kv_cache_request)


class KVCacheManagerFactory:
    """
    Factory class for creating KV cache managers with consistent configuration.
    """
    
    @staticmethod
    def create_manager(
        layer_number: int,
        num_query_groups_per_partition: int,
        hidden_size_per_attention_head: int,
        enable_kv_offload: bool
    ) -> CausVidKVCacheManager:
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
        return CausVidKVCacheManager(
            layer_number=layer_number,
            num_query_groups_per_partition=num_query_groups_per_partition,
            hidden_size_per_attention_head=hidden_size_per_attention_head,
            enable_kv_offload=enable_kv_offload
        ) 