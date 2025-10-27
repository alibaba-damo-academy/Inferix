from typing import List, Sequence, Union, KeysView, Tuple
from dataclasses import dataclass
from typing import Union, List, KeysView, Sequence
import torch


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def align(a: int, b: int) -> int:
    return (a + b - 1) // b * b


def get_dtype_size(dtype: torch.dtype) -> int:
    """Get the size of the data type in bytes."""
    return torch.tensor([], dtype=dtype).element_size()


@dataclass(frozen=True)
class KVCacheRequest: # prompt
    request_id: str


@dataclass(frozen=True)
class KVCacheSpec:
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    kv_offload: bool
    use_mla: bool


@dataclass
class KVCacheRequestSpec:
    num_tokens: int
    block_size: int
    specs: dict[str, KVCacheSpec]


@dataclass(frozen=True)
class KVCacheTensorSpec:
    size: int
    num_tokens: int
    num_blocks: int
    block_size: int
    spec: KVCacheSpec


@dataclass
class KVCaches:
    tensors: dict[str, torch.Tensor] # layer_name -> KVCache Tensor
    specs: dict[str, KVCacheTensorSpec] # layer_name -> KVCacheTensorSpec


class KVCacheManager:
    def __init__(self, device: Union[str, torch.device, int]):
        self.device = torch.device(device)
        self.offload_device: torch.device = torch.device(torch.cpu.current_device())
        self.request_to_kv_caches: dict[str, KVCaches] = {}

    def allocate_slots(
        self,
        req: KVCacheRequest,
        spec: KVCacheRequestSpec,
    ) -> KVCaches:
        with torch.cuda.nvtx.range(f"allocate_slots_{req.request_id}"):
            return self._allocate_slots(req, spec)

    def _allocate_slots(
        self,
        req: KVCacheRequest,
        spec: KVCacheRequestSpec,
    ) -> KVCaches:
        """
        Allocate slots for a request.
        """
        num_blocks = cdiv(spec.num_tokens, spec.block_size)
        num_tokens_aligned = align(spec.num_tokens, spec.block_size)
        tensor_specs: dict[str, KVCacheTensorSpec] = {}
        for layer_name, kv_cache_spec in spec.specs.items():
            coef = 2 if not kv_cache_spec.use_mla else 1
            size = (
                coef
                * num_tokens_aligned
                * kv_cache_spec.num_kv_heads
                * kv_cache_spec.head_size
                * get_dtype_size(kv_cache_spec.dtype)
            )
            tensor_specs[layer_name] = KVCacheTensorSpec(
                size=size,
                num_tokens=num_tokens_aligned,
                num_blocks=num_blocks,
                block_size=spec.block_size,
                spec=kv_cache_spec,
            )
        tensors = self._allocate_kv_cache_tensors(tensor_specs)

        kv_caches = self.request_to_kv_caches.setdefault(
            req.request_id, KVCaches(tensors={}, specs={})
        )
        for layer_name, tensor in tensors.items():
            if layer_name in kv_caches.tensors:
                raise ValueError(f"Layer {layer_name} already exists")
            kv_caches.tensors[layer_name] = tensor
        for layer_name, tensor_spec in tensor_specs.items():
            if layer_name in kv_caches.specs:
                raise ValueError(f"Layer {layer_name} already exists")
            kv_caches.specs[layer_name] = tensor_spec

        return kv_caches

    def free(self, req: KVCacheRequest):
        """
        Free slots for a request.
        """
        del self.request_to_kv_caches[req.request_id]

    def free_layer(self, req: KVCacheRequest, layer_name: str):
        """
        Free slots for a layer of respective request.
        """
        del self.request_to_kv_caches[req.request_id].tensors[layer_name]
        del self.request_to_kv_caches[req.request_id].specs[layer_name]


    def select(self, req: KVCacheRequest, layer_name: str, block_indices: List[int]):
        """
        Select slots for a request of specific layer and block_index.
        """
        return (
            self.request_to_kv_caches[req.request_id]
            .tensors[layer_name][:, block_indices, ...]
            .to(self.device)
        )

    def layers(self, req: KVCacheRequest) -> KeysView[str] | Sequence[str]:
        """
        Get layers for a request.
        """
        if req.request_id not in self.request_to_kv_caches:
            return ()
        return self.request_to_kv_caches[req.request_id].tensors.keys()

    def get(self, req: KVCacheRequest, layer_name: str):
        """
        Get slots for a request.
        """
        return self.get_raw(req, layer_name).to(self.device)

    def get_range(
        self, req: KVCacheRequest, layer_name: str, start: int, length: int
    ) -> torch.Tensor:
        """
        Get range of slots for a request.
        """
        tensor = self.get_raw(req, layer_name)
        if self.request_to_kv_caches[req.request_id].specs[layer_name].spec.use_mla:
            return tensor[0:1, start : start + length, ...].to(
                self.device, non_blocking=True
            )
        else:
            coef, nb, blks, nh, hs = tensor.shape
            ret = torch.empty(
                (coef, length, blks, nh, hs), dtype=tensor.dtype, device=self.device
            )
            ret[0] = tensor[0:1, start : start + length, ...] # K
            ret[1] = tensor[1:2, start : start + length, ...] # V
            return ret

    def get_raw(self, req: KVCacheRequest, layer_name: str):
        """
        Get raw slots for a request.
        """
        return self.request_to_kv_caches[req.request_id].tensors[layer_name]

    def get_range_raw(
        self, req: KVCacheRequest, layer_name: str, start: int, length: int
    ):
        """
        Get range of raw slots for a request.
        """
        tensor = self.get_raw(req, layer_name)
        return tensor[:, start : start + length, ...]

    def layer_spec(self, req: KVCacheRequest, layer_name: str):
        """
        Get spec for a layer.
        """
        return self.request_to_kv_caches[req.request_id].specs[layer_name]

    def set(
        self,
        req: KVCacheRequest,
        layer_name: str,
        start: int,
        size: int,
        new_kv: torch.Tensor,
    ) -> None:
        with torch.cuda.nvtx.range(f"set_{req.request_id}_{layer_name}_{start}_{size}"):
            self._set(req, layer_name, start, size, new_kv)

    def _set(
        self,
        req: KVCacheRequest,
        layer_name: str,
        start: int,
        size: int,
        new_kv: torch.Tensor,
    ) -> None:
        """
        Partial value set for a request with offload.
        """
        tensor_spec = self.request_to_kv_caches[req.request_id].specs[layer_name]
        assert len(new_kv) == 2

        cache_tensor = self.request_to_kv_caches[req.request_id].tensors[layer_name]
        cache_tensor[0, start : start + size, ...] = new_kv[0]
        if not tensor_spec.spec.use_mla:
            cache_tensor[1, start : start + size, ...] = new_kv[1]
        
    def _allocate_kv_cache_tensors(
        self,
        specs: dict[str, KVCacheTensorSpec],
    ) -> dict[str, torch.Tensor]:
        """
        Allocate KV cache tensors for a layer.
        Shape: (coef(2 for KV, 1 for MLA), num_blocks, block_size, num_kv_heads, head_size)
        """
        tensors = {}
        for layer_name, spec in specs.items():
            tensors[layer_name] = torch.empty(
                (
                    2 if not spec.spec.use_mla else 1,
                    spec.num_blocks,
                    spec.block_size,
                    spec.spec.num_kv_heads,
                    spec.spec.head_size,
                ),
                dtype=spec.spec.dtype,
                device=self.offload_device if spec.spec.kv_offload else self.device,
                pin_memory=True if spec.spec.kv_offload else None,
            )
        return tensors