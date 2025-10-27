from inferix.models.attention.backends import collect_supported_attn

class ParallelConfig:
    def __init__(self, ulysses_size=1, ring_size=1, local_rank=0, rank=0, world_size=1, 
                 ring_strategy="pass-kv", attn_backend=None):
        self.ulysses_size = ulysses_size
        self.ring_size = ring_size
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.ring_strategy = ring_strategy
        
        # Auto-detect available attention backend if not specified
        if attn_backend is None:
            supported_attn = collect_supported_attn()
            if "FlashAttnV3" in supported_attn:
                self.attn_backend = "FlashAttnV3"
            elif "FlashAttn" in supported_attn:
                self.attn_backend = "FlashAttn"
            elif "FlexAttention" in supported_attn:
                self.attn_backend = "FlexAttention"
            else:
                raise RuntimeError("No supported attention backend found. Please install at least one of: flash_attn, or ensure PyTorch FlexAttention is available.")
        else:
            # Validate that the specified backend is available
            supported_attn = collect_supported_attn()
            if attn_backend not in supported_attn:
                available_backends = list(supported_attn.keys())
                raise ValueError(f"Specified attention backend '{attn_backend}' is not available. Available backends: {available_backends}")
            self.attn_backend = attn_backend