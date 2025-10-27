from ..attention import attention, CoreAttention
from inferix.models.wan_base import (
    WanRMSNorm,
    WanLayerNorm,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d,
    WAN_CROSSATTENTION_CLASSES
)
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
from typing import Optional
from inferix.models.wan_base import ParallelConfig
from inferix.kvcache_manager.kvcache_manager import KVCacheManager, KVCacheRequest
from inferix.kvcache_manager.model.causvid_kv_cache_manager import KVCacheManagerFactory
from inferix.models.wan_base.components import pad_freqs
from functools import partial

def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame +
                     f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)

def causal_rope_apply_chunked(x, grid_sizes, freqs, world_size, rank, start_frame=0):
    s, n, c = x.size(1), x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        freqs_i = pad_freqs(freqs_i, s * world_size) # s = 585
        s_per_rank = s
        freqs_i_rank = freqs_i[(rank * s_per_rank):((rank + 1) *
                                                        s_per_rank), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 parallel_config: ParallelConfig | None = None):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()


        self.parallel_config = parallel_config
        if self.parallel_config is not None and self.parallel_config.world_size > 1:
            self.parallel_attention = CoreAttention(strategy=self.parallel_config.ring_strategy)
            if self.parallel_config.attn_backend == "MagiAttention":
                self.attention = partial(self.parallel_attention,
                attn_backend=self.parallel_config.attn_backend,
                attn_type_map=torch.tensor([1], device=torch.device("cuda"), dtype=torch.int32))
            else:
                self.attention = partial(self.parallel_attention, attn_backend=self.parallel_config.attn_backend)
        else:
            # For single GPU or when parallel_config is None, use the traditional attention
            self.attention = attention

    def forward(self, x, seq_lens, grid_sizes, freqs, block_mask, kv_start, kv_end, current_start, current_end, k_cache, v_cache):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if self.parallel_config is not None and self.parallel_config.world_size > 1:
            roped_query = causal_rope_apply_chunked(
                q, grid_sizes, freqs, world_size=self.parallel_config.world_size, rank=self.parallel_config.rank, start_frame=current_start // math.prod(grid_sizes[0][1:]).item()).type_as(v)
            roped_key = causal_rope_apply_chunked(
                k, grid_sizes, freqs, world_size=self.parallel_config.world_size, rank=self.parallel_config.rank, start_frame=current_start // math.prod(grid_sizes[0][1:]).item()).type_as(v)

            x = self.attention(
                query=roped_query,
                key=roped_key,
                value=v,
                k_cache=k_cache,
                v_cache=v_cache,
                k_cache_offset=kv_start,
                v_cache_offset=kv_start
            )
        else:
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start // math.prod(grid_sizes[0][1:]).item()).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start // math.prod(grid_sizes[0][1:]).item()).type_as(v)
            
            k_cache[:, kv_start:kv_end, ...] = roped_key
            v_cache[:, kv_start:kv_end, ...] = v
            x = self.attention(
                q=roped_query,
                k=k_cache,
                v=v_cache
            )
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x, k_cache[:, kv_start:kv_end, ...], v_cache[:, kv_start:kv_end, ...]


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 layer_idx,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 enable_kv_offload=True,
                 parallel_config: ParallelConfig | None = None):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.enable_kv_offload = enable_kv_offload
        self.parallel_config = parallel_config

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, window_size, qk_norm,
                                                eps, parallel_config)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.kv_cache_manager = KVCacheManagerFactory.create_manager(layer_idx, self.num_heads, dim // num_heads, enable_kv_offload=enable_kv_offload)
        self.is_cross_attn_init = False

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_start,
        kv_end,
        current_start,
        current_end,
        kv_cache_manager,
        kv_cache_requests,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]

        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # Get KV Cache for self-attention
        all_k_cache = []
        all_v_cache = []
        for kv_cache_request in kv_cache_requests:
            kv_cache = self.kv_cache_manager.get_kv_cache(kv_cache_manager=kv_cache_manager, kv_cache_request=kv_cache_request, start_index=0, length=kv_end)
            all_k_cache.append(kv_cache[0])
            all_v_cache.append(kv_cache[1])
        k_cache = torch.stack(all_k_cache, dim=0)
        v_cache = torch.stack(all_v_cache, dim=0)
        # self-attention
        y, new_k, new_v = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
             * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_start, kv_end, current_start, current_end, k_cache, v_cache)

        # Set KV Cache for self-attention
        for request_idx, kv_cache_request in enumerate(kv_cache_requests):
            cur_k_cache, cur_v_cache = new_k[request_idx], new_v[request_idx]
            self.kv_cache_manager.set_kv_cache(kv_cache_manager=kv_cache_manager, kv_cache_request=kv_cache_request, start_index=kv_start, k_data=cur_k_cache, v_data=cur_v_cache)

        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen))
                 * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache):

            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )

            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x
        
        # Get KV Cache for cross-attention
        all_k_cache = []
        all_v_cache = []
        for kv_cache_request in kv_cache_requests:
            kv_cache = self.kv_cache_manager.get_crossattn_cache(kv_cache_manager=kv_cache_manager, kv_cache_request=kv_cache_request)
            all_k_cache.append(kv_cache[0])
            all_v_cache.append(kv_cache[1])
        k_cache = torch.stack(all_k_cache, dim=0)
        v_cache = torch.stack(all_v_cache, dim=0)
        crossattn_cache = {}
        crossattn_cache["k"] = k_cache
        crossattn_cache["v"] = v_cache
        crossattn_cache["is_init"] = self.is_cross_attn_init

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)

        # Set KV Cache for cross-attention
        if not self.is_cross_attn_init:
            for request_idx, kv_cache_request in enumerate(kv_cache_requests):
                cur_k_cache, cur_v_cache = crossattn_cache["k"][request_idx], crossattn_cache["v"][request_idx]
                self.kv_cache_manager.set_crossattn_cache(kv_cache_manager=kv_cache_manager, kv_cache_request=kv_cache_request, k_data=cur_k_cache, v_data=cur_v_cache)
            self.is_cross_attn_init = True

        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(
            self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) *
            (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 enable_kv_offload=True,
                 parallel_config: ParallelConfig = None):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.parallel_config = parallel_config

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads, layer_idx,
                                    window_size, qk_norm, cross_attn_norm, eps, enable_kv_offload, parallel_config)
            for layer_idx in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_start: int =  0,
        kv_end: int = 0,
        current_start: int = 0,
        current_end: int = 0,
        kv_cache_manager: Optional[KVCacheManager] = None,
        kv_cache_requests: Optional[list] = None,
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask
        )

         # scatter x
        if self.parallel_config is not None and self.parallel_config.world_size > 1:
            x = torch.chunk(x, self.parallel_config.world_size, dim=1)[self.parallel_config.rank]

        for block in self.blocks:
            kwargs.update(
                {
                    "kv_start": kv_start,
                    "kv_end": kv_end,
                    "current_start": current_start,
                    "current_end": current_end,
                    "kv_cache_manager": kv_cache_manager,
                    "kv_cache_requests": kv_cache_requests
                }
            )
            x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # gather x
        if self.parallel_config is not None and self.parallel_config.world_size > 1:
            from xfuser.core.distributed import get_sp_group
            x = get_sp_group().all_gather(x, dim=1)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)