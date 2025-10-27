import math

import torch
try:
    from flash_attn_interface import _flash_attn_forward as flash_attn_forward_v3
    HAS_FLASH_ATTN_HOPPER = True
except ImportError:
    HAS_FLASH_ATTN_HOPPER = False

try:
    from flash_attn.flash_attn_interface import _flash_attn_forward as flash_attn_forward_v2
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

try:
    from flashinfer.prefill import single_prefill_with_kv_cache
    _LOG2_E = math.log2(math.e)
    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False

try:
    from magi_attention.api import flex_flash_attn_func
    HAS_MAGI_ATTENTION = True
except ImportError:
    HAS_MAGI_ATTENTION = False

try:
    from torch.nn.attention.flex_attention import flex_attention as torch_flex_attention
    HAS_TORCH_FLEX_ATTENTION = True
except ImportError:
    HAS_TORCH_FLEX_ATTENTION = False


def flash_attn3_func_forward(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), q_descale=None, k_descale=None, v_descale=None, softcap=0.0):
    out, softmax_lse, *rest = flash_attn_forward_v3(
        q,
        k,
        v,
        None, None,  # k_new, v_new
        None,  # qv
        None,  # out
        None, None, None,   # cu_seqlens_q/k/k_new
        None, None,   # seqused_q/k
        None, None,   # max_seqlen_q/k
        None, None, None,   # page_table, kv_batch_idx, leftpad_k,
        None, None, None,  # rotary_cos/sin, seqlens_rotary
        q_descale, k_descale, v_descale,
        softmax_scale,
        causal=causal,
        window_size=window_size,
        attention_chunk=0,
        softcap=softcap,
    )
    return out, softmax_lse

def flash_attn_forward(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, return_softmax=False):
    block_out, block_lse, _, _ = flash_attn_forward_v2(
        q,
        k,
        v,
        dropout_p = dropout_p,
        softmax_scale = softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        return_softmax=return_softmax,
    )
    return block_out, block_lse

def flashinfer_attn_forward(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, custom_mask=None):
    if q.ndim == 4:
        if q.shape[0] >1:
            raise ValueError("batch size > 1 is not supported")
        out, lse = single_prefill_with_kv_cache(
            q[0],
            k[0],
            v[0],
            sm_scale=softmax_scale,
            causal=causal,
            logits_soft_cap=softcap,
            window_left=window_size[0],
            return_lse=True,
            custom_mask=custom_mask,
        )
        lse = lse.transpose(0, 1)
    elif q.ndim == 3:
        out, lse = single_prefill_with_kv_cache(
            q,
            k,
            v,
            sm_scale=softmax_scale,
            causal=causal,
            logits_soft_cap=softcap,
            window_left=window_size[0],
            return_lse=True,
            custom_mask=custom_mask,
        )
        lse = lse.transpose(0, 1)
    else:
        raise ValueError(f"Invalid input shape: {q.shape}")
    lse = lse / _LOG2_E
    out, lse = out.unsqueeze(0),lse.unsqueeze(0)
    return out, lse

def flex_attention_forward(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, custom_mask=None):
    q_flex = q.transpose(1, 2)
    k_flex = k.transpose(1, 2)
    v_flex = v.transpose(1, 2)
    out, lse = torch_flex_attention(
        query=q_flex,
        key=k_flex,
        value=v_flex,
        score_mod=None,
        block_mask=custom_mask,
        scale=None,
        enable_gqa=False,
        return_lse=True,
        kernel_options=None,
        )   
    return out.transpose(1, 2), lse

def magi_attention_forward(q, k, v, q_ranges, k_ranges, max_seqlen_q, max_seqlen_k, attn_type_map=None, softmax_scale=None, softcap=0.0, deterministic=False, sm_margin=0, disable_fwd_atomic_reduction=False, auto_range_merge=False):
    if q.shape[0] >1:
        raise ValueError("batch size > 1 is not supported")
    
    if q_ranges is None and attn_type_map is not None:
        q_ranges = torch.tensor([[0, q[0].shape[0]]], device=torch.device("cuda"), dtype=torch.int32)
        k_ranges = torch.tensor([[0, k[0].shape[0]]], device=torch.device("cuda"), dtype=torch.int32)
    out, lse = flex_flash_attn_func(
        q[0], 
        k[0], 
        v[0], 
        q_ranges=q_ranges, 
        k_ranges=k_ranges, 
        max_seqlen_q=max_seqlen_q, 
        max_seqlen_k=max_seqlen_k, 
        attn_type_map=attn_type_map,
        # softmax_scale=softmax_scale,
        # softcap=softcap,
        # deterministic=deterministic,
        # sm_margin=sm_margin,
        # disable_fwd_atomic_reduction=disable_fwd_atomic_reduction,
        # auto_range_merge=auto_range_merge,
    )

    out, lse = out.unsqueeze(0),lse.unsqueeze(0)
    return out, lse


def collect_supported_attn():
    supported_attn = {}
    if HAS_FLASH_ATTN_HOPPER:
        supported_attn["FlashAttnV3"] = flash_attn3_func_forward
    if HAS_FLASH_ATTN:
        supported_attn["FlashAttn"] = flash_attn_forward
    if HAS_FLASHINFER:
        supported_attn["FlashInfer"] = flashinfer_attn_forward
    if HAS_MAGI_ATTENTION:
        supported_attn["MagiAttention"] = magi_attention_forward
    # if HAS_TORCH_FLEX_ATTENTION:
    #     supported_attn["FlexAttention"] = flex_attention_forward
    return supported_attn