# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Wan Base Model - Unified implementation for all Wan-based models
This module provides the core Wan model implementation that should be used 
by both CausVid and Self Forcing variants.
"""

from .components import (
    sinusoidal_embedding_1d,
    rope_params, 
    rope_apply,
    WanRMSNorm,
    WanLayerNorm, 
    Head,
    MLPProj
)

from .model import (
    WanBaseModel,
    WanAttentionBlock,
    WAN_CROSSATTENTION_CLASSES,
    WanSelfAttention,
    WanT2VCrossAttention,
    WanI2VCrossAttention
)

from .text_encoder import (
    umt5_xxl,
    HuggingfaceTokenizer,
)

from .vae import (
    WanVAE_,
    _video_vae,
    CausalConv3d,
    RMS_norm,
    ResidualBlock,
    Encoder3d,
    Decoder3d,
    Resample,
    Upsample
)

from.distributed import (
    shard_model,
    usp_dit_forward, 
    usp_attn_forward
)

from.utils import (
    FlowDPMSolverMultistepScheduler, 
    get_sampling_sigmas,
    retrieve_timesteps,
    FlowUniPCMultistepScheduler,
    ParallelConfig
)

__all__ = [
    # Core components
    'sinusoidal_embedding_1d',
    'pad_freqs',
    'rope_params', 
    'rope_apply',
    'WanRMSNorm',
    'WanLayerNorm', 
    'Head',
    'MLPProj',
    
    # Base model
    'WanBaseModel',
    'WanAttentionBlock',
    'WAN_CROSSATTENTION_CLASSES',
    'WanSelfAttention',
    'WanT2VCrossAttention',
    'WanI2VCrossAttention',
    
    # Text encoder
    'umt5_xxl'
    'HuggingfaceTokenizer',
    
    # VAE
    'WanVAE_',
    '_video_vae',
    'CausalConv3d',
    'RMS_norm',
    'ResidualBlock',
    'Encoder3d',
    'Decoder3d',
    'Resample',
    'Upsample'

    # Distributed
    'shard_model',
    'usp_dit_forward',
    'usp_attn_forward',

    # utils
    'get_sampling_sigmas', 
    'retrieve_timesteps',
    'FlowDPMSolverMultistepScheduler', 
    'FlowUniPCMultistepScheduler',
    'ParallelConfig',
]