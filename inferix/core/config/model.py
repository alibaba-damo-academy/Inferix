import dataclasses
import json
import os
import argparse
from typing import Dict, Any, Optional
from glob import glob
import importlib.util
import sys

from typing import Dict, Any, Optional
import dataclasses
import json
import os
import argparse
from glob import glob
import importlib.util
import sys

import torch


@dataclasses.dataclass
class ModelConfig:
    model_name: str

    # Transformer
    num_layers: Optional[int] = None  # Number of transformer layers.
    hidden_size: Optional[int] = None  # Transformer hidden size.
    ffn_hidden_size: Optional[int] = None  # Transformer Feed-Forward Network hidden size
    num_attention_heads: Optional[int] = None  # Number of transformer attention heads.
    num_query_groups: int = 1  # Number of query groups, which used for GQA
    kv_channels: Optional[int] = None  # Projection weights dimension in multi-head attention
    layernorm_epsilon: float = 1e-6  # Epsilon for layer norm and RMS norm.
    apply_layernorm_1p: bool = False  # Adjust LayerNorm weights which improves numerical stability.
    x_rescale_factor: float = 1.0
    half_channel_vae: bool = False
    params_dtype: torch.dtype = None

    # Embedding
    patch_size: int = 2  # (latent) patch size for DiT patch embedding layer
    t_patch_size: int = 1  # (latent) patch size for t dim patch embedding layer
    in_channels: int = 4  # latent input channel for DiT
    out_channels: int = 4  # latent output channel for DiT
    cond_hidden_ratio: float = 0.25
    caption_channels: int = 4096
    caption_max_length: int = 800
    xattn_cond_hidden_ratio: float = 1.0
    cond_gating_ratio: float = 1.0
    gated_linear_unit: bool = False


@dataclasses.dataclass
class RuntimeConfig:
    # Inference settings such as cfg, kv range, clean t, etc.
    cfg_number: Optional[int] = None  # Number of CFG
    cfg_t_range: list = dataclasses.field(
        default_factory=lambda: [0, 0.0217, 0.1000, 0.3, 0.999]
    )  # CFG t-range of each scales
    prev_chunk_scales: list = dataclasses.field(
        default_factory=lambda: [1.5, 1.5, 1.5, 1.5, 1.5]
    )  # CFG scales of previous chunks
    text_scales: list = dataclasses.field(default_factory=lambda: [7.5, 7.5, 7.5, 7.5, 7.5])  # CFG scales of text

    noise2clean_kvrange: list = dataclasses.field(default_factory=list)  # Range of kv for noise2clean chunks
    clean_chunk_kvrange: int = -1  # Range of kv for clean chunks
    clean_t: float = 1.0  # timestep for clean chunks

    # Video settings
    seed: int = 1234  # Random seed used for python, numpy, pytorch, and cuda.
    num_frames: int = 128
    video_size_h: Optional[int] = None
    video_size_w: Optional[int] = None
    num_steps: int = 64  # Number of steps for the diffusion model
    window_size: int = 4  # Window size for the diffusion model
    fps: int = 24  # Frames per second
    chunk_width: int = 6  # Clip width for the diffusion model

    # Checkpoint, includes t5, vae, dit, etc.
    t5_pretrained: Optional[str] = None  # Path to load pretrained T5 model.
    t5_device: str = "cuda"  # Device for T5 model to run on.
    vae_pretrained: Optional[str] = None  # Path to load pretrained VAE model.
    scale_factor: float = 0.18215  # Scale factor for the vae
    temporal_downsample_factor: int = 4  # Temporal downsample factor for the vae
    load: Optional[str] = None  # Directory containing a model checkpoint.


@dataclasses.dataclass
class EngineConfig:
    # Parallism strategy
    distributed_backend: str = "nccl"  # Choices: ["nccl", "gloo"]
    distributed_timeout_minutes: int = 10  # Timeout minutes for torch.distributed.
    pp_size: int = 1  # Degree of pipeline model parallelism.
    cp_size: int = 1  # Degree of context parallelism.
    cp_strategy: str = "none"  # Choices: ["none", "cp_ulysses", "cp_shuffle_overlap"]
    ulysses_overlap_degree: int = 1  # Overlap degree for Ulysses

    # Quantization
    fp8_quant: bool = False  # Enable 8-bit floating point quantization for model weights.

    # Distillation
    distill_nearly_clean_chunk_threshold: float = 0.3  # Threshold for distilling nearly clean chunks
    shortcut_mode: str = "8,16,16"  # Parameters for shortcut mode
    distill: bool = False  # Use distill mode

    # Optimization
    kv_offload: bool = False  # Use kv-offload algorithm
    enable_cuda_graph: bool = False  # Enable CUDA graph for video generation
