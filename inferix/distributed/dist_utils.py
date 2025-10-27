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

import os
from datetime import timedelta

import torch

from . import parallel_state as mpu
from .parallelism.pipeline_parallel import init_pp_scheduler

from . import parallel_state as mpu


def print_rank_0(message):
    """Print message only on rank 0 in distributed training."""
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                print(message)
        else:
            print(message)
    except Exception:
        print(message)


def print_per_rank(message):
    """Print message with rank prefix on all ranks in distributed training."""
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            print(f"[Rank {rank}] {message}")
        else:
            print(f"[Rank 0] {message}")
    except Exception:
        print(f"[Rank 0] {message}")


def dist_init(config):
    """Initialize torch.distributed and core model parallel."""

    assert torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():
        print_rank_0("Torch distribution already initialized, skipping initialization ...")
    else:
        rank = int(os.getenv("RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        # Manually set the device ids.
        if device_count > 0:
            device = rank % device_count
            torch.cuda.set_device(device)
        # Call the init process
        torch.distributed.init_process_group(
            backend=config.engine_config.distributed_backend,
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=config.engine_config.distributed_timeout_minutes),
        )
    assert config.engine_config.cp_size * config.engine_config.pp_size == torch.distributed.get_world_size()
    if device_count > 0:
        if mpu.model_parallel_is_initialized():
            print_rank_0("Model parallel is already initialized")
        else:
            mpu.initialize_model_parallel(
                cp_size=config.engine_config.cp_size,
                pp_size=config.engine_config.pp_size,
                nccl_communicator_config_path=None,
                distributed_timeout_minutes=config.engine_config.distributed_timeout_minutes,
                order="tp-cp-pp-dp",
            )
    if mpu.get_pp_world_size() > 1:
        init_pp_scheduler()
    print_rank_0("Initialize torch distribution and model parallel successfully")


def is_last_rank():
    return torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1)


def is_last_tp_cp_rank():
    return mpu.get_tp_rank(with_context_parallel=True) == mpu.get_tp_world_size(with_context_parallel=True) - 1


def get_world_size():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1
    return world_size


def get_device(local_rank=None):
    backend = torch.distributed.get_backend()
    if backend == "nccl":
        if local_rank is None:
            device = torch.device("cuda")
        else:
            device = torch.device(f"cuda:{local_rank}")
    elif backend == "gloo":
        device = torch.device("cpu")
    else:
        raise RuntimeError
    return device
