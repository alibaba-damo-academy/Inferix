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

"""
MAGI model specific configuration classes.

This module contains configuration classes specifically designed for the MAGI
multimodal video generation model.
"""

import dataclasses
import json
import os
from typing import Dict, Any, Optional

import torch

from inferix.core.config.model import ModelConfig, RuntimeConfig, EngineConfig
from inferix.profiling.config import ProfilingConfig


@dataclasses.dataclass
class MagiConfig:
    """Configuration class specific to MAGI model."""
    model_config: ModelConfig
    runtime_config: RuntimeConfig
    engine_config: EngineConfig
    profiling_config: Optional[ProfilingConfig] = None

    @classmethod
    def _check_missing_fields(cls, config_dict: dict, required_fields: list):
        actual_fields = set(config_dict.keys())
        missing_fields = set(required_fields) - actual_fields
        if missing_fields:
            raise ValueError(f"Missing fields in the configuration file: {', '.join(missing_fields)}")

    @classmethod
    def _create_nested_config(cls, config_dict: dict, config_name: str, config_cls):
        nested_config_dict = config_dict.get(config_name, {})
        cls._check_missing_fields(nested_config_dict, config_cls.__dataclass_fields__.keys())
        return config_cls(**nested_config_dict)

    @classmethod
    def _create_config_from_dict(cls, config_dict: dict):
        cls._check_missing_fields(config_dict, ['model_config', 'runtime_config', 'engine_config'])

        # Create nested configs
        model_config = cls._create_nested_config(config_dict, "model_config", ModelConfig)
        runtime_config = cls._create_nested_config(config_dict, "runtime_config", RuntimeConfig)
        engine_config = cls._create_nested_config(config_dict, "engine_config", EngineConfig)
        
        # Create optional profiling config
        profiling_config = None
        if "profiling_config" in config_dict:
            profiling_config_dict = config_dict.get("profiling_config", {})
            profiling_config = ProfilingConfig(**profiling_config_dict)

        return cls(model_config=model_config, runtime_config=runtime_config, 
                  engine_config=engine_config, profiling_config=profiling_config)

    @classmethod
    def from_json(cls, json_path: str):
        def simple_json_decoder(dct):
            dtype_map = {"torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16, "torch.float32": torch.float32}
            if 'params_dtype' in dct:
                dct['params_dtype'] = dtype_map[dct['params_dtype']]
            return dct

        with open(json_path, "r") as f:
            config_dict = json.load(f, object_hook=simple_json_decoder)
        magi_config = cls._create_config_from_dict(config_dict)

        def post_validation(magi_config):
            if magi_config.engine_config.fp8_quant or magi_config.engine_config.distill:
                assert (
                    magi_config.runtime_config.cfg_number == 1
                ), "Please set `cfg_number: 1` in config.json for distill or quant model"
            else:
                assert magi_config.runtime_config.cfg_number == 3, "Please set `cfg_number: 3` in config.json for base model"

        post_validation(magi_config)

        return magi_config

    def to_json(self, json_path: str):
        class SimpleJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, torch.dtype):
                    return str(obj)
                return super().default(obj)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        config_dict = {
            "model_config": dataclasses.asdict(self.model_config),
            "runtime_config": dataclasses.asdict(self.runtime_config),
            "engine_config": dataclasses.asdict(self.engine_config),
        }
        
        if self.profiling_config is not None:
            config_dict["profiling_config"] = dataclasses.asdict(self.profiling_config)
            
        with open(json_path, "w") as f:
            json.dump(config_dict, f, indent=4, cls=SimpleJSONEncoder)