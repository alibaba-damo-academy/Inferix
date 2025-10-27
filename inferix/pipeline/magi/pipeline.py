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
from typing import Optional, Dict, Any

from inferix.models.magi.config import MagiConfig
from inferix.core.utils import set_random_seed
from inferix.distributed.dist_utils import dist_init, print_rank_0
from inferix.models.magi.dit import get_dit
from inferix.pipeline.base_pipeline import AbstractInferencePipeline

from .prompt_process import get_txt_embeddings
from .video_generate import generate_per_chunk
from .video_process import post_chunk_process, process_image, process_prefix_video, save_video_to_disk


class MagiPipeline(AbstractInferencePipeline):
    def __init__(self, config_path):
        self.config = MagiConfig.from_json(config_path)
        set_random_seed(self.config.runtime_config.seed)
        dist_init(self.config)
        print_rank_0(self.config)
        
        self.magi_config = self.config
        # Call parent class initialization, convert config to dictionary
        super().__init__(vars(self.config))

    def _initialize_pipeline(self):
        """[Implementation of abstract method] Initialize Magi pipeline"""
        # MagiPipeline initialization logic has been completed in __init__
        # Additional initialization logic can be added here (if needed)
        pass

    def load_checkpoint(self, checkpoint_path: str, **kwargs) -> None:
        """[Implementation of abstract method] Load model checkpoint weights (MagiPipeline has loaded the model during initialization)"""
        # MagiPipeline has loaded the model during initialization, no additional operation is needed here
        pass

    def run_text_to_video(self, prompt: str, output_path: Optional[str] = None, **kwargs):
        """[Implementation of abstract method] Run text-to-video generation"""
        if output_path is None:
            output_path = "./output.mp4"
        self._run(prompt, None, output_path)
        return output_path

    def run_image_to_video(self, prompt: str, image_path: str, output_path: Optional[str] = None, **kwargs):
        """[Implementation of abstract method] Run image-to-video generation"""
        if output_path is None:
            output_path = "./output.mp4"
        prefix_video = process_image(image_path, self.magi_config)
        self._run(prompt, prefix_video, output_path)
        return output_path

    def run_video_to_video(self, prompt: str, prefix_video_path: str, output_path: Optional[str] = None, **kwargs):
        """Run video-to-video generation"""
        if output_path is None:
            output_path = "./output.mp4"
        prefix_video = process_prefix_video(prefix_video_path, self.magi_config)
        self._run(prompt, prefix_video, output_path)
        return output_path

    def _run(self, prompt: str, prefix_video: torch.Tensor, output_path: str):
        caption_embs, emb_masks = get_txt_embeddings(prompt, self.magi_config)
        dit = get_dit(self.magi_config)
        videos = torch.cat(
            [
                post_chunk_process(chunk, self.magi_config)
                for chunk in generate_per_chunk(
                    model=dit, prefix_video=prefix_video, caption_embs=caption_embs, emb_masks=emb_masks
                )
            ],
            dim=0,
        )
        save_video_to_disk(videos, output_path, fps=self.magi_config.runtime_config.fps)

        mem_allocated_gb = torch.cuda.max_memory_allocated() / 1024**3
        mem_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
        print_rank_0(
            f"Finish MagiPipeline, max memory allocated: {mem_allocated_gb:.2f} GB, max memory reserved: {mem_reserved_gb:.2f} GB"
        )