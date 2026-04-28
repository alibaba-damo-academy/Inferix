import torch
import torchvision
import os
import glob
import traceback
import json
import argparse

from typing import Union

import numpy as np
import cv2

from vbench.utils import load_video
from metrics import create_metric_func

def vde(
    video_chunks: Union[torch.Tensor, np.ndarray],
    metric_function: callable,
    weight_type: str = 'linear'
) -> torch.Tensor:
    """Compute the Video Drift Error (VDE)."""
    if not hasattr(video_chunks, 'device'):
        device = "cuda"
    else:
        device = video_chunks.device
    N = video_chunks.shape[0]
    if N < 2: 
        return torch.tensor(0.0, device=device, dtype=torch.float32)
    try:
        metrics = torch.tensor([metric_function(chunk) for chunk in video_chunks], device=device, dtype=torch.float32)
        print(f"Metrics of All Chunks: {metrics}")
    except Exception as e:
        print(f"Error while calling metric_function: {e}"); raise
    m1 = metrics[0]
    if m1 == 0: 
        return torch.tensor(torch.inf, device=device)
    drifting_values = torch.abs(metrics[1:] - m1) / m1
    i_indices = torch.arange(2, N + 1, dtype=torch.float32, device=device)
    if weight_type == 'linear': 
        weights = N - i_indices + 1
    elif weight_type == 'log': 
        weights = torch.log(N - i_indices + 1)
    else: 
        raise ValueError("Unsupported weight type")
    return torch.sum(weights * drifting_values)

def evaluate_video(video_path: str, n_chunks: int, metric_func: callable):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    """Load, split, and evaluate the VDE score for a single video file using the specified metric_func."""
    try:
        print(f"\n{'='*50}\n[+] Processing video: {os.path.basename(video_path)}\n{'='*50}")
        # Different metrics may rely on distinct video loading modes
        if metric_func.__name__ == 'clarity':
            # video_frames, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')
            # if video_frames.shape[0] < n_chunks:
            #     print(f"[!] Warning: total frame count ({video_frames.shape[0]}) is too small to split into {n_chunks} chunks. Skipping."); return
            # print(f"    - Splitting into {n_chunks} chunks...")
            # video_tensor = video_frames.permute(3, 0, 1, 2).to(torch.float32) / 255.0
            # total_frames = video_tensor.shape[1]
            # frames_per_chunk = total_frames // n_chunks
            # trimmed_len = frames_per_chunk * n_chunks
            # print(f"    - Original frame count: {total_frames}, trimming to {trimmed_len} frames to ensure an even split.")
            # trimmed_video = video_tensor[:, :trimmed_len, :, :]
            # chunks_list = torch.chunk(trimmed_video, chunks=n_chunks, dim=1)
            # video_chunks_tensor = torch.stack(chunks_list)
            video_tensor = load_video(video_path).to(device)
            if video_tensor.shape[0] < n_chunks:
                print(f"[!] Warning: total frame count ({video_tensor.shape[0]}) is too small to split into {n_chunks} chunks. Skipping."); return
            print(f"    - Splitting into {n_chunks} chunks...")
            total_frames = video_tensor.shape[0]
            frames_per_chunk = total_frames // n_chunks
            trimmed_len = frames_per_chunk * n_chunks
            print(f"    - Original frame count: {total_frames}, trimming to {trimmed_len} frames to ensure an even split.")   
            trimmed_video = video_tensor[:trimmed_len, : , :, :]
            chunks_list = torch.chunk(trimmed_video, chunks=n_chunks, dim=0)
            video_chunks_tensor = torch.stack(chunks_list)
        
        elif metric_func.__name__ == 'motion':
            from vbench.motion_smoothness import FrameProcess
            fp = FrameProcess()
            video_frames = fp.get_frames(video_path)

            # chunking for video lists
            if len(video_frames) < n_chunks:
                print(f"[!] Warning: total frame count ({len(video_frames)}) is too small to split into {n_chunks} chunks. Skipping."); return
            print(f"    - Splitting into {n_chunks} chunks...")
            frames_per_chunk = len(video_frames) // n_chunks
            trimmed_len = frames_per_chunk * n_chunks
            print(f"    - Original frame count: {len(video_frames)}, trimming to {trimmed_len} frames to ensure an even split.")
            trimmed_video = video_frames[:trimmed_len]
            chunks_list = [trimmed_video[i:i + frames_per_chunk] for i in range(0, trimmed_len, frames_per_chunk)]
            video_chunks_tensor: np.ndarray = np.stack([np.stack(chunk) for chunk in chunks_list])
        elif metric_func.__name__ == 'aesthetic':
            video_tensor = load_video(video_path).to(device) # [N, C, H, W]
            print(video_tensor.shape)
            if video_tensor.shape[0] < n_chunks:
                print(f"[!] Warning: total frame count ({video_tensor.shape[0]}) is too small to split into {n_chunks} chunks. Skipping."); return
            print(f"    - Splitting into {n_chunks} chunks...")
            total_frames = video_tensor.shape[0]
            frames_per_chunk = total_frames // n_chunks
            trimmed_len = frames_per_chunk * n_chunks
            print(f"    - Original frame count: {total_frames}, trimming to {trimmed_len} frames to ensure an even split.")
            trimmed_video = video_tensor[:trimmed_len, : , :, :]
            chunks_list = torch.chunk(trimmed_video, chunks=n_chunks, dim=0)
            video_chunks_tensor = torch.stack(chunks_list)
        elif metric_func.__name__ == 'dynamic':
            device = "cuda"
            def extract_frame(frame_list, interval=1):
                extract = []
                for i in range(0, len(frame_list), interval):
                    extract.append(frame_list[i])
                return extract

            def get_frames(video_path):
                frame_list = []
                video = cv2.VideoCapture(video_path)
                fps = video.get(cv2.CAP_PROP_FPS) # get fps
                interval = max(1, round(fps / 8))
                while video.isOpened():
                    success, frame = video.read()
                    if success:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to rgb
                        frame = torch.from_numpy(frame.astype(np.uint8)).permute(2, 0, 1).float()
                        frame = frame[None].to(device)
                        frame_list.append(frame)
                    else:
                        break
                video.release()
                assert frame_list != []
                frame_list = extract_frame(frame_list, interval)
                return frame_list 

            video_frames = get_frames(video_path)
            print(len(video_frames))
            print(video_frames[0].shape)
            
            if len(video_frames) < n_chunks:
                print(f"[!] Warning: total frame count ({len(video_frames)}) is too small to split into {n_chunks} chunks. Skipping."); return
            print(f"    - Splitting into {n_chunks} chunks...")
            frames_per_chunk = len(video_frames) // n_chunks
            trimmed_len = frames_per_chunk * n_chunks
            print(f"    - Original frame count: {len(video_frames)}, trimming to {trimmed_len} frames to ensure an even split.")
            trimmed_video = video_frames[:trimmed_len]
            # convert to torch.Tensor
            trimmed_video = torch.stack(trimmed_video).to(device)
            print(trimmed_video.shape)
            chunks_list = torch.chunk(trimmed_video, chunks=n_chunks, dim=0)
            video_chunks_tensor = torch.stack(chunks_list)
        
        elif metric_name == 'subject':
            video_tensor = load_video(video_path).to(device)  # [N, C, H, W]
            if video_tensor.shape[0] < n_chunks:
                print(f"[!] Warning: total frame count ({video_tensor.shape[0]}) is too small to split into {n_chunks} chunks. Skipping."); return
            print(f"    - Splitting into {n_chunks} chunks...")
            total_frames = video_tensor.shape[0]
            frames_per_chunk = total_frames // n_chunks
            trimmed_len = frames_per_chunk * n_chunks
            print(f"    - Original frame count: {total_frames}, trimming to {trimmed_len} frames to ensure an even split.")
            trimmed_video = video_tensor[:trimmed_len, : , :, :]
            chunks_list = torch.chunk(trimmed_video, chunks=n_chunks, dim=0)
            video_chunks_tensor = torch.stack(chunks_list)
        
        elif metric_name == 'background':
            video_tensor = load_video(video_path)
            if video_tensor.shape[0] < n_chunks:
                print(f"[!] Warning: total frame count ({video_tensor.shape[0]}) is too small to split into {n_chunks} chunks. Skipping."); return
            print(f"    - Splitting into {n_chunks} chunks...")
            total_frames = video_tensor.shape[0]
            frames_per_chunk = total_frames // n_chunks
            trimmed_len = frames_per_chunk * n_chunks
            print(f"    - Original frame count: {total_frames}, trimming to {trimmed_len} frames to ensure an even split.")
            trimmed_video = video_tensor[:trimmed_len, : , :, :]
            chunks_list = torch.chunk(trimmed_video, chunks=n_chunks, dim=0)
            video_chunks_tensor = torch.stack(chunks_list)

        else:
            raise NotImplementedError(f"Unsupported metric_func: {metric_func.__name__}")
        print(f"    - Splitting complete. Each chunk has {video_chunks_tensor.shape[1]} frames.")
        vde_score = vde(video_chunks_tensor, metric_func)
        print(f"\n    >>> Final VDE score: {vde_score.item():.4f} <<<\n")
        return vde_score.item()
    except Exception as e:
        traceback.print_exc()
        print(f"[!] Critical error while processing video {os.path.basename(video_path)}: {e}")

# ==============================================================================
# Main program: simplified logic
# ==============================================================================

SUPPORTED_METRICS = ['clarity', 'motion', 'aesthetic', 'dynamic', 'subject', 'background']
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VDE Video Evaluation Tool')
    parser.add_argument('--video_dir', type=str, required=True,
                       help='Input video directory path', default="your/video/input")
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output results directory path', default="your/results/output")
    args = parser.parse_args()
    # --- Parameter definitions: switch metrics by editing strings ---
    VIDEO_DIR = args.video_dir
    N_CHUNKS = 10
    OUTPUT_DIR = args.output_dir
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    #  <<<<< Easily switch to the metric name you want here >>>>>
    # TODO: integrate VBench metrics!
    # METRIC_NAME = 'brightness'

    for metric_name in SUPPORTED_METRICS:

        kwargs = {}
        if metric_name == "clarity":
            kwargs['num_frames_to_sample'] = 5
        try:
            # --- Step 1: call factory function to get the configured metric ---
            # All heavy model loading, caching, and parameter binding logic lives in create_metric_func
            selected_metric_func = create_metric_func(
                metric_name=metric_name,
                **kwargs  # pass through extra parameters
            )

            # --- Step 2: locate and evaluate videos ---
            if not os.path.isdir(VIDEO_DIR):
                print(f"[!] Error: directory not found '{VIDEO_DIR}'")
            else:
                output_file = os.path.join(OUTPUT_DIR, f"vde_{metric_name}.json")
                search_pattern = os.path.join(VIDEO_DIR, '*.mp4')
                video_files = glob.glob(search_pattern)
                scores = {}

                if not video_files:
                    print(f"[!] No .mp4 files found in directory '{VIDEO_DIR}'.")
                else:
                    print(f"\n[*] Starting evaluation with metric '{selected_metric_func.__name__}'...")
                    for video_path in video_files:
                        vde_score = evaluate_video(video_path, N_CHUNKS, selected_metric_func)
                        scores[str(video_path)] = format(vde_score, '.4f')

                    with open(output_file, 'w') as f:
                        json.dump(scores, f, indent=4)
                    print(f"\n{'='*50}\nAll video evaluations completed.\n{'='*50}")

        except Exception as e:
            print(f"An error occurred while running the program: {e}")
