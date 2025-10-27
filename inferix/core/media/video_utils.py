import torch
import math
import torchvision.transforms as transforms
from . import video_transforms
import numpy as np
import imageio


NUM_FRAMES_MAP = {
    "1x": 51,
    "2x": 102,
    "4x": 204,
    "8x": 408,
    "16x": 816,
    "2s": 51,
    "4s": 102,
    "8s": 204,
    "16s": 408,
    "32s": 816,
}

def get_num_frames(num_frames):
    if num_frames in NUM_FRAMES_MAP:
        return NUM_FRAMES_MAP[num_frames]
    else:
        return int(num_frames)


def save_sample(x, save_path=None, fps=8, normalize=True, value_range=(-1, 1), force_video=False, verbose=True):
    """
    Args:
        x (Tensor): shape [C, T, H, W]
    """
    from torchvision.io import write_video
    from torchvision.utils import save_image
    assert x.ndim == 4

    if not force_video and x.shape[1] == 1:  # T = 1: save as image
        save_path += ".png"
        x = x.squeeze(1)
        save_image([x], save_path, normalize=normalize, value_range=value_range)
    else:
        save_path += ".mp4"
        if normalize:
            low, high = value_range
            x = x.clamp(min=low, max=high)
            x = x.sub(low).div(max(high - low, 1e-5))

        x = x.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 3, 0).to("cpu", torch.uint8)
        write_video(save_path, x, fps=fps, video_codec="h264")
    if verbose:
        print(f"Saved to {save_path}")
    return save_path

def get_transforms_video(name="center", image_size=(256, 256)):
    if name is None:
        return None
    elif name == "center":
        assert image_size[0] == image_size[1], "image_size must be square for center crop"
        transform_video = transforms.Compose(
            [
                video_transforms.ToTensorVideo(),  # TCHW
                # video_transforms.RandomHorizontalFlipVideo(),
                video_transforms.UCFCenterCropVideo(image_size[0]),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    elif name == "resize_crop":
        transform_video = transforms.Compose(
            [
                video_transforms.ToTensorVideo(),  # TCHW
                video_transforms.ResizeCrop(image_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    elif name == "resize_crop_mochi":
        transform_video = transforms.Compose(
            [
                video_transforms.ResizeCrop(image_size),
                video_transforms.ToTensorVideo(),  # TCHW
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    else:
        raise NotImplementedError(f"Transform {name} not implemented")
    return transform_video

def read_video(video_path, new_fps, device):
    reader = imageio.get_reader(video_path)
    original_fps = reader.get_meta_data()["fps"]
    num_original_frames = int(original_fps * reader.get_meta_data()["duration"])
    duration = reader.get_meta_data()["duration"]
    num_new_frames = int(duration * new_fps)
    step = num_original_frames / num_new_frames
    frames = []  # Store the loaded frames
    for i in range(num_new_frames):
        frame_index = int(i * step)
        # Clamp frame_index within valid bounds
        frame_index = min(frame_index, num_original_frames - 1)
        frame = reader.get_data(frame_index)
        frames.append(frame)
    # Convert frames to a numpy array
    video_array = torch.from_numpy(np.array(frames).transpose((0, 3, 1, 2))).to(
        device
    )

    # == Prepare video resolution ==
    _, _, video_H, video_W = video_array.size()
    video_H_prime = (
        int(video_H / math.sqrt(video_H * video_W / 480 / 848) / 16) * 16
    )
    video_W_prime = (
        int(video_W / math.sqrt(video_H * video_W / 480 / 848) / 16) * 16
    )
    image_size = [video_H_prime, video_W_prime]

    transform = get_transforms_video("resize_crop", image_size)
    original_video = (
        transform(video_array).unsqueeze(0).permute(0, 2, 1, 3, 4)
    )  # Shape: 1 C T H W
    return original_video

def to_torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        dtype_mapping = {
            "float64": torch.float64,
            "float32": torch.float32,
            "float16": torch.float16,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "half": torch.float16,
            "bf16": torch.bfloat16,
        }
        if dtype not in dtype_mapping:
            raise ValueError(f"Unsupported dtype string: {dtype}")
        dtype = dtype_mapping[dtype]
        return dtype
    else:
        raise ValueError(f"Unsupported dtype type: {type(dtype)}")

def encode_prompt(
        prompt,
        neg_prompt,
        text_encoder,
        max_seq_len):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    neg_prompt = [neg_prompt] if isinstance(neg_prompt, str) else neg_prompt

    context = text_encoder(prompt)
    context_null = text_encoder(neg_prompt)

    return dict(context=context, context_null=context_null, max_seq_len=max_seq_len)