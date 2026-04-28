import os
import torch
import pyiqa
import clip
import torch.nn.functional as F

from vbench.motion_smoothness import MotionSmoothness

# ==============================================================================
# 1. 定义具体的度量计算逻辑
# ==============================================================================

def simple_brightness_metric(video_chunk: torch.Tensor) -> float:
    """一个简单的示例度量函数，计算视频片段的平均亮度。"""
    return video_chunk.mean().item()

# Clarity
def calculate_iqa_score(
    video_chunk: torch.Tensor,
    iqa_model,
    num_frames_to_sample: int = 5
) -> float:
    """
    使用预加载的 pyiqa 模型计算单个视频片段的质量分数。
    """
    video_chunk_t_first = video_chunk.permute(1, 0, 2, 3)
    # TODO: input all frames increases overhead, whether to use frame sampling?
    # total_frames = video_chunk_t_first.shape[0]
    # indices = torch.linspace(0, total_frames - 1, num_frames_to_sample, dtype=torch.long)
    # sampled_frames_batch = video_chunk_t_first[indices]

    # 直接在整个批次上调用 pyiqa 模型进行推理
    with torch.no_grad():
        scores = iqa_model(video_chunk_t_first)

    return scores.mean().item()

# Imaging Quality (Clarity)
def compute_imaging_quality(
    video_chunk: torch.Tensor,
    model,
    **kwargs,
):
    from vbench.imaging_quality import transform
    if 'imaging_quality_preprocessing_mode' not in kwargs:
        preprocess_mode = 'longer'
    else:
        preprocess_mode = kwargs['imaging_quality_preprocessing_mode']
    device = "cuda"
    images = transform(video_chunk, preprocess_mode)
    acc_score_video = 0.
    for i in range(len(images)):
        frame = images[i].unsqueeze(0).to(device)
        score = model(frame)
        acc_score_video += float(score)
    video_results = acc_score_video / len(images)
    return video_results

# Motion Smoothness
def compute_motion_smoothness(
    video_chunk: torch.Tensor,
    motion
):
    """
    使用预加载的 MotionSmoothness 模型计算单个视频片段的运动平滑度分数。
    """
    # 直接在整个批次上调用模型进行推理
    score = motion.motion_score(video_chunk)
    return score

# Dynamic Degree
def compute_dynamic_degree(
    video_chunk: torch.Tensor,
    dynamic
):
    """
    使用预加载的 DynamicDegree 模型计算单个视频片段的动态度分数。
    """
    whether_move = dynamic.infer(video_chunk)
    return whether_move

# Subject Consistency
def compute_subject_consistency(
    video_chunk: torch.Tensor,
    model,
):
    from vbench.utils import dino_transform
    device = video_chunk.device
    sim = 0.0
    cnt = 0.0
    image_transform = dino_transform(224)
    images = image_transform(video_chunk)
    for i in range(len(images)):
        with torch.no_grad():
            image = images[i].unsqueeze(0)
            image = image.to(device)
            image_features = model(image)
            image_features = torch.nn.functional.normalize(image_features, dim=-1, p=2)
            if i == 0:
                first_image_features = image_features
            else:
                sim_pre = max(0.0, torch.nn.functional.cosine_similarity(former_image_features, image_features).item())
                sim_fir = max(0.0, torch.nn.functional.cosine_similarity(first_image_features, image_features).item())
                cur_sim = (sim_pre + sim_fir) / 2
                sim += cur_sim
                cnt += 1
        former_image_features = image_features
    sim_per_images = sim / (len(images) - 1)
    return sim_per_images

# background consistency
def compute_background_consistency(
    video_chunk: torch.Tensor,
    clip_model,
    preprocess,
):
    from vbench.utils import clip_transform
    device = "cuda"
    sim = 0.0
    cnt = 0
    image_transform = clip_transform(224)
    images = image_transform(video_chunk)
    images = images.to(device)
    image_features = clip_model.encode_image(images)
    image_features = F.normalize(image_features, dim=-1, p=2)
    for i in range(len(image_features)):
        image_feature = image_features[i].unsqueeze(0)
        if i == 0:
            first_image_feature = image_feature
        else:
            sim_pre = max(0.0, F.cosine_similarity(former_image_feature, image_feature).item())
            sim_fir = max(0.0, F.cosine_similarity(first_image_feature, image_feature).item())
            cur_sim = (sim_pre + sim_fir) / 2
            sim += cur_sim
            cnt += 1
        former_image_feature = image_feature
    sim_per_image = sim / (len(image_features) - 1)
    return sim_per_image


# ==============================================================================
# 2. 实现度量函数的工厂接口
# ==============================================================================

def create_metric_func(metric_name: str, **kwargs) -> callable:
    """
    根据名称和参数创建一个度量函数 (简化版，无缓存)。

    Args:
        metric_name (str): 度量名称。
        **kwargs: 传递给度量函数的额外参数。

    Returns:
        callable: 一个配置好的、可直接调用的度量函数。
    """
    if metric_name == 'brightness':
        # 对于简单度量，直接返回函数本身
        metric_func = simple_brightness_metric
        metric_func.__name__ = 'Brightness'
        return metric_func

    # Clarity
    elif metric_name == "clarity":
        # --- 简化逻辑: 每次调用都直接加载模型 ---
        # device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        # num_frames = kwargs.get('num_frames_to_sample', 5)

        # iqa_model = pyiqa.create_metric("musiq-spaq", device=device)

        # # 使用一个 "闭包" 来封装模型和参数，这比 partial 更直接
        # def metric_closure(video_chunk: torch.Tensor) -> float:
        #     """这个内部函数可以访问外部加载的 iqa_model 和 num_frames"""
        #     return calculate_iqa_score(video_chunk, iqa_model, num_frames)

        # # 为函数设置一个易于理解的名称
        # metric_closure.__name__ = f'{metric_name}'
        # return metric_closure
        from pyiqa.archs.musiq_arch import MUSIQ
        model_path = "pretrained/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth"
        kwargs = {'category': None, 
                  'imaging_quality_preprocessing_mode': 'longer'}
        device = "cuda"

        model = MUSIQ(pretrained_model_path=model_path)
        model.to(device)
        model.training = False

        def metric_closure(video_chunk: torch.Tensor) -> float:
            return compute_imaging_quality(video_chunk, model, **kwargs)
        metric_closure.__name__ = f'{metric_name}'
        return metric_closure
    
    # Motion Smoothness
    elif metric_name == "motion":
        config = "pretrained/amt_model/AMT-S.yaml"
        ckpt = "pretrained/amt_model/amt-s.pth"
        device = "cuda"
        motion = MotionSmoothness(config, ckpt, device=device)

        def metric_closure(video_chunk: torch.Tensor) -> float:
            return compute_motion_smoothness(video_chunk, motion)
        metric_closure.__name__ = f'{metric_name}'
        return metric_closure
    
    # Aesthetic Quality
    elif metric_name == "aesthetic":
        from vbench.aesthetic_quality import get_aesthetic_model, laion_aesthetic
        device = "cuda"
        vit_path = "pretrained/clip_model/ViT-L-14.pt"
        aes_path = "pretrained/aesthetic_model/emb_reader"
        aesthetic_model = get_aesthetic_model(aes_path).to(device)
        clip_model, preprocess = clip.load(vit_path, device=device)

        def metric_closure(video_chunk: torch.Tensor) -> float:
            # manually add batch dimension (batch size = 1) to fit the interface
            aesthetic_avg, video_results = laion_aesthetic(aesthetic_model, clip_model, video_chunk.unsqueeze(0), device)
            return aesthetic_avg
        metric_closure.__name__ = f'{metric_name}'
        return metric_closure
    
    # Dynamic Degree
    elif metric_name == "dynamic":
        from vbench.dynamic_degree import DynamicDegree
        from easydict import EasyDict as edict

        model_path = kwargs.get("model_path", "pretrained/raft_model/models/raft-things.pth")
        device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        args_new = edict({"model": model_path, "small": False, "mixed_precision": False, "alternate_corr": False})
        dynamic = DynamicDegree(args_new, device)

        def metric_closure(video_chunk: torch.Tensor) -> float:
            return compute_dynamic_degree(video_chunk, dynamic)
        metric_closure.__name__ = f'{metric_name}'
        return metric_closure

    # Subject Consistency
    elif metric_name == "subject":
        os.environ["TORCH_HOME"] = "pretrained/"
        submodules_list = {
            'repo_or_dir': 'pretrained/hub/facebookresearch-dino-7c446df',
            'source': 'local', 
            'model': 'dino_vitb16', 
            'read_frame': None
        }
        device = "cuda"
        dino_model = torch.hub.load(**submodules_list).to(device)

        def metric_closure(video_chunk: torch.Tensor) -> float:
            print(video_chunk.device)
            return compute_subject_consistency(video_chunk, dino_model)
        metric_closure.__name__ = f'{metric_name}'
        return metric_closure
    
    # Background Consistency
    elif metric_name == "background":
        device = "cuda"
        vit_path = 'pretrained/clip_model/ViT-B-32.pt'
        clip_model, preprocess = clip.load(vit_path, device=device)
        def metric_closure(video_chunk: torch.Tensor) -> float:
            return compute_background_consistency(video_chunk, clip_model, preprocess)
        metric_closure.__name__ = f'{metric_name}'
        return metric_closure

    else:
        raise ValueError(f"未知的 metric_name: '{metric_name}'。 "
                         f"支持的名称包括 'brightness' 或 pyiqa 库中的模型。")