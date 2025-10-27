import os
import random

import numpy as np
import torch


def env_is_true(env_name: str) -> bool:
    return str(os.environ.get(env_name, "0")).lower() in {"1", "true", "yes", "y", "on", "enabled"}


def divide(numerator, denominator):
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)
    return numerator // denominator


def set_random_seed(seed):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
    """
    assert seed is not None, "Please provide a seed in config.json"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def get_logger():
    """Simple logger function - placeholder for now."""
    import logging
    return logging.getLogger(__name__)