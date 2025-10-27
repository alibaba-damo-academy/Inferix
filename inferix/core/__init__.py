"""
Core module for Inferix framework.

This module provides the fundamental components and utilities for the Inferix 
deep learning inference framework, organized by functional domains rather than
generic categories.
"""

from .utils import env_is_true, divide, set_random_seed

__all__ = [
    "env_is_true",
    "divide", 
    "set_random_seed",
]