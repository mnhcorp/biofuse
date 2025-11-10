"""
Reproducibility utilities for BioFuse.

Provides functions to set seeds and ensure deterministic behavior
across PyTorch, NumPy, and Python's random module.
"""

import os
import random
import torch
import numpy as np
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """
    Set seed for all random number generators to ensure reproducibility.

    This function sets seeds for:
    - PyTorch (CPU and CUDA)
    - NumPy
    - Python's random module
    - CUDA operations (deterministic mode)

    Args:
        seed: Seed value for random number generation. Default is 42.

    Example:
        >>> from biofuse.utils import set_seed
        >>> set_seed(42)  # All subsequent operations will be deterministic
    """
    # Set PyTorch seed
    torch.manual_seed(seed)

    # Set seed for CUDA operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Set NumPy seed
    np.random.seed(seed)

    # Set Python's random module seed
    random.seed(seed)

    # Configure PyTorch backends for deterministic behavior
    # Note: This may impact performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(device: Optional[str] = None, gpu_id: int = 0) -> torch.device:
    """
    Get the appropriate PyTorch device for computation.

    Args:
        device: Device specification ('cuda', 'cpu', or None for auto-detect)
        gpu_id: GPU ID to use if CUDA is available

    Returns:
        PyTorch device object

    Example:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device('cuda')  # Force CUDA
        >>> device = get_device('cuda', gpu_id=1)  # Use GPU 1
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda' and torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    else:
        return torch.device('cpu')


def print_cuda_memory_stats() -> None:
    """
    Print CUDA memory statistics for debugging.

    Prints allocated and reserved memory for all available GPUs.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return

    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)
        print(f"GPU {i}:")
        print(f"  Allocated: {allocated:.2f} MB")
        print(f"  Reserved: {reserved:.2f} MB")
