"""Reproducibility utilities for deterministic training.

Sets all random seeds and configures PyTorch/CUDA for deterministic behavior.
This is critical for research-grade work where experiments must be replicable.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility across libraries.

    Args:
        seed: The random seed to use (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set Python environment seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[SEED] Random seed set to {seed}")


def enable_determinism() -> None:
    """Configure PyTorch for deterministic operations.

    Note: This may impact performance. Disable for production training
    and only use for debugging/experiment replication.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print("[DETERM] Deterministic mode enabled (may be slower)")


def print_system_info() -> dict:
    """Print system information for reproducibility logging.

    Returns:
        Dictionary of system information.
    """
    import platform

    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_device_capability"] = torch.cuda.get_device_capability(0)
        info["total_vram_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9

    print("\n" + "=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)
    for key, value in info.items():
        print(f"  {key}: {value}")
    print("=" * 50 + "\n")

    return info
