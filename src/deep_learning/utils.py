"""
Utility Functions for Phase 5 Deep Learning.

Provides random seed reproducibility setters, model parameter count calculators,
and PyTorch device selection helpers.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Returns PyTorch torch.device ('cuda' if available else 'cpu')."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Counts total trainable parameters in a PyTorch nn.Module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
