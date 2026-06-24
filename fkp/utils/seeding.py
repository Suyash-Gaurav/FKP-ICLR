"""
Global random seed management for strict reproducibility.

IMPORTANT: Every experiment script must call seed_everything() as the
first operation before any data loading or model instantiation.

Reference:
    §6 "Reproducibility & Paper Generation" in the intern onboarding guide.
"""

from __future__ import annotations

import os
import random

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for fully deterministic execution.

    Covers Python's random module, NumPy, PyTorch (CPU + CUDA), and
    the hash-randomisation seed used by Python's built-in hash().

    Parameters
    ----------
    seed : int
        The master random seed.  Default: 42.
        All FKP experiments use seed=42 for reproducibility.

    Notes
    -----
    CUBLAS_WORKSPACE_CONFIG is set to make cuBLAS deterministic.
    This may slightly reduce performance on GPU.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make cuBLAS deterministic
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
