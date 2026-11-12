"""Deterministic-seeding helper. Used by every training script."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed python, numpy, and (if available) torch + cuda.

    `deterministic_torch=True` flips cuDNN to deterministic mode at the cost of speed —
    appropriate for headline experiments, not for sweeps.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
