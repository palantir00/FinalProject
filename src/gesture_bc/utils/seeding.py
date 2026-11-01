"""One call that makes every random number generator used in training reproducible."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and all GPUs).

    Network initialization and the order of training batches then depend only on ``seed``,
    which is what "training seed" means in the experiments.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 (legacy global generator, used by some libraries)
    torch.manual_seed(seed)  # also seeds CUDA
