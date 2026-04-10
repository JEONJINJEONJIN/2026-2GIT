"""Reproducibility utilities for the RSA project.

Sets random seeds across all relevant libraries so that experiments
produce deterministic results (modulo non-deterministic CUDA kernels).
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all frameworks.

    Seeds are set for:
    * :mod:`random` (Python stdlib)
    * :mod:`numpy`
    * :mod:`torch` (CPU **and** all CUDA devices)
    * HuggingFace :func:`transformers.set_seed`

    Additionally enables deterministic CuDNN behaviour when CUDA is
    available.

    Parameters
    ----------
    seed:
        Non-negative integer used as the global random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        logger.debug("transformers not installed; skipping transformers.set_seed")

    logger.info("Global seed set to %d", seed)


def get_seed_from_config(config: Optional[dict] = None) -> int:
    """Retrieve the seed from the experiment configuration.

    The function looks for the seed in the following locations (in
    order):

    1. ``config["experiment"]["seed"]``
    2. ``config["seed"]``

    If *config* is ``None``, the experiment config is loaded
    automatically via :func:`~src.utils.config.get_config`.

    Parameters
    ----------
    config:
        Pre-loaded configuration dictionary.  If ``None``, the
        ``experiment.yaml`` config is loaded from disk.

    Returns
    -------
    int
        The configured seed value.

    Raises
    ------
    KeyError
        If no seed entry is found in the configuration.
    """
    if config is None:
        from src.utils.config import get_config
        config = get_config("experiment")

    # Try nested path first, then top-level
    if "experiment" in config and "seed" in config["experiment"]:
        return int(config["experiment"]["seed"])
    if "seed" in config:
        return int(config["seed"])

    raise KeyError(
        "No 'seed' found in configuration. "
        "Expected at config['experiment']['seed'] or config['seed']."
    )
