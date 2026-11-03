"""Training-time augmentation helpers for the current count representation."""

import numpy as np


def sample_global_time_scale(probability=0.35, min_scale=0.85):
    """Sample a reciprocal-symmetric log-uniform scale or identity."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("time scale probability must be in [0, 1]")
    if not 0.0 < min_scale <= 1.0:
        raise ValueError("minimum time scale must be in (0, 1]")
    if np.random.random() >= probability:
        return 1.0
    magnitude = np.random.uniform(0.0, -np.log(min_scale))
    return float(np.exp(magnitude if np.random.randint(2) else -magnitude))
