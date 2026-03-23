from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def stable_softmax(values: np.ndarray) -> NDArray[np.float32]:
    """Numerically stable softmax for runtime inference post-processing."""

    arr = np.asarray(values, dtype=np.float32)
    shifted = arr - np.max(arr)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp)
    return probs.astype(np.float32, copy=False)
