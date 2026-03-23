"""
Математические вспомогательные функции для постобработки инференса.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray


def stable_softmax(values: np.ndarray) -> NDArray[np.float32]:
    """
    Численно стабильный softmax для runtime-постобработки.
    """
    arr = np.asarray(values, dtype=np.float32)
    shifted = arr - np.max(arr)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp)
    return cast(NDArray[np.float32], np.asarray(probs, dtype=np.float32))
