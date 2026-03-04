"""
Utility functions for the Wings3D problem.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def get_slice_coords(coords: npt.NDArray, slice_num: int) -> npt.NDArray[np.float32]:
    """
    Extract one slice curve from coords.

    Expected shapes:
      - (9, 192, 2): full wing sections, return coords[slice_num] -> (192, 2)
      - (192, 2): already a single slice, return as-is
    """
    arr = np.asarray(coords, dtype=np.float32)

    if arr.ndim == 3:
        return arr[int(slice_num)]
    if arr.ndim == 2:
        return arr

    raise ValueError(f"Unexpected coords shape: {arr.shape}")