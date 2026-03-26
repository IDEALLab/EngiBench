"""Helper functions for mini CLI's used by problems to pass data between scripts."""

import io
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray


def np_array_to_bytes(arr: NDArray[Any]) -> bytes:
    """Serialize a numpy arr to bytes.

    To be passed to a CLI calling `np_array_from_stdin`
    """
    serialized = io.BytesIO()
    np.save(serialized, arr)
    return serialized.getvalue()


def np_array_from_stdin() -> NDArray[Any]:
    """Load a numpy array from stdin."""
    return np.load(io.BytesIO(sys.stdin.buffer.read()))


def cast_argv(*types: type[Any]) -> tuple[Any, ...]:
    """Cast argv to the given types."""
    return tuple(t(arg) for t, arg in zip(types, sys.argv[1:], strict=True))
