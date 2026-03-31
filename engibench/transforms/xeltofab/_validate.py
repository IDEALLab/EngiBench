"""Input and output validation for the xeltofab mesh transform."""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from xeltofab import PipelineState


def validate_input(design: npt.NDArray) -> npt.NDArray:
    """Validate and sanitize a density-field design array.

    Args:
        design: A numpy array representing a density field.

    Returns:
        The (possibly clipped) design array.

    Raises:
        TypeError: If *design* is not a numpy array.
        ValueError: If *design* is not 2-D or 3-D.
    """
    if not isinstance(design, np.ndarray):
        msg = f"design must be a numpy ndarray, got {type(design).__name__}"
        raise TypeError(msg)

    if not np.issubdtype(design.dtype, np.floating) and not np.issubdtype(design.dtype, np.integer):
        msg = f"design must have a numeric dtype, got {design.dtype}"
        raise TypeError(msg)

    if design.size == 0:
        msg = f"design must be non-empty, got shape {design.shape}"
        raise ValueError(msg)

    if design.ndim not in (2, 3):
        msg = f"design must be 2-D or 3-D, got {design.ndim}-D with shape {design.shape}"
        raise ValueError(msg)

    # Single pass: min/max propagate NaN, so NaN detection comes for free.
    vmin, vmax = float(design.min()), float(design.max())
    if np.isnan(vmin) or np.isnan(vmax) or np.isinf(vmin) or np.isinf(vmax):
        msg = "design contains non-finite values (NaN or Inf)"
        raise ValueError(msg)

    if vmin < 0.0 or vmax > 1.0:
        warnings.warn(
            f"Design values outside [0, 1] (min={vmin:.4f}, max={vmax:.4f}). Clipping.",
            stacklevel=3,
        )
        design = np.clip(design, 0.0, 1.0)

    return design


def validate_output(
    state: PipelineState,
    input_volume_fraction: float,
    tolerance: float,
) -> list[str]:
    """Run post-pipeline validation checks.

    Args:
        state: A ``xeltofab.PipelineState`` instance.
        input_volume_fraction: Volume fraction of the input design (``np.mean(design)``).
        tolerance: Maximum allowed absolute deviation in volume fraction.

    Returns:
        A list of warning messages (empty if all checks pass).

    Raises:
        RuntimeError: If the pipeline produced no mesh (3-D) or no contours (2-D).
    """
    warnings_list: list[str] = []

    ndim: int = getattr(state, "ndim", 0)

    if ndim == 3:  # noqa: PLR2004
        vertices = getattr(state, "vertices", None)
        faces = getattr(state, "faces", None)
        if vertices is None or faces is None:
            msg = "3-D pipeline produced no mesh (vertices or faces are None)."
            raise RuntimeError(msg)
    elif ndim == 2:  # noqa: PLR2004
        contours = getattr(state, "contours", None)
        if contours is None or len(contours) == 0:
            msg = "2-D pipeline produced no contours."
            raise RuntimeError(msg)
    else:
        msg = f"Unsupported or missing 'ndim' in pipeline state: {ndim!r}. Expected 2 or 3."
        raise RuntimeError(msg)

    output_vf = getattr(state, "volume_fraction", None)
    if output_vf is not None:
        delta = abs(input_volume_fraction - output_vf)
        if delta > tolerance:
            warnings_list.append(
                f"Volume fraction changed by {delta:.4f} "
                f"(input={input_volume_fraction:.4f}, output={output_vf:.4f}, "
                f"tolerance={tolerance:.4f})."
            )

    for msg in warnings_list:
        warnings.warn(msg, stacklevel=3)

    return warnings_list
