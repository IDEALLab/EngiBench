"""Core bridge between EngiBench density-field problems and xeltofab mesh generation."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from engibench.transforms.xeltofab._presets import PROBLEM_PRESETS
from engibench.transforms.xeltofab._validate import validate_input
from engibench.transforms.xeltofab._validate import validate_output

if TYPE_CHECKING:
    from pathlib import Path

    from engibench.core import Problem

try:
    from xeltofab import PipelineParams
    from xeltofab import PipelineState
    from xeltofab import process as _process
    from xeltofab import save_mesh as _save_mesh

    _HAS_XELTOFAB = True
except ImportError:
    _HAS_XELTOFAB = False


def _check_xeltofab() -> None:
    if not _HAS_XELTOFAB:
        msg = (
            "xeltofab >= 0.3.0 is required for mesh transforms. "
            "Install with: pip install 'engibench[transforms]'  (requires Python >= 3.13)"
        )
        raise ImportError(msg)


def to_mesh(
    problem: Problem,
    design: npt.NDArray,
    *,
    params: PipelineParams | None = None,
    validate: bool = True,
    volume_tolerance: float = 0.05,
    **kwargs: Any,
) -> PipelineState:
    """Convert an EngiBench density-field design to a mesh via xeltofab.

    Args:
        problem: An EngiBench ``Problem`` instance.  Used to look up
            per-problem pipeline presets by class name.
        design: A 2-D or 3-D numpy array with values in ``[0, 1]``
            representing a density field.
        params: Explicit ``PipelineParams``.  When provided, presets and
            *kwargs* are ignored for pipeline parameters.
        validate: Whether to run post-conversion validation checks.
        volume_tolerance: Maximum allowed absolute change in volume
            fraction between input field and output mesh / contours.
        **kwargs: Forwarded to ``PipelineParams(...)`` and merged on top
            of the per-problem preset defaults.

    Returns:
        A ``PipelineState`` containing the processed mesh (3-D) or
        contours (2-D).

    Raises:
        TypeError: If *design* is not a numpy array.
        ValueError: If *design* is not 2-D or 3-D.
        ImportError: If xeltofab is not installed.
    """
    _check_xeltofab()

    design = validate_input(design)

    if params is None:
        problem_name = type(problem).__name__
        preset = PROBLEM_PRESETS.get(problem_name, {})
        merged = {**preset, **kwargs}
        params = PipelineParams(**merged)

    input_vf = float(np.mean(design)) if validate else 0.0

    # Preserve float32 to avoid doubling memory for large 3D fields.
    if not np.issubdtype(design.dtype, np.floating):
        design = design.astype(np.float64, copy=False)

    state = PipelineState(field=design, params=params)
    state = _process(state)

    if validate:
        validate_output(state, input_vf, volume_tolerance)

    return state


def save(state: PipelineState, path: str | Path) -> None:
    """Save a processed mesh to a file.

    Convenience wrapper around ``xeltofab.save_mesh``.

    Args:
        state: A ``PipelineState`` returned by :func:`to_mesh`.
        path: Output file path.  Format is inferred from the extension
            (``.stl``, ``.obj``, ``.ply``).

    Raises:
        ImportError: If xeltofab is not installed.
    """
    _check_xeltofab()
    _save_mesh(state, str(path))
