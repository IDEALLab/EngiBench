"""Per-problem default pipeline parameters for xeltofab mesh generation.

This module contains no xeltofab imports — it stores plain dicts of kwargs
that are passed to ``PipelineParams(...)`` at runtime.
"""

from __future__ import annotations

from typing import Any

# 2D problems: contour extraction only; repair/remesh/decimate are 3D-only and irrelevant.
# 3D problems: full pipeline with marching cubes, repair, remesh, and decimation.

_BASE_2D: dict[str, Any] = {
    "field_type": "density",
    "threshold": 0.5,
    "smooth_sigma": 1.0,
    "morph_radius": 1,
}

_BASE_3D: dict[str, Any] = {
    **_BASE_2D,
    "extraction_method": "mc",
    "repair": True,
    "remesh": True,
    "decimate": True,
    "decimate_ratio": 0.5,
}

PROBLEM_PRESETS: dict[str, dict[str, Any]] = {
    # --- 2D density-field problems ---
    "Beams2D": {**_BASE_2D},
    "ThermoElastic2D": {**_BASE_2D, "smooth_sigma": 0.8},
    "Photonics2D": {
        **_BASE_2D,
        "smooth_sigma": 0.5,  # lower: photonics designs have sharp features
        "morph_radius": 0,  # no morphological cleanup for sharp features
    },
    "HeatConduction2D": {**_BASE_2D},
    # --- 3D density-field problems ---
    "ThermoElastic3D": {**_BASE_3D},
    "HeatConduction3D": {**_BASE_3D},
}
