"""Dataclasses sent from the main script to scripts inside the airfoil container (v1).

This mirrors the v0 ``cli_interface`` but extends it for the v1 problem:

- ``Algorithm`` gains ``IPOPT`` (the optimizer the v1 dataset was collected with).
- ``OptimizeParameters`` / ``AnalysisParameters`` gain two *advanced* knobs that are
  ``None`` by default so a normal user never has to touch them:
  ``surface_variables`` (which surface fields ADflow writes) and ``solver_options``
  (an escape hatch dict merged last into the ADflow options, e.g. to override the
  internal per-Mach ANK/NK schedule).
- ``PreprocessParameters`` carries the pyHyp options used by the v1 mesh so the
  mesh is built identically to the reference 2D study.
"""

# Lazy annotations so the ``X | None`` field hints also parse under the container's
# (pre-3.10) Python interpreter, where they would otherwise raise at class definition.
from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Any


class Task(Enum):
    """The task to perform by analysis."""

    ANALYSIS = auto()
    POLAR = auto()


@dataclass
class AnalysisParameters:
    """Parameters for airfoil_analysis.py."""

    mesh_fname: str
    """Path to the mesh file"""
    output_dir: str
    """Path to the output directory"""
    task: Task
    """The task to perform: "analysis" or "polar"""
    mach: float
    """mach number"""
    reynolds: float
    """Reynolds number"""
    altitude: float
    """altitude"""
    temperature: float
    """temperature"""
    use_altitude: bool
    """Whether to use altitude"""
    alpha: float
    """Angle of attack"""
    surface_variables: list[str] | None = None
    """Advanced: surface fields ADflow writes. ``None`` -> a sensible default set."""
    solver_options: dict[str, Any] | None = None
    """Advanced: extra ADflow options merged last (overrides the internal schedule)."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to python primitives."""
        return {**asdict(self), "task": self.task.name}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisParameters:
        """Deserialize from python primitives."""
        return cls(task=Task[data.pop("task")], **data)


class Algorithm(Enum):
    """Algorithm to be used by optimize."""

    SLSQP = auto()
    SNOPT = auto()
    IPOPT = auto()


@dataclass
class OptimizeParameters:
    """Parameters for airfoil_opt.py."""

    cl_target: float
    """The lift coefficient constraint"""
    alpha: float
    """The angle of attack"""
    mach: float
    """The Mach number"""
    reynolds: float
    """Reynolds number"""
    temperature: float
    """Temperature"""
    altitude: int
    """The cruising altitude"""
    area_ratio_min: float
    """Minimum fraction of the (baseline) area the section is allowed to shrink to"""
    area_initial: float
    """Baseline airfoil area used as the constraint reference"""
    area_input_design: float
    """Area of the actual input design (so the scaled constraint is baseline-consistent)"""
    ffd_fname: str
    """Path to the FFD file"""
    mesh_fname: str
    """Path to the mesh file"""
    output_dir: str
    """Path to the output directory"""
    opt: Algorithm
    """The optimization algorithm: SLSQP, SNOPT or IPOPT"""
    opt_options: dict[str, Any]
    """Extra optimizer options (merged onto the built-in defaults for ``opt``)"""
    use_altitude: bool
    """Whether to use altitude"""
    surface_variables: list[str] | None = None
    """Advanced: surface fields ADflow writes. ``None`` -> the full default set."""
    solver_options: dict[str, Any] | None = None
    """Advanced: extra ADflow options merged last (overrides the internal schedule)."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to python primitives."""
        return {**asdict(self), "opt": self.opt.name}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizeParameters:
        """Deserialize from python primitives."""
        return cls(opt=Algorithm[data.pop("opt")], **data)


@dataclass
class PreprocessParameters:
    """Parameters for pre_process.py."""

    design_fname: str
    """Path to the design file"""
    tmp_xyz_fname: str
    """Path to the temporary xyz file"""
    mesh_fname: str
    """Path to the generated mesh file"""
    ffd_fname: str
    """Path to the generated FFD file"""
    N_sample: int = 201
    """Number of points to sample on the airfoil surface (part of the mesh resolution)"""
    n_tept_s: int = 4
    """Number of points on the (blunt) trailing edge"""
    x_cut: float = 0.98
    """Blunt-edge dimensionless cut location (the v1 baselines reach x=0.98)"""
    ffd_ymarginu: float = 0.05
    """Upper (y-axis) margin for the fitted FFD cage"""
    ffd_ymarginl: float = 0.05
    """Lower (y-axis) margin for the fitted FFD cage"""
    ffd_pts: int = 10
    """Number of chordwise FFD points"""
    N_grid: int = 55
    """Number of off-wall layers to march from the surface (part of the mesh resolution)"""
    s0: float = 4e-6
    """Off-the-wall spacing (floored at 5e-6 inside the script for stability)"""
    input_blunted: bool = True
    """v1 baselines are already blunt, so do NOT re-blunt the section"""
    march_dist: float = 80.0
    """Farfield march distance (chords)"""
    # ---- pyHyp options, matched to the reference 2D mesh_template ----
    vol_coef: float = 0.25
    vol_blend: float = 1e-4
    vol_smooth_iter: int = 50
    eps_e: float = 1.0
    eps_i: float = 2.0
    theta: float = 3.0
    ps0: float = -1.0
    p_grid_ratio: float = -1.0
    c_max: float = 1.0
    ksp_rel_tol: float = 1e-10
    ksp_max_its: int = 2000
    ksp_subspace_size: int = 50

    def to_dict(self) -> dict[str, Any]:
        """Serialize to python primitives."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessParameters:
        """Deserialize from python primitives."""
        return cls(**data)


# Default surface fields written by the v1 solver (the full set the dataset was
# collected with). Exposed so callers can request a subset via ``surface_variables``.
DEFAULT_SURFACE_VARIABLES: list[str] = [
    "rho",
    "p",
    "temp",
    "vx",
    "vy",
    "vz",
    "cp",
    "mach",
    "ch",
    "cfx",
    "cfy",
    "cfz",
    "forceInDragDir",
    "forceInLiftDir",
    "yplus",
    "sepsensor",
    "sepsensorks",
    "sepsensorksarea",
]
