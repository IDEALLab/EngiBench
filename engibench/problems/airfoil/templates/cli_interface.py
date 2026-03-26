"""Dataclasses sent from the main script to scripts inside the airfoil container."""

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
    """Parameters for airfoil_analyse.py."""

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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to python primitives."""
        return {**asdict(self), "task": self.task.name}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisParameters":
        """Deserialize from python primitives."""
        return cls(task=Task[data.pop("task")], **data)


class Algorithm(Enum):
    """Algorithm to be used by optimize."""

    SLSQP = auto()
    SNOPT = auto()


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
    area_initial: float
    area_input_design: float
    ffd_fname: str
    """Path to the FFD file"""
    mesh_fname: str
    """Path to the mesh file"""
    output_dir: str
    """Path to the output directory"""
    opt: Algorithm
    """The optimization algorithm: SLSQP or SNOPT"""
    opt_options: dict[str, Any]
    """The optimization options"""
    use_altitude: bool
    """Whether to use altitude"""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to python primitives."""
        return {**asdict(self), "opt": self.opt.name}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizeParameters":
        """Deserialize from python primitives."""
        return cls(opt=Algorithm[data.pop("opt")], **data)


@dataclass
class PreprocessParameters:
    """Parameters for pre_process.py."""

    design_fname: str
    """Path to the design file"""
    N_sample: int
    """Number of points to sample on the airfoil surface. Defines part of the mesh resolution"""
    n_tept_s: int
    """Number of points on the trailing edge"""
    x_cut: float
    """Blunt edge dimensionless cut location"""
    tmp_xyz_fname: str
    """Path to the temporary xyz file"""
    mesh_fname: str
    """Path to the generated mesh file"""
    ffd_fname: str
    """Path to the generated FFD file"""
    ffd_ymarginu: float
    """Upper (y-axis) margin for the fitted FFD cage"""
    ffd_ymarginl: float
    """Lower (y-axis) margin for the fitted FFD cage"""
    ffd_pts: int
    """Number of FFD points"""
    N_grid: int
    """Number of grid levels to march from the airfoil surface. Defines part of the mesh resolution"""
    s0: float
    """Off-the-wall spacing for the purpose of modeling the boundary layer"""
    input_blunted: bool
    march_dist: float
    """Distance to march the grid from the airfoil surface"""
