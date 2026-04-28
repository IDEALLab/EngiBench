"""Bridge between EngiBench density-field problems and xeltofab mesh generation.

Example::

    from engibench.problems.beams2d import Beams2D
    from engibench.transforms.xeltofab import to_mesh, save

    problem = Beams2D()
    design, _ = problem.random_design()
    state = to_mesh(problem, design)
    save(state, "beam.stl")  # 3-D only
"""

from engibench.transforms.xeltofab._core import save
from engibench.transforms.xeltofab._core import to_mesh
from engibench.transforms.xeltofab._presets import PROBLEM_PRESETS

__all__ = ["PROBLEM_PRESETS", "save", "to_mesh"]
