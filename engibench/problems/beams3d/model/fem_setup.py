"""Compatibility exports for the structural Beams 3D FEM setup."""

from engibench.problems.beams3d.model.fem_structural_setup import fe_structural_bc_3d
from engibench.problems.beams3d.model.fem_structural_setup import FEMStructuralResult3D

__all__ = ["FEMStructuralResult3D", "fe_structural_bc_3d"]
