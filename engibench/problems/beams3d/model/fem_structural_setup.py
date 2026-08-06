"""Structural-only forward solver setup for the Beams 3D problem."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix

from engibench.problems.beams3d.model.linear_solver import solve_spd_with_amg


@dataclass
class FEMStructuralContext3D:
    """Iteration-invariant structural FEM bookkeeping."""

    ndofsm: int
    km_row: np.ndarray
    km_col: np.ndarray
    fixeddofsm: np.ndarray
    alldofsm: np.ndarray
    freedofsm: np.ndarray
    fp: np.ndarray
    edof24: np.ndarray
    ex: np.ndarray
    ey: np.ndarray
    ez: np.ndarray


@dataclass
class FEMStructuralResult3D:
    """Mechanical assembly and solve results."""

    km: csr_matrix
    """Global mechanical stiffness matrix."""

    um: np.ndarray
    """Mechanical displacement vector."""

    fm: np.ndarray
    """External mechanical load vector."""

    fixeddofsm: np.ndarray
    """Fixed mechanical DOFs."""

    alldofsm: np.ndarray
    """All mechanical DOFs."""

    freedofsm: np.ndarray
    """Free mechanical DOFs."""

    fp: np.ndarray
    """Applied mechanical load vector before boundary-condition solve."""

    edof24: np.ndarray
    """Element-to-mechanical-DOF map with shape (nelem, 24)."""

    ex: np.ndarray
    """Element x indices flattened in assembly order."""

    ey: np.ndarray
    """Element y indices flattened in assembly order."""

    ez: np.ndarray
    """Element z indices flattened in assembly order."""


def _node_index(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, nely: int, nelz: int) -> np.ndarray:
    """Map node coordinates to flat node ids with z as the fastest axis."""
    return (nely + 1) * (nelz + 1) * ix + (nelz + 1) * iy + iz


def _mask_to_indices(mask3d: np.ndarray) -> np.ndarray:
    """Flatten a node mask to the corresponding flat node indices."""
    mask = np.asarray(mask3d, dtype=bool)
    return np.flatnonzero(mask.ravel(order="C")).astype(int)


def _index_dtype(limit: int) -> type[np.int32] | type[np.int64]:
    return np.int32 if limit <= np.iinfo(np.int32).max else np.int64


def build_structural_context_3d(
    nely: int,
    nelx: int,
    nelz: int,
    bcs: dict[str, Any],
) -> FEMStructuralContext3D:
    """Build structural FEM data that does not change across MMA iterations."""
    nn = (nelx + 1) * (nely + 1) * (nelz + 1)
    ndofsm = 3 * nn

    ex_full, ey_full, ez_full = np.meshgrid(np.arange(nelx), np.arange(nely), np.arange(nelz), indexing="ij")
    ex = ex_full.ravel()
    ey = ey_full.ravel()
    ez = ez_full.ravel()
    nelem = ex.size

    n000 = _node_index(ex, ey, ez, nely, nelz)
    n100 = _node_index(ex + 1, ey, ez, nely, nelz)
    n110 = _node_index(ex + 1, ey + 1, ez, nely, nelz)
    n010 = _node_index(ex, ey + 1, ez, nely, nelz)
    n001 = _node_index(ex, ey, ez + 1, nely, nelz)
    n101 = _node_index(ex + 1, ey, ez + 1, nely, nelz)
    n111 = _node_index(ex + 1, ey + 1, ez + 1, nely, nelz)
    n011 = _node_index(ex, ey + 1, ez + 1, nely, nelz)

    edof8 = np.stack([n000, n100, n110, n010, n001, n101, n111, n011], axis=1)
    edof24 = np.stack([3 * edof8 + 0, 3 * edof8 + 1, 3 * edof8 + 2], axis=2).reshape(nelem, 24)

    index_dtype = _index_dtype(ndofsm)
    km_row = np.repeat(edof24, 24, axis=1).astype(index_dtype, copy=False).ravel()
    km_col = np.tile(edof24, 24).astype(index_dtype, copy=False).ravel()

    fix_nodes = _mask_to_indices(bcs["fixed_elements"])
    fixeddofsm = np.concatenate((3 * fix_nodes + 0, 3 * fix_nodes + 1, 3 * fix_nodes + 2))
    alldofsm = np.arange(ndofsm)
    freedofsm = np.setdiff1d(alldofsm, fixeddofsm, assume_unique=True)

    fp = np.zeros(ndofsm, dtype=np.float64)

    def add_load(mask_key: str, comp: int, mag: float = 0.5) -> None:
        """Accumulate nodal loads for one component from a boundary mask."""
        if mask_key in bcs and bcs[mask_key] is not None:
            nodes = _mask_to_indices(bcs[mask_key])
            fp[3 * nodes + comp] += mag

    add_load("force_elements_x", 0)
    add_load("force_elements_y", 1)
    add_load("force_elements_z", 2)

    return FEMStructuralContext3D(
        ndofsm=ndofsm,
        km_row=km_row,
        km_col=km_col,
        fixeddofsm=fixeddofsm,
        alldofsm=alldofsm,
        freedofsm=freedofsm,
        fp=fp,
        edof24=edof24,
        ex=ex,
        ey=ey,
        ez=ez,
    )


def fe_structural_bc_3d(
    nely: int,
    nelx: int,
    nelz: int,
    penal: float,
    x: np.ndarray,
    ke: np.ndarray,
    bcs: dict[str, Any],
    context: FEMStructuralContext3D | None = None,
) -> FEMStructuralResult3D:
    """Assemble and solve the structural 3D FEM system only.

    The design array is indexed as ``x[y, x, z]`` while boundary-condition
    masks are node arrays indexed as ``mask[x, y, z]``.
    """
    if context is None:
        context = build_structural_context_3d(nely, nelx, nelz, bcs)

    penalized = (x[context.ey, context.ex, context.ez] ** penal).astype(np.float64)
    km_blk = penalized[:, None, None] * ke
    km_dat = km_blk.reshape(-1)
    km = coo_matrix((km_dat, (context.km_row, context.km_col)), shape=(context.ndofsm, context.ndofsm))
    km = ((km + km.T) / 2.0).tocsr()

    fm = context.fp.copy()
    um = np.zeros(context.ndofsm, dtype=np.float64)
    if context.freedofsm.size > 0:
        um[context.freedofsm] = solve_spd_with_amg(
            km[context.freedofsm, :][:, context.freedofsm].tocsr(), fm[context.freedofsm]
        )
    if context.fixeddofsm.size > 0:
        um[context.fixeddofsm] = 0.0

    return FEMStructuralResult3D(
        km=km,
        um=um,
        fm=fm,
        fixeddofsm=context.fixeddofsm,
        alldofsm=context.alldofsm,
        freedofsm=context.freedofsm,
        fp=context.fp,
        edof24=context.edof24,
        ex=context.ex,
        ey=context.ey,
        ez=context.ez,
    )
