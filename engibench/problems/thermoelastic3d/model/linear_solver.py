"""Module for solving sparse linear systems."""

import numpy as np
from numpy.typing import NDArray
import pyamg
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import spsolve


def solve_spd_with_amg(
    a: csr_matrix, b: NDArray[np.float64], tol: float = 1e-8, maxiter: int | None = None
) -> NDArray[np.float64]:
    """Solves a symmetric positive-definite linear system using AMG-preconditioned CG.

    This function applies Algebraic Multigrid (AMG) as a right preconditioner for the
    Conjugate Gradient (CG) method to solve large sparse SPD linear systems. If CG fails
    to converge within the prescribed tolerance, the solution is refined using additional
    AMG V-cycles.

    Args:
        a (csr_matrix): Sparse system matrix in compressed sparse row (CSR) format.
            Must be symmetric positive-definite for CG to converge.
        b (np.ndarray): Right-hand-side vector of shape (n,).
        tol (float): Relative tolerance used for CG convergence.
        maxiter (Optional[int]): Maximum number of CG iterations. If None, the default
            SciPy value is used.

    Returns:
        np.ndarray: The computed solution vector of shape (n,).
    """
    ml = pyamg.smoothed_aggregation_solver(a)
    m = ml.aspreconditioner()

    x, info = cg(a, b, rtol=tol, M=m, maxiter=maxiter)

    if info != 0:
        x = ml.solve(b, x0=x, tol=tol)

    if not np.all(np.isfinite(x)):
        raise RuntimeError(f"Linear solver failed: solution contains non-finite values: {info}")

    return x


def solve_with_spsolve(a: csr_matrix, b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solves a sparse linear system using SciPy's direct sparse solver.

    This function computes the solution of a sparse linear system using a direct
    factorization method (SuperLU via SciPy). It supports general sparse matrices,
    including non-symmetric or indefinite systems.

    Args:
        a (csr_matrix): Sparse system matrix in CSR format.
        b (np.ndarray): Right-hand-side vector of shape (n,).

    Returns:
        np.ndarray: The computed solution vector of shape (n,).
    """
    return spsolve(a, b)
