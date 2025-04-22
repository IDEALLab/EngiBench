"""This module contains the MMA subroutine used in the thermoelastic2d problem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mmapy import mmasub as external_mmasub
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

RESIDUAL_MAX_VAL = 0.9
ITERATION_MAX = 500
ITERATION_MAX_SMALL = 50
ITERATION_ASYM_MAX = 2.5


@dataclass(frozen=True)
class MMAInputs:
    """Dataclass encapsulating all input parameters for the MMA subroutine."""

    m: int
    """The number of constraints"""
    n: int
    """The number of design variables"""
    iterr: int
    """The current iteration number"""
    xval: NDArray[np.float64]
    """The flattened array of design variables: shape (n,)"""
    xmin: float
    """The lower bounds on the design variables"""
    xmax: float
    """The upper bounds on the design variables"""
    xold1: NDArray[np.float64]
    """The previous design variables at iteration k-1: shape (n, 1)"""
    xold2: NDArray[np.float64]
    """The design variables at iteration k-2: shape (n, 1)"""
    df0dx: NDArray[np.float64]
    """The gradients of the objective function at xval: shape (n,)"""
    fval: NDArray[np.float64]
    """The value of the constraint functions evaluated at xval: shape (m,)"""
    dfdx: NDArray[np.float64]
    """The gradients of the constraint functions at xval: shape (m, n)"""
    low: NDArray[np.float64]
    """The lower asymptotes from the previous iteration: shape (n,)"""
    upp: NDArray[np.float64]
    """The upper asymptotes from the previous iteration: shape (n,)"""
    a0: float
    """The constants a_0 in the term a_0*z"""
    a: NDArray[np.float64]
    """the constants a_i in the terms a_i*z"""
    c: NDArray[np.float64]
    """the constants c_i in the terms c_i*y_i"""
    d: NDArray[np.float64]
    """the constants d_i in the terms 0.5*d_i*(y_i)^2"""
    f0val: float = 0.0
    """The value of the objective function at xval"""


def mmasub(inputs: MMAInputs) -> NDArray[np.float64]:
    """Perform one MMA iteration to solve a nonlinear programming problem using the GCMMA-MMA-Python library.

    Minimize:
        f_0(x) + a_0 * z + sum(c_i * y_i + 0.5 * d_i * (y_i)^2)

    Subject to:
        f_i(x) - a_i * z - y_i <= 0,    i = 1,...,m
        xmin_j <= x_j <= xmax_j,        j = 1,...,n
        z >= 0, y_i >= 0,               i = 1,...,m

    Parameters:
        inputs (MMAInputs): A dataclass encapsulating all input parameters.

    Returns:
        xmma (NDArray[np.float64]): the updated design variables.
    """
    # Unpack parameters from the dataclass.
    m = int(inputs.m)
    n = int(inputs.n)
    iterr = int(inputs.iterr)
    xval = np.expand_dims(inputs.xval, axis=1)
    xmin = np.full((n, 1), inputs.xmin)
    xmax = np.full((n, 1), inputs.xmax)
    xold1 = inputs.xold1
    xold2 = inputs.xold2
    f0val = inputs.f0val
    df0dx = np.expand_dims(inputs.df0dx, axis=1)
    fval = inputs.fval
    dfdx = inputs.dfdx
    low = np.expand_dims(inputs.low, axis=1)
    upp = np.expand_dims(inputs.upp, axis=1)
    a0 = inputs.a0
    a = np.expand_dims(inputs.a, axis=1)
    c = np.expand_dims(inputs.c, axis=1)
    d = np.expand_dims(inputs.d, axis=1)

    # MMA parameters
    raa0 = 1e-5
    move = 1.0
    albefa = 0.1
    asyinit = 0.01
    asyincr = 1.2
    asydecr = 0.7
    asymax = 0.2
    asymin = 0.01

    results = external_mmasub(
        m,
        n,
        iterr,
        xval,
        xmin,
        xmax,
        xold1,
        xold2,
        f0val,
        df0dx,
        fval,
        dfdx,
        low,
        upp,
        a0,
        a,
        c,
        d,
        move=move,
        asyinit=asyinit,
        asydecr=asydecr,
        asyincr=asyincr,
        asymin=asymin,
        asymax=asymax,
        raa0=raa0,
        albefa=albefa,
    )

    return results[0]
