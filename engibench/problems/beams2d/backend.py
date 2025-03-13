# ruff: noqa: E741, N806, N815, N816
# Disabled variable name conventions

"""Beams 2D problem.

This code has been adapted from the Python implementation by Niels Aage and Villads Egede Johansen: https://github.com/arjendeetman/TopOpt-MMA-Python
"""

from __future__ import annotations

from copy import deepcopy
import dataclasses
from typing import Any

import cvxopt
import cvxopt.cholmod
import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix
from scipy.sparse import csc_array


@dataclasses.dataclass
class Params:
    """A structured representation of configuration parameters for a numerical computation.

    Attributes:
        nelx (int): Width of the design domain (100 by default).
        nely (int): Height of the design domain (50 by default).
        volfrac (float): Desired solid volume fraction of the beam (0.35 by default).
        penal (float): Intermediate material penalization term (3.0 by default).
        rmin (float): Minimum feature scale (i.e., beam element width, 2.0 by default).
        ft (int): 0 for sensitivity-based filter, 1 for density-based filter (1 by default).
        max_iter (int): Maximum optimization iterations, assuming no convergence (100 by default).
        overhang_constraint (bool): Whether to use a 45-degree overhang constraint in optimization (False by default).
        ndof (int): Number of degrees of freedom.
        Emin: (float) Minimum possible stiffness (1e-9 by default).
        Emax: (float) Maximum possible stiffness (1 by default).
        min_change (float): Minimum change in terms of design variables between two consecutive designs to continue optimization (0.025 by default).
        min_ratio (float): Parameter determining when the bisection search on the Lagrange multiplier should stop (1e-3 by default).
        edofMat (np.ndarray): Element degrees of freedom mapping.
        iK (np.ndarray): Row indices for stiffness matrix.
        jK (np.ndarray]): Column indices for stiffness matrix.
        H (csc_array): Filter matrix.
        Hs (np.ndarray): Filter normalization factor.
        dofs (np.ndarray): Degrees of freedom indices.
        fixed (np.ndarray): Fixed degrees of freedom.
        free (np.ndarray): Free degrees of freedom.
        f (np.ndarray): Force vector.
        u (np.ndarray): Displacement vector.
        KE (np.ndarray): Stiffness matrix.
    """

    # Boundary conditions (editable by user)
    nelx: int = dataclasses.field(default=0)
    nely: int = dataclasses.field(default=0)
    volfrac: float = dataclasses.field(default=0)
    penal: float = dataclasses.field(default=0)
    rmin: float = dataclasses.field(default=0)
    ft: int = dataclasses.field(default=0)
    max_iter: int = dataclasses.field(default=0)
    overhang_constraint: bool = dataclasses.field(default=False)

    # Other parameters (non-editable)
    Emin: float = 1e-9
    Emax: float = 1.0
    min_change: float = 0.025
    min_ratio: float = 1.0e-3
    ndof: int = dataclasses.field(default=0)
    edofMat: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))
    iK: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))
    jK: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))
    H: csc_array = dataclasses.field(default_factory=lambda: csc_array((0, 0)))
    Hs: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))
    dofs: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))
    fixed: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))
    free: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))
    f: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))
    u: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))
    KE: np.ndarray = dataclasses.field(default_factory=lambda: np.array(lk()))

    def get(self, keys: list[str]) -> dict[str, Any]:
        """Retrieve a subset of parameter values based on a list of keys.

        Args:
            keys (List[str]): List of parameter names to retrieve.

        Returns:
            Dict[str, Any]: Dictionary of requested parameter names and values.
        """
        return {key: getattr(self, key) for key in keys if hasattr(self, key)}

    def update(self, updates: dict[str, Any]) -> None:
        """Update multiple parameter values efficiently.

        Args:
            updates (Dict[str, Any]): Dictionary of key-value pairs to update.
        """
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing all parameters.
        """
        return dataclasses.asdict(self)


def image_to_design(im: npt.NDArray) -> npt.NDArray:
    r"""Flatten the 2D image(s) to 1D vector(s).

    Args:
        im (npt.NDArray): The image(s) to convert.

    Returns:
        npt.NDArray: The transformed vector(s).
    """
    return np.swapaxes(im, -2, -1).reshape(*im.shape[:-2], -1)


def design_to_image(x: npt.NDArray, nelx: int = 100, nely: int = 50) -> npt.NDArray:
    r"""Reshape the 1D vector(s) into 2D image(s).

    Args:
        x (npt.NDArray): The design(s) to convert.
        nelx (int): Width of the problem domain.
        nely (int): Height of the problem domain.

    Returns:
        npt.NDArray: The transformed image(s).
    """
    return np.swapaxes(x.reshape(*x.shape[:-1], nelx, nely), -2, -1)


def lk() -> npt.NDArray:
    r"""Set up the stiffness matrix.

    Returns:
        KE (npt.NDArray): The stiffness matrix.
    """
    E = 1  # 1
    nu = 0.3  # 0.3
    k = np.array(
        [
            1 / 2 - nu / 6,
            1 / 8 + nu / 8,
            -1 / 4 - nu / 12,
            -1 / 8 + 3 * nu / 8,
            -1 / 4 + nu / 12,
            -1 / 8 - nu / 8,
            nu / 6,
            1 / 8 - 3 * nu / 8,
        ]
    )
    KE = (
        E
        / (1 - nu**2)
        * np.array(
            [
                [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
            ]
        )
    )
    return KE


def calc_sensitivity(design: npt.NDArray, p: Params) -> npt.NDArray:
    """Simulates the performance of a beam design. Assumes the Params object is already set up.

    Args:
        design (np.ndarray): The design to simulate.
        p: Params object with configs (e.g., boundary conditions) and needed vectors/matrices for the simulation.

    Returns:
        npt.NDArray: The sensitivity of the current design.
    """
    sK = ((p.KE.flatten()[np.newaxis]).T * (p.Emin + (design) ** p.penal * (p.Emax - p.Emin))).flatten(order="F")
    K = coo_matrix((sK, (p.iK, p.jK)), shape=(p.ndof, p.ndof)).tocsc()
    m = K.shape[0]
    keep = np.delete(np.arange(0, m), p.fixed)
    K = K[keep, :]
    keep = np.delete(np.arange(0, m), p.fixed)
    K = K[:, keep].tocoo()
    # Solve system
    K = cvxopt.spmatrix(K.data, K.row.astype(int), K.col.astype(int))
    B = cvxopt.matrix(p.f[p.free, 0])
    cvxopt.cholmod.linsolve(K, B)
    p.u[p.free, 0] = np.array(B)[:, 0]

    ############################################################################################################
    # Sensitivity
    ce = (np.dot(p.u[p.edofMat].reshape(p.nelx * p.nely, 8), p.KE) * p.u[p.edofMat].reshape(p.nelx * p.nely, 8)).sum(1)
    return np.array(ce)


def setup(p: Params) -> Params:
    r"""Set up the matrices and parameters for optimization or simulation.

    Args:
        p: Params object with initial configuration (e.g., boundary conditions).

    Returns:
        Params object with the relevant matrices and other parameters used in optimization and simulation.
    """
    ndof = 2 * (p.nelx + 1) * (p.nely + 1)
    edofMat = np.zeros((p.nelx * p.nely, 8), dtype=int)
    for elx in range(p.nelx):
        for ely in range(p.nely):
            el = ely + elx * p.nely
            n1 = (p.nely + 1) * elx + ely
            n2 = (p.nely + 1) * (elx + 1) + ely
            edofMat[el, :] = np.array(
                [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1]
            )
    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()

    nfilter = int(p.nelx * p.nely * ((2 * (np.ceil(p.rmin) - 1) + 1) ** 2))
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    cc = 0
    for i in range(p.nelx):
        for j in range(p.nely):
            row = i * p.nely + j
            kk1 = int(np.maximum(i - (np.ceil(p.rmin) - 1), 0))
            kk2 = int(np.minimum(i + np.ceil(p.rmin), p.nelx))
            ll1 = int(np.maximum(j - (np.ceil(p.rmin) - 1), 0))
            ll2 = int(np.minimum(j + np.ceil(p.rmin), p.nely))
            for k in range(kk1, kk2):
                for l in range(ll1, ll2):
                    col = k * p.nely + l
                    fac = p.rmin - np.sqrt((i - k) * (i - k) + (j - l) * (j - l))
                    iH[cc] = row
                    jH[cc] = col
                    sH[cc] = np.maximum(0.0, fac)
                    cc = cc + 1
    # Finalize assembly and convert to csc format
    H = coo_matrix((sH, (iH, jH)), shape=(p.nelx * p.nely, p.nelx * p.nely)).tocsc()
    Hs = H.sum(1)

    # BC's and support
    dofs = np.arange(2 * (p.nelx + 1) * (p.nely + 1))
    fixed = np.union1d(dofs[0 : 2 * (p.nely + 1) : 2], np.array([2 * (p.nelx + 1) * (p.nely + 1) - 1]))
    free = np.setdiff1d(dofs, fixed)

    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))

    # Set load
    f[1, 0] = -1

    p.update(
        {
            "ndof": ndof,
            "edofMat": edofMat,
            "iK": iK,
            "jK": jK,
            "H": H,
            "Hs": Hs,
            "dofs": dofs,
            "fixed": fixed,
            "free": free,
            "f": f,
            "u": u,
        }
    )

    return p


def overhang_filter(
    x: npt.NDArray, p: Params, dc: npt.NDArray, dv: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Topology Optimization (TO) filter.

    Args:
        x: (npt.NDArray) The current density field during optimization.
        p: Params object with configs (e.g., boundary conditions) and needed vectors/matrices for the optimization.
        dc: (npt.NDArray) The sensitivity field wrt. compliance.
        dv: (npt.NDArray) The sensitivity field wrt. volume fraction.

    Returns:
        Tuple[npt.NDArray, npt.NDArray, npt.NDArray]: The updated design, sensitivity dc, and sensitivity dv, respectively.
    """
    if p.overhang_constraint:
        P = 40
        ep = 1e-4
        xi_0 = 0.5
        Ns = 3
        nSens = 2  # dc and dv (hard-coded)

        x = design_to_image(x, p.nelx, p.nely)
        dc = design_to_image(dc, p.nelx, p.nely)
        dv = design_to_image(dv, p.nelx, p.nely)

        Q = P + np.log(Ns) / np.log(xi_0)
        SHIFT = 100 * (np.finfo(float).tiny) ** (1 / P)
        BACKSHIFT = 0.95 * (Ns ** (1 / Q)) * (SHIFT ** (P / Q))
        xi = np.zeros(x.shape)
        Xi = np.zeros(x.shape)
        keep = np.zeros(x.shape)
        sq = np.zeros(x.shape)

        xi[p.nely - 1, :] = deepcopy(x[p.nely - 1, :])
        for i in reversed(range(p.nely - 1)):
            cbr = np.array([0, *list(xi[i + 1, :]), 0]) + SHIFT
            keep[i, :] = cbr[: p.nelx] ** P + cbr[1 : p.nelx + 1] ** P + cbr[2:] ** P
            Xi[i, :] = keep[i, :] ** (1 / Q) - BACKSHIFT
            sq[i, :] = np.sqrt((x[i, :] - Xi[i, :]) ** 2 + ep)
            xi[i, :] = 0.5 * ((x[i, :] + Xi[i, :]) - sq[i, :] + np.sqrt(ep))

        if np.sum(dc) != 0:
            dc_copy = deepcopy(dc)
            dv_copy = deepcopy(dv)
            dfxi = [np.array(dc_copy), np.array(dv_copy)]
            dfx = [np.array(dc_copy), np.array(dv_copy)]
            lamb = np.zeros((nSens, p.nelx))
            for i in range(p.nely - 1):
                dsmindx = 0.5 * (1 - (x[i, :] - Xi[i, :]) / sq[i, :])
                dsmindXi = 1 - dsmindx
                cbr = np.array([0, *list(xi[i + 1, :]), 0]) + SHIFT

                dmx = np.zeros((Ns, p.nelx))
                for j in range(Ns):
                    dmx[j, :] = (P / Q) * (keep[i, :] ** (1 / Q - 1)) * (cbr[j : p.nelx + j] ** (P - 1))

                qi = np.ravel([[i] * 3 for i in range(p.nelx)])
                qj = qi + [-1, 0, 1] * p.nelx
                qs = np.ravel(dmx.T)

                dsmaxdxi = coo_matrix((qs[1:-1], (qi[1:-1], qj[1:-1]))).tocsc()
                for k in range(nSens):
                    dfx[k][i, :] = dsmindx * (dfxi[k][i, :] + lamb[k, :])
                    lamb[k, :] = ((dfxi[k][i, :] + lamb[k, :]) * dsmindXi) @ dsmaxdxi

            i = p.nely - 1
            for k in range(nSens):
                dfx[k][i, :] = dfx[k][i, :] + lamb[k, :]

            dc = dfx[0]
            dv = dfx[1]

        xi = image_to_design(xi)
        dc = image_to_design(dc)
        dv = image_to_design(dv)

    else:
        xi = x

    return (xi, dc, dv)


def inner_opt(x: npt.NDArray, p: Params, dc: npt.NDArray, dv: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Inner optimization loop: Lagrange Multiplier Optimization.

    Args:
        x: (npt.NDArray) The current density field during optimization.
        p: Params object with configs (e.g., boundary conditions) and needed vectors/matrices for the optimization.
        dc: (npt.NDArray) The sensitivity field wrt. compliance.
        dv: (npt.NDArray) The sensitivity field wrt. volume fraction.

    Returns:
        Tuple of:
            npt.NDArray: The raw density field
            npt.NDArray: The processed density field (without overhang constraint)
            npt.NDArray: The processed density field (with overhang constraint if applicable)
    """
    # Optimality criteria
    l1, l2, move = (0, 1e9, 0.2)
    # reshape to perform vector operations
    xnew = np.zeros(p.nelx * p.nely)

    while l1 + l2 > 0 and (l2 - l1) / (l1 + l2) > p.min_ratio:
        lmid = 0.5 * (l2 + l1)
        if lmid > 0:
            xnew = np.maximum(
                0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid))))
            )  # type: ignore
        else:
            xnew = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, x + move)))

        # Filter design variables
        if p.ft == 0:
            xPhys = xnew
        elif p.ft == 1:
            xPhys = np.asarray(p.H * xnew[np.newaxis].T / p.Hs)[:, 0]

        xPrint, _, _ = overhang_filter(xPhys, p, dc, dv)

        if xPrint.sum() > p.volfrac * p.nelx * p.nely:
            l1 = lmid
        else:
            l2 = lmid

    return (xnew, xPhys, xPrint)
