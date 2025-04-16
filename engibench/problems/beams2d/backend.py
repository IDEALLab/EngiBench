# ruff: noqa: N806, N815, N816
# Disabled variable name conventions

"""Beams 2D problem.

This code has been adapted from the Python implementation by Niels Aage and Villads Egede Johansen: https://github.com/arjendeetman/TopOpt-MMA-Python
"""

from __future__ import annotations

import dataclasses
from typing import Any, overload

import cvxopt
import cvxopt.cholmod
import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix
from scipy.sparse import csc_array


@dataclasses.dataclass
class State:
    """A structured representation of scalars and matrices for optimization and simulation.

    Attributes:
        Emin: (float) Minimum possible stiffness (1e-9 by default).
        Emax: (float) Maximum possible stiffness (1 by default).
        min_change (float): Minimum change in terms of design variables between two consecutive designs to continue optimization (0.025 by default).
        min_ratio (float): Parameter determining when the bisection search on the Lagrange multiplier should stop (1e-3 by default).
        ndof (int): Number of degrees of freedom.
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

    # Non-editable Constants
    Emin: float = 1e-9
    Emax: float = 1.0
    min_change: float = 0.025
    min_ratio: float = 1.0e-3

    # Items calculated for optimization and simulation
    ndof: int = 0
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
        """Retrieve a subset of state values based on a list of keys.

        Args:
            keys (List[str]): List of state names to retrieve.

        Returns:
            Dict[str, Any]: Dictionary of requested state names and values.
        """
        return {key: getattr(self, key) for key in keys if hasattr(self, key)}

    def update(self, updates: dict[str, Any]) -> None:
        """Update multiple state values efficiently.

        Args:
            updates (Dict[str, Any]): Dictionary of key-value pairs to update.
        """
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing all state values.
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
    return (
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


def calc_sensitivity(design: npt.NDArray, st: State, cfg: dict[str, Any] | None = None) -> npt.NDArray:
    """Simulates the performance of a beam design. Assumes the State object is already set up.

    Args:
        design (np.ndarray): The design to simulate.
        st: State object with needed vectors/matrices for the simulation.
        cfg (dict): A dictionary with configurations (e.g., boundary conditions) for the simulation.

    Returns:
        npt.NDArray: The sensitivity of the current design.
    """
    cfg = cfg or {}
    sK = ((st.KE.flatten()[np.newaxis]).T * (st.Emin + design ** cfg["penal"] * (st.Emax - st.Emin))).flatten(order="F")
    K = coo_matrix((sK, (st.iK, st.jK)), shape=(st.ndof, st.ndof)).tocsc()
    m = K.shape[0]
    keep = np.delete(np.arange(0, m), st.fixed)
    K = K[keep, :]
    keep = np.delete(np.arange(0, m), st.fixed)
    K = K[:, keep].tocoo()
    # Solve system
    K = cvxopt.spmatrix(K.data, K.row.astype(int), K.col.astype(int))
    B = cvxopt.matrix(st.f[st.free, 0])
    cvxopt.cholmod.linsolve(K, B)
    st.u[st.free, 0] = np.array(B)[:, 0]

    ############################################################################################################
    # Sensitivity
    ce = (
        np.dot(st.u[st.edofMat].reshape(cfg["nelx"] * cfg["nely"], 8), st.KE)
        * st.u[st.edofMat].reshape(cfg["nelx"] * cfg["nely"], 8)
    ).sum(1)
    return np.array(ce)


def setup(cfg: dict[str, Any] | None = None) -> State:
    r"""Set up the scalars and matrices for optimization or simulation.

    Args:
        cfg (dict): A dictionary with configurations (e.g., boundary conditions) for the optimization or simulation.

    Returns:
        State object with the relevant scalars and matrices used in optimization and simulation.
    """
    st = State()
    cfg = cfg or {}

    ndof = 2 * (cfg["nelx"] + 1) * (cfg["nely"] + 1)
    edofMat = np.zeros((cfg["nelx"] * cfg["nely"], 8), dtype=int)
    for elx in range(cfg["nelx"]):
        for ely in range(cfg["nely"]):
            el = ely + elx * cfg["nely"]
            n1 = (cfg["nely"] + 1) * elx + ely
            n2 = (cfg["nely"] + 1) * (elx + 1) + ely
            edofMat[el, :] = np.array(
                [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1]
            )
    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()

    nfilter = int(cfg["nelx"] * cfg["nely"] * ((2 * (np.ceil(cfg["rmin"]) - 1) + 1) ** 2))
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    cc = 0
    for i in range(cfg["nelx"]):
        for j in range(cfg["nely"]):
            row = i * cfg["nely"] + j
            kk1 = int(np.maximum(i - (np.ceil(cfg["rmin"]) - 1), 0))
            kk2 = int(np.minimum(i + np.ceil(cfg["rmin"]), cfg["nelx"]))
            ll1 = int(np.maximum(j - (np.ceil(cfg["rmin"]) - 1), 0))
            ll2 = int(np.minimum(j + np.ceil(cfg["rmin"]), cfg["nely"]))
            for k in range(kk1, kk2):
                for l in range(ll1, ll2):
                    col = k * cfg["nely"] + l
                    fac = cfg["rmin"] - np.sqrt((i - k) * (i - k) + (j - l) * (j - l))
                    iH[cc] = row
                    jH[cc] = col
                    sH[cc] = np.maximum(0.0, fac)
                    cc = cc + 1
    # Finalize assembly and convert to csc format
    H = coo_matrix((sH, (iH, jH)), shape=(cfg["nelx"] * cfg["nely"], cfg["nelx"] * cfg["nely"])).tocsc()
    Hs = H.sum(1)

    # BC's and support
    dofs = np.arange(2 * (cfg["nelx"] + 1) * (cfg["nely"] + 1))
    fixed = np.union1d(dofs[0 : 2 * (cfg["nely"] + 1) : 2], np.array([2 * (cfg["nelx"] + 1) * (cfg["nely"] + 1) - 1]))
    free = np.setdiff1d(dofs, fixed)

    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))

    # Set load at the specified fractional distance (p.forcedist) from the top-left (default) to the top-right corner.
    f[int(1 + (2 * cfg["forcedist"] * cfg["nelx"]) * (cfg["nely"] + 1)), 0] = -1

    st.update(
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

    return st


def inner_opt(
    x: npt.NDArray,
    st: State,
    dc: npt.NDArray,
    dv: npt.NDArray,
    cfg: dict[str, Any] | None = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Inner optimization loop: Lagrange Multiplier Optimization.

    Args:
        x: (npt.NDArray) The current density field during optimization.
        st: State object with needed vectors/matrices for the optimization.
        dc: (npt.NDArray) The sensitivity field wrt. compliance.
        dv: (npt.NDArray) The sensitivity field wrt. volume fraction.
        cfg (dict): A dictionary with configurations (e.g., boundary conditions) for the optimization.

    Returns:
        Tuple of:
            npt.NDArray: The raw density field
            npt.NDArray: The processed density field (without overhang constraint)
            npt.NDArray: The processed density field (with overhang constraint if applicable)
    """
    cfg = cfg or {}
    # Optimality criteria
    l1, l2, move = (0.0, 1e9, 0.2)
    # reshape to perform vector operations
    xnew = np.zeros(cfg["nelx"] * cfg["nely"])

    while l1 + l2 > 0 and (l2 - l1) / (l1 + l2) > st.min_ratio:
        lmid = 0.5 * (l2 + l1)
        if lmid > 0:
            xnew = np.maximum(
                0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid))))
            )
        else:
            xnew = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, x + move)))

        # Filter design variables
        xPhys = np.asarray(st.H * xnew[np.newaxis].T / st.Hs)[:, 0]
        xPrint, _, _ = overhang_filter(xPhys, cfg)

        if xPrint.sum() > cfg["volfrac"] * cfg["nelx"] * cfg["nely"]:
            l1 = lmid
        else:
            l2 = lmid

        # Ensures this loop does not become stuck due to abs(l2 - l1) converging to near 0
        if abs(l2 - l1) < np.finfo(float).eps:
            break

    return (xnew, xPhys, xPrint)


@overload
def overhang_filter(
    x: npt.NDArray[np.float64], cfg: dict[str, Any] | None = None
) -> tuple[npt.NDArray[np.float64], None, None]: ...


@overload
def overhang_filter(
    x: npt.NDArray[np.float64], cfg: dict[str, Any] | None, dc: npt.NDArray[np.float64], dv: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...


def overhang_filter(
    x: npt.NDArray[np.float64],
    cfg: dict[str, Any] | None = None,
    dc: npt.NDArray[np.float64] | None = None,
    dv: npt.NDArray[np.float64] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None]:
    """Topology Optimization (TO) filter.

    Args:
        x: (npt.NDArray) The current density field during optimization.
        cfg (dict): A dictionary with configurations (e.g., boundary conditions) for the optimization.
        dc: (npt.NDArray) The sensitivity field wrt. compliance.
        dv: (npt.NDArray) The sensitivity field wrt. volume fraction.

    Returns:
        Tuple[npt.NDArray, npt.NDArray, npt.NDArray]: The updated design, sensitivity dc, and sensitivity dv, respectively.
    """
    cfg = cfg or {}
    if cfg["overhang_constraint"]:
        P = 40
        ep = 1e-4
        xi_0 = 0.5
        Ns = 3
        nSens = 2  # dc and dv (hard-coded)

        x = design_to_image(x, cfg["nelx"], cfg["nely"])
        if dc is not None and dv is not None:
            dc = design_to_image(dc, cfg["nelx"], cfg["nely"])
            dv = design_to_image(dv, cfg["nelx"], cfg["nely"])

        Q = P + np.log(Ns) / np.log(xi_0)
        SHIFT = 100 * (np.finfo(float).tiny) ** (1 / P)
        BACKSHIFT = 0.95 * (Ns ** (1 / Q)) * (SHIFT ** (P / Q))
        xi = np.zeros(x.shape)
        Xi = np.zeros(x.shape)
        keep = np.zeros(x.shape)
        sq = np.zeros(x.shape)

        xi[cfg["nely"] - 1, :] = x[cfg["nely"] - 1, :].copy()
        for i in reversed(range(cfg["nely"] - 1)):
            cbr = np.array([0, *list(xi[i + 1, :]), 0]) + SHIFT
            keep[i, :] = cbr[: cfg["nelx"]] ** P + cbr[1 : cfg["nelx"] + 1] ** P + cbr[2:] ** P
            Xi[i, :] = keep[i, :] ** (1 / Q) - BACKSHIFT
            sq[i, :] = np.sqrt((x[i, :] - Xi[i, :]) ** 2 + ep)
            xi[i, :] = 0.5 * ((x[i, :] + Xi[i, :]) - sq[i, :] + np.sqrt(ep))

        if dc is not None and dv is not None:
            dc_copy = dc.copy()
            dv_copy = dv.copy()
            dfxi = [np.array(dc_copy), np.array(dv_copy)]
            dfx = [np.array(dc_copy), np.array(dv_copy)]
            lamb = np.zeros((nSens, cfg["nelx"]))
            for i in range(cfg["nely"] - 1):
                dsmindx = 0.5 * (1 - (x[i, :] - Xi[i, :]) / sq[i, :])
                dsmindXi = 1 - dsmindx
                cbr = np.array([0, *list(xi[i + 1, :]), 0]) + SHIFT

                dmx = np.zeros((Ns, cfg["nelx"]))
                for j in range(Ns):
                    dmx[j, :] = (P / Q) * (keep[i, :] ** (1 / Q - 1)) * (cbr[j : cfg["nelx"] + j] ** (P - 1))

                qi = np.ravel([[i] * 3 for i in range(cfg["nelx"])])
                qj = qi + [-1, 0, 1] * cfg["nelx"]
                qs = np.ravel(dmx.T)

                dsmaxdxi = coo_matrix((qs[1:-1], (qi[1:-1], qj[1:-1]))).tocsc()
                for k in range(nSens):
                    dfx[k][i, :] = dsmindx * (dfxi[k][i, :] + lamb[k, :])
                    lamb[k, :] = ((dfxi[k][i, :] + lamb[k, :]) * dsmindXi) @ dsmaxdxi

            i = cfg["nely"] - 1
            for k in range(nSens):
                dfx[k][i, :] = dfx[k][i, :] + lamb[k, :]

            dc, dv = dfx
            dc, dv = image_to_design(dc), image_to_design(dv)

        xi = image_to_design(xi)

    else:
        xi = x

    return (xi, dc, dv)
