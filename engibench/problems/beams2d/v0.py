# ruff: noqa: E741, N806, N815, N816

"""Beams 2D problem.

Filename convention is that folder paths do not end with /. For example, /path/to/folder is correct, but /path/to/folder/ is not.
"""

from __future__ import annotations

from copy import deepcopy
import dataclasses
from typing import Any

import cvxopt
import cvxopt.cholmod
from gymnasium import spaces
import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix
from scipy.sparse import csc_array

from engibench.core import DesignType
from engibench.core import OptiStep
from engibench.core import Problem


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
    """

    nelx: int = 100
    nely: int = 50
    volfrac: float = 0.35
    penal: float = 3.0
    rmin: float = 2.0
    ft: int = 1
    max_iter: int = 100
    overhang_constraint: bool = False
    ndof: int = 10302  # ndof = 2 * (p.nelx + 1) * (p.nely + 1)
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


@dataclasses.dataclass
class ExtendedOptiStep(OptiStep):
    """Extended OptiStep to store a list of NumPy arrays, each of these being a density field at a given optimization step."""

    stored_design: list[npt.NDArray[np.float64]] = dataclasses.field(default_factory=list)

    def add_array(self, new_array: npt.NDArray[np.float64]) -> None:
        r"""Add a new array representing a snapshot of xPrint. Assumes all incoming arrays are the same shape.

        Args:
            new_array (npt.NDArray): The current array, typically shape (N,), representing the intermediate density field xPrint.
        """
        self.stored_design.append(new_array)


class Beams2D(Problem):
    r"""Beam 2D topology optimization problem.

    ## Problem Description (docstring)
    This problem simulates bending in an MBB beam, where the beam is symmetric about the central vertical axis and a force is applied at the top-center of the design. Problems are formulated using Density-based Topology Optimization (TO) based on an existing Python [implementation](https://github.com/arjendeetman/TopOpt-MMA-Python).

    ## Design space
    The design space is an array of solid densities in [0,1] with shape (5000,) that can also be represented as a (100,50) image, where nelx=100 and nely=50.

    ## Objectives
    The objectives are defined and indexed as follows:
    0. `c`: Compliance to minimize.

    ## Boundary conditions
    The boundary conditions are defined by the following parameters:
    - `volfrac`: Desired volume fraction (in terms of solid material) for the design.
    - `penal`: Intermediate density penalty term.
    - `rmin`: Minimum feature length of beam members.
    - `overhang_constraint`: Boolean input condition to decide whether a 45 degree overhang constraint is imposed on the design.

    ## Dataset
    The dataset linked to this problem is hosted on the [Hugging Face Datasets Hub](https://huggingface.co/datasets/IDEALLab/beams_2d).

    ## Simulator
    The objective (compliance) is calculated by the equation c = ( (Emin+xPrint**penal*(Emax-Emin))*ce ).sum() where xPrint is the current true density field.

    ## Lead
    Arthur Drake @arthurdrake1
    """

    version = 0
    possible_objectives: tuple[tuple[str, str]] = (("c", "minimize"),)
    boundary_conditions: frozenset[tuple[str, Any]] = frozenset(
        {
            ("nelx", 100),
            ("nely", 50),
            ("volfrac", 0.35),
            ("penal", 3.0),
            ("rmin", 2.0),
            ("ft", 1),
            ("overhang_constraint", False),
        }
    )
    design_space = spaces.Box(low=0.0, high=1.0, shape=(5000,), dtype=np.float32)
    dataset_id = "IDEALLab/beams_2d_v0"
    _dataset = None
    container_id = None

    def __init__(self) -> None:
        """Initializes the Beams2D problem."""
        super().__init__()

        Emin = 1e-9
        Emax = 1
        min_change = 0.025
        min_ratio = 1e-3
        KE = self.lk()

        self.Emin = Emin
        self.Emax = Emax
        self.min_change = min_change
        self.min_ratio = min_ratio
        self.KE = KE

    def __design_to_simulator_input(self, design: npt.NDArray) -> npt.NDArray:
        r"""Convert a design to a simulator input.

        Args:
            design (DesignType): The design to convert.

        Returns:
            SimulatorInputType: The corresponding design as a simulator input.
        """
        return np.swapaxes(design, 0, 1).ravel()

    def __simulator_output_to_design(self, simulator_output: npt.NDArray, nelx: int = 100, nely: int = 50) -> npt.NDArray:
        r"""Convert a simulator input to a design.

        Args:
            simulator_output (SimulatorInputType): The input to convert.
            nelx: Width of the problem domain.
            nely: Height of the problem domain.

        Returns:
            DesignType: The corresponding design.
        """
        return np.swapaxes(simulator_output.reshape(nelx, nely), 0, 1)

    def simulate(self, design: npt.NDArray, p: Params) -> tuple[npt.NDArray, npt.NDArray]:
        """Simulates the performance of a beam design. Assumes the Params object is already set up.

        Args:
            design (np.ndarray): The design to simulate.
            p: Params object with configs (e.g., boundary conditions) and needed vectors/matrices for the simulation.

        Returns:
            dict: The performance of the design - each entry of the dict corresponds to a named objective value.
        """
        sK = ((self.KE.flatten()[np.newaxis]).T * (self.Emin + (design) ** p.penal * (self.Emax - self.Emin))).flatten(
            order="F"
        )
        K = coo_matrix((sK, (p.iK, p.jK)), shape=(p.ndof, p.ndof)).tocsc()
        # Remove constrained dofs from matrix and convert to coo
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
        # TODO: Split into a separate function and only return c in this main simulate function
        ce = (np.dot(p.u[p.edofMat].reshape(p.nelx * p.nely, 8), self.KE) * p.u[p.edofMat].reshape(p.nelx * p.nely, 8)).sum(
            1
        )

        c = ((self.Emin + design**p.penal * (self.Emax - self.Emin)) * ce).sum()  # compliance (objective)
        return (np.array(c), np.array(ce))

    def optimize(self, p: Params) -> tuple[np.ndarray, list[OptiStep]]:
        """Optimizes the design of a beam.

        Args:
            p: Params object with configs (e.g., boundary conditions) and needed vectors/matrices for the optimization.

        Returns:
            Tuple[np.ndarray, dict]: The optimized design and its performance.
        """
        # Prepares the optimization script/function with the optimization configuration
        # IMPORTANT: FIRST update the values with the boundary conditions, THEN perform setup of remaining matrices, etc.
        p.update(dict(self.boundary_conditions))
        p = self.setup(p)

        # Make sure to include the intermediate designs of size (5000,)
        # Make sure to return the full history of the optimization instead of just the last step
        optisteps_history = []

        dv = np.zeros(p.nely * p.nelx)
        dc = np.zeros(p.nely * p.nelx)
        ce = np.ones(p.nely * p.nelx)

        x = p.volfrac * np.ones(p.nely * p.nelx, dtype=float)
        xPhys = x = p.volfrac * np.ones(p.nely * p.nelx, dtype=float)
        xPrint, _, _ = self.base_filter(xPhys, p, dc, dv)

        loop = 0
        change = 1

        while change > self.min_change and loop < p.max_iter:  # while change>0.01 and loop<max_iter:
            loop = loop + 1

            c, ce = self.simulate(xPrint, p=p)

            dc = (-p.penal * xPrint ** (p.penal - 1) * (self.Emax - self.Emin)) * ce
            dv = np.ones(p.nely * p.nelx)
            xPrint, dc, dv = self.base_filter(xPhys, p, dc, dv)  # MATLAB implementation

            if p.ft == 0:
                dc = np.asarray((p.H * (x * dc))[np.newaxis].T / p.Hs)[:, 0] / np.maximum(0.001, x)  # type: ignore
            elif p.ft == 1:
                dc = np.asarray(p.H * (dc[np.newaxis].T / p.Hs))[:, 0]
                dv = np.asarray(p.H * (dv[np.newaxis].T / p.Hs))[:, 0]

            # Optimality criteria
            l1 = 0
            l2 = 1e9
            move = 0.2
            # reshape to perform vector operations
            xnew = np.zeros(p.nelx * p.nely)

            while (l2 - l1) / (l1 + l2) > self.min_ratio:
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

                xPrint, _, _ = self.base_filter(xPhys, p, dc, dv)

                if xPrint.sum() > p.volfrac * p.nelx * p.nely:
                    l1 = lmid
                else:
                    l2 = lmid
                if l1 + l2 == 0:
                    break

            # Compute the change by the inf. norm
            change = np.linalg.norm(xnew.reshape(p.nelx * p.nely, 1) - x.reshape(p.nelx * p.nely, 1), np.inf)
            x = deepcopy(xnew)
            x = np.array(x)

            # Record the current state in optisteps_history
            current_step = ExtendedOptiStep(obj_values=np.array([c]), step=loop)
            current_step.add_array(xPrint)
            optisteps_history.append(current_step)

        return (xPrint, optisteps_history)

    def render(self, design: np.ndarray, open_window: bool = False) -> Any:
        """Renders the design in a human-readable format.

        Args:
            design (np.ndarray): The design to render.
            open_window (bool): If True, opens a window with the rendered design.

        Returns:
            Any: The rendered design.
        """
        import matplotlib.pyplot as plt

        # TODO: Change this to reshape input and plot as heatmap
        fig, ax = plt.subplots()

        ax.scatter(design[0], design[1], s=10, alpha=0.7)
        if open_window:
            plt.show()
        return fig, ax

    def random_design(self) -> DesignType:
        # Make this an actual random value
        """Samples a valid random design.

        Returns:
            DesignType: The valid random design.
        """
        return self.design_space.sample()

    def setup(self, p: Params) -> Params:
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
        # Construct the index pointers for the coo format
        iK = np.kron(edofMat, np.ones((8, 1))).flatten()
        jK = np.kron(edofMat, np.ones((1, 8))).flatten()

        # Filter: Build (and assemble) the index+data vectors for the coo matrix format
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

    def lk(self) -> npt.NDArray:
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

    def base_filter(
        self, x1: npt.NDArray, p: Params, dc: npt.NDArray, dv: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Topology Optimization (TO) filter.

        Args:
            x1: (npt.NDArray) The current density field during optimization.
            p: Params object with configs (e.g., boundary conditions) and needed vectors/matrices for the optimization.
            dc: (npt.NDArray) The sensitivity field wrt. compliance.
            dv: (npt.NDArray) The sensitivity field wrt. volume fraction.

        Returns:
            Tuple[npt.NDArray, npt.NDArray, npt.NDArray]: The updated design, sensitivity dc, and sensitivty dv, respectively.
        """
        x = self.__simulator_output_to_design(x1, p.nelx, p.nely)
        if p.overhang_constraint:
            if np.sum(dc) != 0:
                dc = self.__simulator_output_to_design(dc, p.nelx, p.nely)
                dv = self.__simulator_output_to_design(dv, p.nelx, p.nely)
            P = 40
            ep = 1e-4
            xi_0 = 0.5
            Ns = 3
            nSens = 2  # dc and dv (hard-coded)

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
                dc = self.__design_to_simulator_input(dc)
                dv = self.__design_to_simulator_input(dv)

            xi = self.__design_to_simulator_input(xi)
        else:
            xi = x1

        return (xi, dc, dv)


if __name__ == "__main__":
    # Calling Params() initiates an object with only the base configs (see top)
    # These should be updated with the desired boundary conditions...
    # Followed by the setup() function to properly initialize other params
    print("Loading dataset.")
    init_params = Params()
    problem = Beams2D()
    dataset = problem.dataset  # NOTE: Use xPrint.reshape(nely, nelx) i.e., xPrint.reshape(100, 200) to obtain images.

    # Get design and conditions from the dataset
    design = problem.random_design()
    # TODO: Render here

    # Sample Optimization
    print("Conducting a sample optimization with default configs.")
    xPrint, optisteps_history = problem.optimize(init_params)
    print(f"Final compliance: {optisteps_history[-1].obj_values[0]:.4f}")
    print(
        # TODO: Convert stored_design to a single object rather than list
        f"Final design volume fraction: {optisteps_history[-1].stored_design[0].sum() / (init_params.nelx * init_params.nely):.4f}"  # type: ignore
    )

# [DONE] Test dataset loading.
# [DONE] Clean up base filter and optimization with "p" object instead of passing individual variables: nelx --> p.nelx
# [DONE] Check if this passes the precommits
# [DONE] Commit new to repo
