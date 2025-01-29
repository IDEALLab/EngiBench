# ruff: noqa: N806, ERA001, E741, FIX002, PLR0915

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

from engibench.core import DesignType
from engibench.core import OptiStep
from engibench.core import Problem


@dataclasses.dataclass
class ExtendedOptiStep(OptiStep):
    """Extended version of OptiStep to store a list of NumPy arrays."""

    array_list: list[npt.NDArray[np.float64]] = dataclasses.field(default_factory=list)

    # Assumes all incoming arrays are the same shape
    def add_array(self, new_array: npt.NDArray[np.float64]) -> None:
        r"""Add a new array representing a snapshot of xPrint.

        Args:
            new_array (npt.NDArray): The current array, typically shape (N,), representing the intermediate density field xPrint.
        """
        self.array_list.append(new_array)


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
            ("volfrac", 0.35),
            ("penal", 3.0),
            ("rmin", 2.0),
            ("overhang_constraint", True),
        }
    )
    design_space = spaces.Box(low=0.0, high=1.0, shape=(5000,), dtype=np.float32)
    dataset_id = "IDEALLab/beams_2d_v0"
    # container_id = "mdolab/public:u22-gcc-ompi-stable"
    _dataset = None  # type: ignore

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
        self.seed = None

    def __design_to_simulator_input(self, design: npt.NDArray) -> npt.NDArray:
        r"""Convert a design to a simulator input.

        Args:
            design (DesignType): The design to convert.

        Returns:
            SimulatorInputType: The corresponding design as a simulator input.
        """
        if len(design.shape) == 2:
            design = np.swapaxes(design, 0, 1).ravel()
        return design

    def __simulator_output_to_design(self, simulator_output: npt.NDArray, nelx: int = 100, nely: int = 50) -> npt.NDArray:
        r"""Convert a simulator input to a design.

        Args:
            simulator_output (SimulatorInputType): The input to convert.
            nelx: Width of the problem domain.
            nely: Height of the problem domain.

        Returns:
            DesignType: The corresponding design.
        """
        if len(simulator_output.shape) == 1:
            simulator_output = np.swapaxes(simulator_output.reshape(nelx, nely), 0, 1)
        return simulator_output

    def simulate(self, design: npt.NDArray, config: dict[str, Any] = {}) -> tuple[npt.NDArray, npt.NDArray]:
        """Simulates the performance of a beam design.

        Args:
            design (np.ndarray): The design to simulate.
            config (dict): A dictionary with configuration (e.g., boundary conditions, filenames) for the simulation.

        Returns:
            dict: The performance of the design - each entry of the dict corresponds to a named objective value.
        """
        # pre-process the design and run the simulation
        self.__design_to_simulator_input(design)

        cfg = {
            "nelx": 100,
            "nely": 50,
            "volfrac": 0.35,
            "penal": 3,
            "rmin": 2,
            "ft": 1,
            "max_iter": 100,
            "overhang_constraint": False,
        }

        cfg.update(self.boundary_conditions)
        cfg.update(config)
        params = self.setup(cfg)
        cfg.update(params)

        sK = ((self.KE.flatten()[np.newaxis]).T * (self.Emin + (design) ** cfg["penal"] * (self.Emax - self.Emin))).flatten(
            order="F"
        )
        K = coo_matrix((sK, (cfg["iK"], cfg["jK"])), shape=(cfg["ndof"], cfg["ndof"])).tocsc()
        # Remove constrained dofs from matrix and convert to coo
        m = K.shape[0]
        keep = np.delete(np.arange(0, m), cfg["fixed"])
        K = K[keep, :]
        keep = np.delete(np.arange(0, m), cfg["fixed"])
        K = K[:, keep].tocoo()
        # Solve system
        K = cvxopt.spmatrix(K.data, K.row.astype(int), K.col.astype(int))
        B = cvxopt.matrix(cfg["f"][cfg["free"], 0])
        cvxopt.cholmod.linsolve(K, B)
        cfg["u"][cfg["free"], 0] = np.array(B)[:, 0]

        ############################################################################################################
        # Sensitivity
        ce = (
            np.dot(cfg["u"][cfg["edofMat"]].reshape(cfg["nelx"] * cfg["nely"], 8), self.KE)
            * cfg["u"][cfg["edofMat"]].reshape(cfg["nelx"] * cfg["nely"], 8)
        ).sum(1)

        # COMPLIANCE (OBJECTIVE)
        c = ((self.Emin + design ** cfg["penal"] * (self.Emax - self.Emin)) * ce).sum()
        return (np.array(c), np.array(ce))

    def optimize(self, config: dict[str, Any] = {}) -> tuple[np.ndarray, list[OptiStep]]:
        """Optimizes the design of a beam.

        Args:
            config (dict): A dictionary with configuration (e.g., boundary conditions, filenames) for the optimization.

        Returns:
            Tuple[np.ndarray, dict]: The optimized design and its performance.
        """

        # Prepares the optimization script/function with the optimization configuration
        cfg = {
            "nelx": 100,
            "nely": 50,
            "volfrac": 0.35,
            "penal": 3,
            "rmin": 2,
            "ft": 1,
            "max_iter": 100,
            "overhang_constraint": False,
        }

        cfg.update(self.boundary_conditions)
        cfg.update(config)
        params = self.setup(cfg)
        cfg.update(params)

        # Make sure to include the intermediate designs of size (5000,)
        # Make sure to return the full history of the optimization instead of just the last step
        optisteps_history = []

        volfrac = cfg["volfrac"]
        nelx = cfg["nelx"]
        nely = cfg["nely"]
        overhang_constraint = cfg["overhang_constraint"]
        max_iter = cfg["max_iter"]
        penal = cfg["penal"]
        ft = cfg["ft"]
        H = cfg["H"]
        Hs = cfg["Hs"]

        dv = np.zeros(nely * nelx)
        dc = np.zeros(nely * nelx)

        x = volfrac * np.ones(nely * nelx, dtype=float)
        xPhys = x = volfrac * np.ones(nely * nelx, dtype=float)
        xPrint, _, _ = self.base_filter(xPhys, (nelx, nely), dc, dv, overhang_constraint=overhang_constraint)

        loop = 0
        change = 1
        dv = np.ones(nely * nelx)
        dc = np.ones(nely * nelx)
        ce = np.ones(nely * nelx)

        while change > self.min_change and loop < max_iter:  # while change>0.01 and loop<max_iter:
            loop = loop + 1

            c, ce = self.simulate(xPrint, config=cfg)

            dc = (-penal * xPrint ** (penal - 1) * (self.Emax - self.Emin)) * ce
            dv = np.ones(nely * nelx)
            xPrint, dc, dv = self.base_filter(
                xPhys, (nelx, nely), dc, dv, overhang_constraint=overhang_constraint
            )  # MATLAB implementation

            if ft == 0:
                dc = np.asarray((H * (x * dc))[np.newaxis].T / Hs)[:, 0] / np.maximum(0.001, x)  # type: ignore
            elif ft == 1:
                dc = np.asarray(H * (dc[np.newaxis].T / Hs))[:, 0]
                dv = np.asarray(H * (dv[np.newaxis].T / Hs))[:, 0]

            # Optimality criteria
            l1 = 0
            l2 = 1e9
            move = 0.2
            # reshape to perform vector operations
            xnew = np.zeros(nelx * nely)

            while (l2 - l1) / (l1 + l2) > self.min_ratio:
                lmid = 0.5 * (l2 + l1)
                if lmid > 0:
                    xnew = np.maximum(
                        0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid))))
                    )  # type: ignore
                else:
                    xnew = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, x + move)))

                # Filter design variables
                if ft == 0:
                    xPhys = xnew
                elif ft == 1:
                    xPhys = np.asarray(H * xnew[np.newaxis].T / Hs)[:, 0]

                xPrint, _, _ = self.base_filter(xPhys, (nelx, nely), dc, dv, overhang_constraint=overhang_constraint)

                if xPrint.sum() > volfrac * nelx * nely:
                    l1 = lmid
                else:
                    l2 = lmid
                if l1 + l2 == 0:
                    break

            # Compute the change by the inf. norm
            change = np.linalg.norm(xnew.reshape(nelx * nely, 1) - x.reshape(nelx * nely, 1), np.inf)
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
        # rnd = self.np_random.integers(low=0, high=len(self.dataset["train"]["initial"]))  # pyright: ignore[reportOptionalMemberAccess] # type: ignore
        # return np.array(self.dataset["train"]["initial"][rnd])  # type: ignore

    def setup(self, config: dict[str, Any] = {}) -> dict[str, Any]:
        r"""Set up the matrices and parameters for optimization or simulation.

        Args:
            config (dict): A dictionary with configuration (e.g., boundary conditions, filenames) for the optimization.

        Returns:
            params (dict): A dictionary with the relevant matrices and other parameters used in optimization and simulation.
        """
        params = {}
        nelx = config["nelx"]
        nely = config["nely"]
        rmin = config["rmin"]
        ndof = 2 * (nelx + 1) * (nely + 1)
        edofMat = np.zeros((nelx * nely, 8), dtype=int)
        for elx in range(nelx):
            for ely in range(nely):
                el = ely + elx * nely
                n1 = (nely + 1) * elx + ely
                n2 = (nely + 1) * (elx + 1) + ely
                edofMat[el, :] = np.array(
                    [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1]
                )
        # Construct the index pointers for the coo format
        iK = np.kron(edofMat, np.ones((8, 1))).flatten()
        jK = np.kron(edofMat, np.ones((1, 8))).flatten()

        # Filter: Build (and assemble) the index+data vectors for the coo matrix format
        nfilter = int(nelx * nely * ((2 * (np.ceil(rmin) - 1) + 1) ** 2))
        iH = np.zeros(nfilter)
        jH = np.zeros(nfilter)
        sH = np.zeros(nfilter)
        cc = 0
        for i in range(nelx):
            for j in range(nely):
                row = i * nely + j
                kk1 = int(np.maximum(i - (np.ceil(rmin) - 1), 0))
                kk2 = int(np.minimum(i + np.ceil(rmin), nelx))
                ll1 = int(np.maximum(j - (np.ceil(rmin) - 1), 0))
                ll2 = int(np.minimum(j + np.ceil(rmin), nely))
                for k in range(kk1, kk2):
                    for l in range(ll1, ll2):
                        col = k * nely + l
                        fac = rmin - np.sqrt((i - k) * (i - k) + (j - l) * (j - l))
                        iH[cc] = row
                        jH[cc] = col
                        sH[cc] = np.maximum(0.0, fac)
                        cc = cc + 1
        # Finalize assembly and convert to csc format
        H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsc()
        Hs = H.sum(1)

        # BC's and support
        dofs = np.arange(2 * (nelx + 1) * (nely + 1))
        fixed = np.union1d(dofs[0 : 2 * (nely + 1) : 2], np.array([2 * (nelx + 1) * (nely + 1) - 1]))
        free = np.setdiff1d(dofs, fixed)

        # Solution and RHS vectors
        f = np.zeros((ndof, 1))
        u = np.zeros((ndof, 1))

        # Set load
        f[1, 0] = -1

        params["ndof"] = ndof
        params["edofMat"] = edofMat
        params["iK"] = iK
        params["jK"] = jK
        params["H"] = H
        params["Hs"] = Hs
        params["dofs"] = dofs
        params["fixed"] = fixed
        params["free"] = free
        params["f"] = f
        params["u"] = u

        return params

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
        self, x1: npt.NDArray, dims: tuple[int, int], dc: npt.NDArray, dv: npt.NDArray, overhang_constraint: bool = False
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Topology Optimization (TO) filter.

        Args:
            x1: (npt.NDArray) The current density field during optimization.
            dims: (tuple of ints) 1st term nelx is the domain width; 2nd term nely is the domain height.
            dc: (npt.NDArray) The sensitivity field wrt. compliance;
            dv: (npt.NDArray) The sensitivity field wrt. volume fraction.
            overhang_constraint: (bool) Indicates whether the 45-degree overhang constraint is applied.

        Returns:
            Tuple[npt.NDArray, npt.NDArray, npt.NDArray]: The updated design, sensitivity dc, and sensitivty dv, respectively.
        """
        nelx = dims[0]
        nely = dims[1]

        x = self.__simulator_output_to_design(x1, nelx, nely)
        if overhang_constraint:
            if np.sum(dc) != 0:
                dc = self.__simulator_output_to_design(dc, nelx, nely)
                dv = self.__simulator_output_to_design(dv, nelx, nely)
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

            xi[nely - 1, :] = deepcopy(x[nely - 1, :])
            for i in reversed(range(nely - 1)):
                cbr = np.array([0, *list(xi[i + 1, :]), 0]) + SHIFT
                keep[i, :] = cbr[:nelx] ** P + cbr[1 : nelx + 1] ** P + cbr[2:] ** P
                Xi[i, :] = keep[i, :] ** (1 / Q) - BACKSHIFT
                sq[i, :] = np.sqrt((x[i, :] - Xi[i, :]) ** 2 + ep)
                xi[i, :] = 0.5 * ((x[i, :] + Xi[i, :]) - sq[i, :] + np.sqrt(ep))

            if np.sum(dc) != 0:
                dc_copy = deepcopy(dc)
                dv_copy = deepcopy(dv)
                dfxi = [np.array(dc_copy), np.array(dv_copy)]
                dfx = [np.array(dc_copy), np.array(dv_copy)]
                lamb = np.zeros((nSens, nelx))
                for i in range(nely - 1):
                    dsmindx = 0.5 * (1 - (x[i, :] - Xi[i, :]) / sq[i, :])
                    dsmindXi = 1 - dsmindx
                    cbr = np.array([0, *list(xi[i + 1, :]), 0]) + SHIFT

                    dmx = np.zeros((Ns, nelx))
                    for j in range(Ns):
                        dmx[j, :] = (P / Q) * (keep[i, :] ** (1 / Q - 1)) * (cbr[j : nelx + j] ** (P - 1))

                    qi = np.ravel([[i] * 3 for i in range(nelx)])
                    qj = qi + [-1, 0, 1] * nelx
                    qs = np.ravel(dmx.T)

                    dsmaxdxi = coo_matrix((qs[1:-1], (qi[1:-1], qj[1:-1]))).tocsc()
                    for k in range(nSens):
                        dfx[k][i, :] = dsmindx * (dfxi[k][i, :] + lamb[k, :])
                        lamb[k, :] = ((dfxi[k][i, :] + lamb[k, :]) * dsmindXi) @ dsmaxdxi

                i = nely - 1
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


r""" if __name__ == "__main__":
    problem = Beams2D()
    # problem.reset(seed=0, cleanup=False)

    # Get design and conditions from the dataset
    design = problem.random_design()

    # This would just be noise
    fig, ax = problem.render(design, open_window=True)
    fig.savefig(
        "mbb_beam.png",
        dpi=300,
    )

    # TODO: Define this as a subclass of Dataset from core.py and set up the import from HuggingFace. Then uncomment the final lines.
    dataset = problem.dataset
    # config_keys = dataset["train"].features.keys()
    # config = {key: dataset["train"][key][0] for key in config_keys}

    # print(problem.optimize(design, config=config)) 
"""

if __name__ == "__main__":
    problem = Beams2D()
    xPrint, optisteps_history = problem.optimize()
    print("Final compliance:", optisteps_history[-1].obj_values)

