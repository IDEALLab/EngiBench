# ruff: noqa: N806

"""Beams 2D problem.

Filename convention is that folder paths do not end with /. For example, /path/to/folder is correct, but /path/to/folder/ is not.
"""

from __future__ import annotations

import os
from typing import Any

# Problem-specific
import cvxopt
import cvxopt.cholmod
from gymnasium import spaces
import numpy as np
import numpy.typing as npt
import pyoptsparse
from scipy.sparse import coo_matrix

from engibench.core import DesignType
from engibench.core import OptiStep
from engibench.core import Problem
from engibench.utils.files import replace_template_values

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
            ("volfrac", 0.15),
            ("penal", 3.0),
            ("rmin", 2.0),
            ("overhang_constraint", True),
        }
    )
    design_space = spaces.Box(low=0.0, high=1.0, shape=(5000,), dtype=np.float32)
    dataset_id = "IDEALLab/beams_2d_v0"
    # container_id = "mdolab/public:u22-gcc-ompi-stable"
    _dataset = None  # type: ignore

    def __init__(self, base_directory: str | None = None) -> None:
        """Initializes the Beams2D problem.

        Args:
            base_directory (str, optional): The base directory for the problem. If None, the current directory is selected.
        """
        super().__init__()

        Emin = 1e-9
        Emax = 1
        KE = self.lk()

        self.Emin = Emin
        self.Emax = Emax
        self.KE = KE
        self.seed = None
        self.current_study = f"study_{self.seed}"
        # This is used for intermediate files
        # Local file are prefixed with self.local_base_directory
        if base_directory is not None:
            self.__local_base_directory = base_directory
        else:
            self.__local_base_directory = os.getcwd()
        self.__local_target_dir = self.__local_base_directory + "/engibench_studies/problems/beams2d"
        # self.__local_template_dir = (
        #     os.path.dirname(os.path.abspath(__file__)) + "/templates"
        # )  # These templates are shipped with the lib
        # self.__local_scripts_dir = os.path.dirname(os.path.abspath(__file__)) + "/scripts"
        self.__local_study_dir = self.__local_target_dir + "/" + self.current_study

        # Docker target directory
        # This is used for files that are mounted into the docker container
        # self.__docker_base_dir = "/home/mdolabuser/mount/engibench"
        # self.__docker_target_dir = self.__docker_base_dir + "/engibench_studies/problems/beams2d"
        # self.__docker_study_dir = self.__docker_target_dir + "/" + self.current_study

    def __design_to_simulator_input(self, design: npt.NDArray) -> npt.NDArray:
        r"""Convert a design to a simulator input.

        Args:
            design (DesignType): The design to convert.
            **kwargs: Additional keyword arguments.

        Returns:
            SimulatorInputType: The corresponding design as a simulator input.
        """
        return np.swapaxes(design, 0, 1).ravel()

    def __simulator_output_to_design(self, simulator_output: npt.NDArray, nelx: int = 100, nely: int = 50) -> npt.NDArray:
        r"""Convert a simulator input to a design.

        Args:
            simulator_output (SimulatorInputType): The input to convert.
            **kwargs: Additional keyword arguments.

        Returns:
            DesignType: The corresponding design.
        """
        return np.swapaxes(simulator_output.reshape(nelx, nely), 0, 1)

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
            "volfrac": 0.15,
            "penal": 3,
            "rmin": 2,
            "ft": 1,
            "max_iter": 100,
            "overhang_constraint": False,
            "display": False,
        }

        cfg.update(self.boundary_conditions)
        cfg.update(config)
        params = self.setup(cfg)
        cfg.update(params)

        # replace_template_values(
        #     self.__local_study_dir + "/beam_analysis.py",
        #     base_config,
        # )

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

    def optimize(self, starting_point: npt.NDArray, config: dict[str, Any] = {}) -> tuple[np.ndarray, list[OptiStep]]:
        """Optimizes the design of a beam.

        Args:
            starting_point (np.ndarray): The starting point for the optimization.
            config (dict): A dictionary with configuration (e.g., boundary conditions, filenames) for the optimization.

        Returns:
            Tuple[np.ndarray, dict]: The optimized design and its performance.
        """
        # pre-process the design and run the simulation
        filename = "candidate_design"
        self.__design_to_simulator_input(starting_point)  # self.__design_to_simulator_input(starting_point, filename)

        # Prepares the optimization script/function with the optimization configuration
        cfg = {
            "nelx": 100,
            "nely": 50,
            "volfrac": 0.15,
            "penal": 3,
            "rmin": 2,
            "ft": 1,
            "max_iter": 100,
            "overhang_constraint": False,
            "display": False,
        }

        cfg.update(self.boundary_conditions)
        cfg.update(config)
        params = self.setup(cfg)
        cfg.update(params)

        # Note: the variable names base_config and cfg are interchangeable, cfg is just the shorter form.
        replace_template_values(
            self.__local_study_dir + "/beam_opt.py",
            cfg,
        )

        # post process -- extract the shape and objective values
        # TODO: subclass the current optistep class and include intermediate designs of size (5000,)
        optisteps_history = []
        history = pyoptsparse.History(self.__local_study_dir + "/output/opt.hst")

        # TODO return the full history of the optimization instead of just the last step
        # Also, this is inconsistent with the definition of the problem saying we optimize 2 objectives...
        objective = history.getValues(names=["obj"], callCounters=None, allowSens=False, major=False, scale=True)["obj"][
            -1, -1
        ]
        optisteps_history.append(OptiStep(obj_values=np.array([objective]), step=0))
        history.close()

        volfrac = cfg["volfrac"]
        nelx = cfg["nelx"]
        nely = cfg["nely"]
        overhang_constraint = cfg["overhang_constraint"]
        max_iter = cfg["max_iter"]
        penal = cfg["penal"]
        ft = cfg["ft"]
        H = cfg["H"]
        Hs = cfg["Hs"]

        x = volfrac * np.ones(nely * nelx, dtype=float)
        xPhys = x = volfrac * np.ones(nely * nelx, dtype=float)
        xPrint, _, _, _ = self.base_filter(xPhys, nelx, nely, None, None, overhang_constraint=overhang_constraint)

        loop = 0
        change = 1
        dv = np.ones(nely * nelx)
        dc = np.ones(nely * nelx)
        ce = np.ones(nely * nelx)

        while change > 0.025 and loop < max_iter:  # while change>0.01 and loop<max_iter:
            loop = loop + 1

            c, ce = self.simulate(xPrint, config=cfg)

            dc = (-penal * xPrint ** (penal - 1) * (self.Emax - self.Emin)) * ce
            dv = np.ones(nely * nelx)
            xPrint, dx_m, dc, dv = self.base_filter(xPhys, nelx, nely, dc, dv, overhang_constraint)  # MATLAB implementation

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

            while (l2 - l1) / (l1 + l2) > 1e-3:
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

                xPrint, _, _, _ = self.base_filter(
                    xPhys, nelx, nely, None, None, overhang_constraint=cfg["overhang_constraint"]
                )

                if xPrint.sum() > volfrac * nelx * nely:
                    l1 = lmid
                else:
                    l2 = lmid
                if l1 + l2 == 0:
                    break

            # Compute the change by the inf. norm
            change = np.linalg.norm(xnew.reshape(nelx * nely, 1) - x.reshape(nelx * nely, 1), np.inf)
            x = [item for item in xnew]
            x = np.array(x)

        return (xPrint, optisteps_history)

    def render(self, design: np.ndarray, open_window: bool = False) -> Any:  # noqa: ANN401
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

    def base_filter(self, x1, nelx, nely, dc, dv, overhang_constraint=False):
        x = self.__simulator_output_to_design(x1, nelx, nely)
        if overhang_constraint:
            if dc and dv:
                dc = self.__simulator_output_to_design(dc, nelx, nely)
                dv = self.__simulator_output_to_design(dv, nelx, nely)
            else:
                dx_m = np.ones(x.shape)
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

            xi[nely - 1, :] = [item for item in x[nely - 1, :]]
            for i in reversed(range(nely - 1)):
                cbr = np.array([0] + list(xi[i + 1, :]) + [0]) + SHIFT
                keep[i, :] = cbr[:nelx] ** P + cbr[1 : nelx + 1] ** P + cbr[2:] ** P
                Xi[i, :] = keep[i, :] ** (1 / Q) - BACKSHIFT
                sq[i, :] = np.sqrt((x[i, :] - Xi[i, :]) ** 2 + ep)
                xi[i, :] = 0.5 * ((x[i, :] + Xi[i, :]) - sq[i, :] + np.sqrt(ep))

            if dc is not None and dv is not None:
                varargin = [dc, dv]
                dc_copy = [[x for x in y] for y in dc]
                dv_copy = [[x for x in y] for y in dv]
                dfxi = [np.array(dc_copy), np.array(dv_copy)]
                dfx = [np.array(dc_copy), np.array(dv_copy)]
                lamb = np.zeros((nSens, nelx))
                for i in range(nely - 1):
                    dsmindx = 0.5 * (1 - (x[i, :] - Xi[i, :]) / sq[i, :])
                    dsmindXi = 1 - dsmindx
                    cbr = np.array([0] + list(xi[i + 1, :]) + [0]) + SHIFT

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
                dx_m = np.nan_to_num(dc / varargin[0], 1)  # type: ignore
                dc = self.__design_to_simulator_input(dc)
                dv = self.__design_to_simulator_input(dv)

            dx_m = np.expand_dims(dx_m, axis=(0, 1))
            xi = self.__design_to_simulator_input(xi)
        else:
            dx_m = np.expand_dims(np.ones(x.shape), axis=(0, 1))
            xi = x1

        return xi, dx_m, dc, dv


if __name__ == "__main__":
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
