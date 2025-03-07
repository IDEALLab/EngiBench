# ruff: noqa: E741, N806, N815, N816
# Disabled variable name conventions

"""Beams 2D problem."""

from __future__ import annotations

from copy import deepcopy
import dataclasses
from typing import Any

from gymnasium import spaces
import numpy as np
import numpy.typing as npt

from engibench.core import ObjectiveDirection
from engibench.core import OptiStep
from engibench.core import Problem
from engibench.problems.beams2d.backend import calc_sensitivity
from engibench.problems.beams2d.backend import design_to_image
from engibench.problems.beams2d.backend import overhang_filter
from engibench.problems.beams2d.backend import Params
from engibench.problems.beams2d.backend import setup


@dataclasses.dataclass
class ExtendedOptiStep(OptiStep):
    """Extended OptiStep to store a single NumPy array representing a density field at a given optimization step."""

    stored_design: npt.NDArray[np.float64] = dataclasses.field(default_factory=lambda: np.array([], dtype=np.float64))


class Beams2D(Problem[npt.NDArray, npt.NDArray]):
    r"""Beam 2D topology optimization problem.

    ## Problem Description
    This problem simulates bending in an MBB beam, where the beam is symmetric about the central vertical axis and a force is applied at the top-center of the design. Problems are formulated using Density-based Topology Optimization (TO) based on an existing Python [implementation](https://github.com/arjendeetman/TopOpt-MMA-Python).

    ## Design space
    The design space is an array of solid densities in `[0.,1.]` with shape `(5000,)` that can also be represented as a `(100, 50)` image, where `nelx = 100` and `nely = 50`.

    ## Objectives
    The objectives are defined and indexed as follows:
    0. `c`: Compliance to minimize.

    ## Conditions
    The conditions are defined by the following parameters:
    - `nelx`: Width of the domain.
    - `nely`: Height of the domain.
    - `volfrac`: Desired volume fraction (in terms of solid material) for the design.
    - `penal`: Intermediate density penalty term.
    - `rmin`: Minimum feature length of beam members.
    - `ft`: Filtering method; 0 for sensitivity-based and 1 for density-based.
    - `overhang_constraint`: Boolean input condition to decide whether a 45 degree overhang constraint is imposed on the design.

    ## Dataset
    The dataset linked to this problem is hosted on the [Hugging Face Datasets Hub](https://huggingface.co/datasets/IDEALLab/beams_2d).

    ## Simulator
    The objective (compliance) is calculated by the equation `c = ( (Emin+xPrint**penal*(Emax-Emin))*ce ).sum()` where `xPrint` is the current true density field.

    ## References
    If you use this problem in your research, please cite the following paper:
    E. Andreassen, A. Clausen, M. Schevenels, B. S. Lazarov, and O. Sigmund, “Efficient topology optimization in MATLAB using 88 lines of code,” in Structural and Multidisciplinary Optimization, vol. 43, pp. 1-16, 2011.

    ## Lead
    Arthur Drake @arthurdrake1
    """

    version = 0
    objectives: tuple[tuple[str, str]] = (("c", ObjectiveDirection.MINIMIZE),)
    conditions: frozenset[tuple[str, Any]] = frozenset(
        [
            ("nelx", 100),
            ("nely", 50),
            ("volfrac", 0.35),
            ("penal", 3.0),
            ("rmin", 2.0),
            ("ft", 1),
            ("overhang_constraint", False),
        ]
    )
    design_space = spaces.Box(low=0.0, high=1.0, shape=(5000,), dtype=np.float32)
    dataset_id = "IDEALLab/beams_2d_v0"
    _dataset = None
    __p = None
    container_id = None  # type: ignore

    def __init__(self) -> None:
        """Initializes the Beams2D problem."""
        super().__init__()

        self.seed = None

    def simulate(self, design: npt.NDArray, ce: npt.NDArray | None = None, config: dict[str, Any] = {}) -> npt.NDArray:
        """Simulates the performance of a beam design.

        Args:
            design (np.ndarray): The design to simulate.
            ce: (np.ndarray, optional): If applicable, the pre-calculated sensitivity of the current design.
            config (dict): A dictionary with configuration (e.g., boundary conditions) for the simulation.

        Returns:
            npt.NDArray: The performance of the design in terms of compliance.
        """
        if self.__p is None:
            self.__p = Params()
            base_config = {"max_iter": 100}
            base_config.update(self.boundary_conditions)
            base_config.update(config)
            self.__p.update(base_config)
            self.__p = setup(self.__p)

        if ce is None:
            ce = calc_sensitivity(design, self.__p)
        c = (
            (self.__p.Emin + design**self.__p.penal * (self.__p.Emax - self.__p.Emin)) * ce
        ).sum()  # compliance (objective)
        return np.array(c)

    def optimize(self, config: dict[str, Any] = {}) -> tuple[np.ndarray, list[OptiStep]]:
        """Optimizes the design of a beam.

        Args:
            config (dict): A dictionary with configuration (e.g., boundary conditions) for the optimization.

        Returns:
            Tuple[np.ndarray, dict]: The optimized design and its performance.
        """
        # Prepares the optimization script/function with the optimization configuration
        if self.__p is None:
            self.__p = Params()
            base_config = {"max_iter": 100}
            base_config.update(self.boundary_conditions)
            base_config.update(config)
            self.__p.update(base_config)
            self.__p = setup(self.__p)

        # Make sure to include the intermediate designs of size (5000,)
        # Make sure to return the full history of the optimization instead of just the last step
        optisteps_history = []

        dc = np.zeros(self.__p.nely * self.__p.nelx)
        dv = np.zeros(self.__p.nely * self.__p.nelx)
        ce = np.ones(self.__p.nely * self.__p.nelx)

        xPhys = x = self.__p.volfrac * np.ones(self.__p.nely * self.__p.nelx, dtype=float)
        xPrint, _, _ = overhang_filter(xPhys, self.__p, dc, dv)

        loop, change = (0, 1)

        while change > self.__p.min_change and loop < self.__p.max_iter:
            loop = loop + 1

            ce = calc_sensitivity(xPrint, p=self.__p)
            c = self.simulate(xPrint, ce=ce)

            dc = (-self.__p.penal * xPrint ** (self.__p.penal - 1) * (self.__p.Emax - self.__p.Emin)) * ce
            dv = np.ones(self.__p.nely * self.__p.nelx)
            xPrint, dc, dv = overhang_filter(xPhys, self.__p, dc, dv)  # MATLAB implementation

            if self.__p.ft == 0:
                dc = np.asarray((self.__p.H * (x * dc))[np.newaxis].T / self.__p.Hs)[:, 0] / np.maximum(0.001, x)  # type: ignore
            elif self.__p.ft == 1:
                dc = np.asarray(self.__p.H * (dc[np.newaxis].T / self.__p.Hs))[:, 0]
                dv = np.asarray(self.__p.H * (dv[np.newaxis].T / self.__p.Hs))[:, 0]

            # Optimality criteria
            l1, l2, move = (0, 1e9, 0.2)
            # reshape to perform vector operations
            xnew = np.zeros(self.__p.nelx * self.__p.nely)

            while l1 + l2 > 0 and (l2 - l1) / (l1 + l2) > self.__p.min_ratio:
                lmid = 0.5 * (l2 + l1)
                if lmid > 0:
                    xnew = np.maximum(
                        0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid))))
                    )  # type: ignore
                else:
                    xnew = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, x + move)))

                # Filter design variables
                if self.__p.ft == 0:
                    xPhys = xnew
                elif self.__p.ft == 1:
                    xPhys = np.asarray(self.__p.H * xnew[np.newaxis].T / self.__p.Hs)[:, 0]

                xPrint, _, _ = overhang_filter(xPhys, self.__p, dc, dv)

                if xPrint.sum() > self.__p.volfrac * self.__p.nelx * self.__p.nely:
                    l1 = lmid
                else:
                    l2 = lmid

            # Compute the change by the inf. norm
            change = np.linalg.norm(
                xnew.reshape(self.__p.nelx * self.__p.nely, 1) - x.reshape(self.__p.nelx * self.__p.nely, 1), np.inf
            )
            x = deepcopy(xnew)

            # Record the current state in optisteps_history
            current_step = ExtendedOptiStep(obj_values=np.array([c]), step=loop)
            current_step.stored_design = np.array(xPrint)
            optisteps_history.append(current_step)

        return xPrint, optisteps_history

    def reset(self, seed: int | None = None, **kwargs) -> None:
        r"""Reset the simulator and numpy random to a given seed.

        Args:
            seed (int, optional): The seed to reset to. If None, a random seed is used.
            **kwargs: Additional keyword arguments.
        """
        super().reset(seed, **kwargs)
        self.__p = None

    def render(self, design: np.ndarray, nelx: int = 100, nely: int = 50, open_window: bool = False) -> Any:
        """Renders the design in a human-readable format.

        Args:
            design (np.ndarray): The design to render.
            nelx (int): Width of the problem domain.
            nely (int): Height of the problem domain.
            open_window (bool): If True, opens a window with the rendered design.

        Returns:
            Any: The rendered design.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(design_to_image(design, nelx, nely), cmap="coolwarm", ax=ax, vmin=0, vmax=1)

        if open_window:
            plt.show()
        return fig, ax

    def random_design(self) -> tuple[npt.NDArray, int]:
        """Samples a valid random design.

        Returns:
            Tuple of:
                np.ndarray: The valid random design.
                int: The random index selected.
        """
        rnd = self.np_random.integers(low=0, high=len(self._dataset["train"]), dtype=int)  # type: ignore
        return np.array(self._dataset["train"]["xPrint"][rnd]), rnd  # type: ignore


if __name__ == "__main__":
    print("Loading dataset.")
    problem = Beams2D()
    problem.reset()
    dataset = problem.dataset

    # Example of getting the training set
    xPrint_train = dataset["train"]["xPrint"]  # type: ignore
    c_train = dataset["train"]["compliance"]  # type: ignore
    params_train = dataset["train"].remove_columns(["xPrint", "compliance"])  # type: ignore

    # Get design and conditions from the dataset, render design
    design, idx = problem.random_design()
    config = params_train[idx]
    compliance = c_train[idx]
    nelx = config["nelx"]
    nely = config["nely"]
    fig, ax = problem.render(design, nelx=nelx, nely=nely, open_window=True)
    fig.savefig(
        "beam_random.png",
        dpi=300,
    )

    print(f"Verifying compliance via simulation. Reference value: {compliance:.4f}")

    try:
        c_ref = problem.simulate(design, config=config)
        print(f"Calculated compliance: {c_ref:.4f}")
    except ArithmeticError:
        print("Failed to calculate compliance for upscaled design.")

    # Sample Optimization
    print("\nNow conducting a sample optimization with the given configs:", config)

    # NOTE: xPrint and optisteps_history[-1].stored_design are interchangeable.
    xPrint, optisteps_history = problem.optimize(config=config)
    print(f"Final compliance: {optisteps_history[-1].obj_values[0]:.4f}")
    print(
        f"Final design volume fraction: {xPrint.sum() / (nelx*nely):.4f}"  # type: ignore
    )

    fig, ax = problem.render(xPrint, nelx=nelx, nely=nely, open_window=True)  # type: ignore
    fig.savefig(
        "beam_optim.png",
        dpi=300,
    )
