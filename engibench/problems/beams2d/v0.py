# ruff: noqa: E741, N806, N815, N816
# Disabled variable name conventions

"""Beams 2D problem."""

from __future__ import annotations

from copy import deepcopy
import dataclasses
import sys
from typing import Any

from gymnasium import spaces
import numpy as np
import numpy.typing as npt

from engibench.core import ObjectiveDirection
from engibench.core import OptiStep
from engibench.core import Problem
from engibench.problems.beams2d.backend import calc_sensitivity
from engibench.problems.beams2d.backend import design_to_image
from engibench.problems.beams2d.backend import image_to_design
from engibench.problems.beams2d.backend import inner_opt
from engibench.problems.beams2d.backend import overhang_filter
from engibench.problems.beams2d.backend import Params
from engibench.problems.beams2d.backend import setup


@dataclasses.dataclass
class ExtendedOptiStep(OptiStep):
    """Extended OptiStep to store a single NumPy array representing a density field at a given optimization step."""

    design: npt.NDArray[np.float64] = dataclasses.field(default_factory=lambda: np.array([], dtype=np.float64))


class Beams2D(Problem[npt.NDArray, npt.NDArray]):
    r"""Beam 2D topology optimization problem.

    ## Problem Description
    This problem simulates bending in an MBB beam, where the beam is symmetric about the central vertical axis and a force is applied at the top of the design. Problems are formulated using Density-based Topology Optimization (TO) based on an existing Python [implementation](https://github.com/arjendeetman/TopOpt-MMA-Python).

    ## Design space
    The design space is an array of solid densities in `[0.,1.]` with default image shape `(100, 50)`, where `nelx = 100` and `nely = 50`, that is also represented internally as a flattened `(5000,)` array.

    ## Objectives
    The objectives are defined and indexed as follows:
    0. `c`: Compliance to minimize.

    ## Conditions
    The conditions are defined by the following parameters:
    - `nelx`: Width of the domain.
    - `nely`: Height of the domain.
    - `volfrac`: Desired volume fraction (in terms of solid material) for the design.
    - `rmin`: Minimum feature length of beam members.
    - `forcedist`: Fractional distance of the downward force from the top-left (default) to the top-right corner.
    - `overhang_constraint`: Boolean input condition to decide whether a 45 degree overhang constraint is imposed on the design.

    ## Simulator
    The objective (compliance) is calculated by the equation `c = ( (Emin+xPrint**penal*(Emax-Emin))*ce ).sum()` where `xPrint` is the current true density field.

    ## Dataset
    The dataset linked to this problem is hosted on the [Hugging Face Datasets Hub](https://huggingface.co/datasets/IDEALLab/beams_2d).

    ### v0

    #### Fields

    The dataset contains optimal design, conditions, and final objective value (compliance). It also contains the objective value history over the optimization steps of each sample.

    #### Creation Method
    We created this dataset via uniform sampling across the following parameters: design space resolution (represented by nely), solid volume fraction (volfrac), minimum length scale (rmin), and fractional distance of the applied force between the top-left and top-right of the domain (forcedist). In this case, the top-left corresponds to the center of the full beam, since we only optimize and simulate over half of the beam.

    ## References
    If you use this problem in your research, please cite the following paper:
    E. Andreassen, A. Clausen, M. Schevenels, B. S. Lazarov, and O. Sigmund, “Efficient topology optimization in MATLAB using 88 lines of code,” in Structural and Multidisciplinary Optimization, vol. 43, pp. 1-16, 2011.

    ## Lead
    Arthur Drake @arthurdrake1
    """

    version = 0
    objectives: tuple[tuple[str, ObjectiveDirection]] = (("c", ObjectiveDirection.MINIMIZE),)
    conditions: frozenset[tuple[str, Any]] = frozenset(
        [
            ("nelx", 100),
            ("nely", 50),
            ("volfrac", 0.35),
            ("rmin", 2.0),
            ("forcedist", 0.0),
            ("overhang_constraint", False),
        ]
    )
    dataset_id = "IDEALLab/beams_2d_v0"
    _dataset = None
    container_id = None  # type: ignore

    def __init__(self, resolution: tuple[int, int] = (50, 100), penal: float = 3.0, max_iter: int = 100):
        """Initializes the Beams2D problem.

        Args:
            resolution (tuple of [int, int]): The image resolution represented by (nely, nelx).
            penal (float): Intermediate material penalization term (3.0 by default).
            max_iter (int): Maximum optimization iterations, assuming no convergence (100 by default).
        """
        super().__init__()
        self.__p = Params()
        self.resolution = resolution
        self.design_space = spaces.Box(low=0.0, high=1.0, shape=self.resolution, dtype=np.float64)
        self.dataset_id = f"IDEALLab/beams_2d_{self.resolution[0]}_{self.resolution[1]}_v0"
        self.__p.update({"nely": self.resolution[0], "nelx": self.resolution[1], "penal": penal, "max_iter": max_iter})
        self.__p = setup(self.__p)

    def simulate(self, design: npt.NDArray, ce: npt.NDArray | None = None, config: dict[str, Any] = {}) -> npt.NDArray:
        """Simulates the performance of a beam design.

        Args:
            design (np.ndarray): The design to simulate.
            ce: (np.ndarray, optional): If applicable, the pre-calculated sensitivity of the current design.
            config (dict): A dictionary with configuration (e.g., boundary conditions) for the simulation.

        Returns:
            npt.NDArray: The performance of the design in terms of compliance.
        """
        if len(config) > 0:
            self.__p.update(config)
            self.__p = setup(self.__p)

        # This condition is needed to convert user-provided designs (images) to flat arrays. Normally does not apply, i.e., during optimization.
        if len(design.shape) > 1:
            design = image_to_design(design)

        if ce is None:
            ce = calc_sensitivity(design, self.__p)
        c = (
            (self.__p.Emin + design**self.__p.penal * (self.__p.Emax - self.__p.Emin)) * ce
        ).sum()  # compliance (objective)
        return np.array([c])

    def optimize(self, design: npt.NDArray | None = None, config: dict[str, Any] = {}) -> tuple[np.ndarray, list[OptiStep]]:
        """Optimizes the design of a beam.

        Args:
            design (npt.NDArray or None): The design to begin warm-start optimization from (optional).
            config (dict): A dictionary with configuration (e.g., boundary conditions) for the optimization.

        Returns:
            Tuple[np.ndarray, dict]: The optimized design and its performance.
        """
        if len(config) > 0:
            self.__p.update(config)
            self.__p = setup(self.__p)

        # Returns the full history of the optimization instead of just the last step
        optisteps_history = []

        if design is None:
            xPhys = x = self.__p.volfrac * np.ones(self.__p.nely * self.__p.nelx, dtype=float)
            dc = np.zeros(self.__p.nely * self.__p.nelx)
            dv = np.zeros(self.__p.nely * self.__p.nelx)
        else:
            design = image_to_design(design)
            assert design is not None
            xPhys = x = deepcopy(design)
            ce = calc_sensitivity(design, p=self.__p)
            dc = (-self.__p.penal * design ** (self.__p.penal - 1) * (self.__p.Emax - self.__p.Emin)) * ce
            dv = np.ones(self.__p.nely * self.__p.nelx)

        xPrint, _, _ = overhang_filter(xPhys, self.__p, dc, dv)
        loop, change = (0, 1)

        while change > self.__p.min_change and loop < self.__p.max_iter:
            ce = calc_sensitivity(xPrint, p=self.__p)
            c = self.simulate(xPrint, ce=ce)

            # Record the current state in optisteps_history
            current_step = ExtendedOptiStep(obj_values=np.array(c), step=loop)
            current_step.design = np.array(xPrint)
            optisteps_history.append(current_step)

            loop = loop + 1

            dc = (-self.__p.penal * xPrint ** (self.__p.penal - 1) * (self.__p.Emax - self.__p.Emin)) * ce
            dv = np.ones(self.__p.nely * self.__p.nelx)
            xPrint, dc, dv = overhang_filter(xPhys, self.__p, dc, dv)  # MATLAB implementation

            dc = np.asarray(self.__p.H * (dc[np.newaxis].T / self.__p.Hs))[:, 0]
            dv = np.asarray(self.__p.H * (dv[np.newaxis].T / self.__p.Hs))[:, 0]

            xnew, xPhys, xPrint = inner_opt(x, self.__p, dc, dv)

            # Compute the change by the inf. norm
            change = np.linalg.norm(
                xnew.reshape(self.__p.nelx * self.__p.nely, 1) - x.reshape(self.__p.nelx * self.__p.nely, 1), np.inf
            )
            x = deepcopy(xnew)

        return design_to_image(xPrint, self.__p.nelx, self.__p.nely), optisteps_history

    def reset(self, seed: int | None = None, **kwargs) -> None:
        r"""Reset numpy random to a given seed.

        Args:
            seed (int, optional): The seed to reset to. If None, a random seed is used.
            **kwargs: Additional keyword arguments.
        """
        super().reset(seed, **kwargs)

    def render(self, design: np.ndarray, open_window: bool = False) -> Any:
        """Renders the design in a human-readable format.

        Args:
            design (np.ndarray): The design to render.
            open_window (bool): If True, opens a window with the rendered design.

        Returns:
            Any: The rendered design.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(design, cmap="coolwarm", ax=ax, vmin=0, vmax=1)

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
        rnd = self.np_random.integers(low=0, high=len(self.dataset["train"]), dtype=int)  # type: ignore
        return np.array(self.dataset["train"]["optimal_design"][rnd]), rnd  # type: ignore


if __name__ == "__main__":
    p = Params()
    nely, nelx, penal, max_iter = p.nely, p.nelx, p.penal, p.max_iter
    param_list = [nely, nelx, penal, max_iter]
    for i in range(min(len(sys.argv[1:]), len(param_list))):
        param_list[i] = type(param_list[i])(sys.argv[i + 1])
    nely, nelx, penal, max_iter = param_list

    problem = Beams2D(resolution=(nely, nelx), penal=penal, max_iter=max_iter)
    problem.reset(seed=0)

    print(
        f"Loading dataset for nely={nely}, nelx={nelx}.\nOptimization parameters are penal={penal} and max_iter={max_iter}."
    )
    dataset = problem.dataset

    # Example of getting the training set
    optimal_train = dataset["train"]["optimal_design"]  # type: ignore
    c_train = dataset["train"]["c"]  # type: ignore
    params_train = dataset["train"].select_columns(tuple(dict(problem.conditions).keys()))  # type: ignore

    # Get design and conditions from the dataset, render design
    design, idx = problem.random_design()
    config = params_train[idx]
    compliance = c_train[idx]
    fig, ax = problem.render(design, open_window=True)

    print(f"Verifying compliance via simulation. Reference value: {compliance:.4f}")

    try:
        c_ref = problem.simulate(design, config=config)[0]
        print(f"Calculated compliance: {c_ref:.4f}")
    except ArithmeticError:
        print("Failed to calculate compliance for upscaled design.")

    # Sample Optimization
    print("\nNow conducting a sample optimization with the given configs:", config)

    # NOTE: optimal_design and optisteps_history[-1].stored_design are interchangeable.
    optimal_design, optisteps_history = problem.optimize(config=config)
    print(f"Final compliance: {optisteps_history[-1].obj_values[0]:.4f}")
    print(f"Final design volume fraction: {optimal_design.sum() / (np.prod(optimal_design.shape)):.4f}")

    fig, ax = problem.render(optimal_design, open_window=True)
