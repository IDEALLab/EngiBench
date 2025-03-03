"""Thermo Elastic 2D Problem.

Filename convention is that folder paths do not end with /. For example, /path/to/folder is correct, but /path/to/folder/ is not.
"""
# ruff: noqa: ARG002, ERA001

from __future__ import annotations

import ast
from typing import Any

from gymnasium import spaces
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from engibench.core import DesignType
from engibench.core import OptiStep
from engibench.core import Problem
from engibench.problems.thermoelastic2d.model.PythonModel import PythonModel
from engibench.problems.thermoelastic2d.utils import get_res_bounds

# -- Problem specific imports


def build(**kwargs) -> ThermoElastic2D:
    """Builds an ThermoElastic2D problem.

    Args:
        **kwargs: Arguments to pass to the constructor.
    """
    return ThermoElastic2D(**kwargs)


class ThermoElastic2D(Problem[npt.NDArray, npt.NDArray]):
    r"""Truss 2D integer optimization problem.

    ## Problem Description
    This is 2D topology optimization problem for minimizing weakly coupled thermo-elastic compliance subject to boundary conditions and a volume fraction constraint.

    ## Design space
    The design space is represented by a 2D tensor of continuous design variables in the range [0, 1] that represent the material density at each voxel in the design space.

    ## Objectives
    The objectives are defined and indexed as follows:
    0. `sc`: Structural compliance to minimize.
    1. `tc`: Thermal compliance to minimize.
    2. `vf`: Volume fraction error to minimize.

    ## Boundary Conditions



    ## Boundary Conditions
    Creating a problem formulation requires defining a python dict with the following info:
    - `fixed_elements`: Encodes the indices of the structurally fixed elements in the domain.
    - `force_elements_x`: Encodes which elements have a structural load in the x-direction.
    - `force_elements_x`: Encodes which elements have a structural load in the y-direction.
    - `heatsink_elements`: Encodes which elements have a heat sink.
    - `volfrac`: Encodes the target volume fraction for the volume fraction constraint.
    - `rmin`: Encodes the filter size used in the optimization routine.
    - `weight`: Allows one to control which objective is optimized for. 1.0 Is pure structural optimization, while 0.0 is pure thermal optimization.


    ## Dataset
    The dataset linked to this problem is on huggingface [Hugging Face Datasets Hub](https://huggingface.co/datasets/IDEALLab/thermoelastic_2d_v0).


    ## Simulator
    The evaluation code models the problem as a weakly coupled thermo-elastic problem. The code is written in pure python, and the evaluation is done in a single process.

    ## Lead
    Gabriel Apaza @gapaza
    """

    input_space = str
    possible_objectives: tuple[tuple[str, str]] = (("sc", "minimize"), ("tc", "minimize"), ("vf", "minimize"))
    nelx = 64
    nely = 64
    lci, tri, rci, bri = get_res_bounds(nelx + 1, nely + 1)
    boundary_conditions: frozenset[tuple[str, Any]] = frozenset(
        {
            ("nelx", nelx),
            ("nely", nely),
            ("fixed_elements", (lci[21], lci[32], lci[43])),
            ("force_elements_x", (bri[31])),
            ("force_elements_y", (bri[31])),
            ("heatsink_elements", (lci[31], lci[32], lci[33])),
            ("volfrac", 0.2),
            ("rmin", 1.1),
            ("weight", 1.0),  # 1.0 for pure structural, 0.0 for pure thermal
        }
    )
    design_space = spaces.Box(low=0.0, high=1.0, shape=(nelx, nely), dtype=np.float32)
    dataset_id = "IDEALLab/thermoelastic_2d_v0"
    container_id = ""
    _dataset = None

    def __init__(self, base_directory: str | None = None) -> None:
        """Initializes the Airfoil2D problem.

        Args:
            base_directory (str, optional): The base directory for the problem. If None, the current directory is selected.
        """
        super().__init__()
        self.seed = None
        self.current_study = f"study_{self.seed}"

    def reset(self, seed: int | None = None, *, cleanup: bool = False) -> None:
        """Resets the simulator and numpy random to a given seed.

        Args:
            seed (int, optional): The seed to reset to. If None, a random seed is used.
            cleanup (bool): Deletes the previous study directory if True.
        """
        super().reset(seed)
        self.current_study = f"study_{self.seed}"

    def simulate(self, design: npt.NDArray, config: dict[str, Any] = {}, mpicores: int = 4) -> npt.NDArray:
        """Simulates the performance of an airfoil design.

        Args:
            design (np.ndarray): The design to simulate.
            config (dict): A dictionary with configuration (e.g., boundary conditions, filenames) for the simulation.
            mpicores (int): The number of MPI cores to use in the simulation.

        Returns:
            dict: The performance of the design - each entry of the dict corresponds to a named objective value.
        """
        boundary_dict = dict(self.boundary_conditions)
        design = np.array(design)
        results = PythonModel(plot=False, eval_only=True).run(boundary_dict, x_init=design)
        objectives = np.array([results["sc"], results["tc"], results["vf"]])
        return objectives

    def optimize(
        self, starting_point: npt.NDArray, config: dict[str, Any] = {}, mpicores: int = 4
    ) -> tuple[np.ndarray, list[OptiStep]]:
        """Optimizes a topology for the current problem.

        Args:
            starting_point (np.ndarray): The starting point for the optimization.
            config (dict): A dictionary with configuration (e.g., boundary conditions, filenames) for the optimization.
            mpicores (int): The number of MPI cores to use in the optimization.

        Returns:
            Tuple[np.ndarray, dict]: The optimized design and its performance.
        """
        boundary_dict = dict(self.boundary_conditions)
        results = PythonModel(plot=True, eval_only=False).run(boundary_dict)
        design = np.array(results["design"])
        objectives = {"sc": results["sc"], "tc": results["tc"], "vf": results["vf"]}
        return design, objectives

    def render(self, design: np.ndarray, open_window: bool = False) -> Any:
        """Renders the design in a human-readable format.

        Args:
            design (np.ndarray): The design to render.
            open_window (bool): If True, opens a window with the rendered design.

        Returns:
            Any: The rendered design.
        """
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.imshow(-design, cmap="gray", interpolation="none", norm=colors.Normalize(vmin=-1, vmax=0))
        ax.axis("off")
        plt.tight_layout()
        if open_window is True:
            plt.show()

        return fig

    def random_design(self) -> DesignType:
        """Samples a valid random design.

        Returns:
            DesignType: The valid random design.
        """
        design = np.random.rand(self.nelx, self.nely)
        design = np.clip(design, 1e-3, 1.0)
        return design


if __name__ == "__main__":
    # --- Create a new problem
    problem = ThermoElastic2D()
    problem.reset()

    # --- Load the problem dataset
    dataset = problem.dataset
    first_item = dataset["train"][0]
    first_item_design = np.array(ast.literal_eval(first_item["design"]))
    problem.render(first_item_design, open_window=True)

    # # --- Render the design
    # design = problem.random_design()
    # problem.render(design, open_window=True)

    # # --- Optimize a design ---
    # design = problem.random_design()
    # design, objectives = problem.optimize(design)
    # problem.render(design, open_window=True)

    # # --- Evaluate a design ---
    # design = problem.random_design()
    # objectives = problem.simulate(design)
    # print(objectives)
