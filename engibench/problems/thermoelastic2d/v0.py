"""Thermo Elastic 2D Problem."""

import dataclasses
from dataclasses import dataclass
from dataclasses import field
from typing import Annotated, Any, ClassVar

from gymnasium import spaces
from matplotlib import colors
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from engibench.constraint import bounded
from engibench.constraint import constraint
from engibench.constraint import IMPL
from engibench.constraint import THEORY
from engibench.core import ObjectiveDirection
from engibench.core import OptiStep
from engibench.core import Problem
from engibench.problems.thermoelastic2d.model import fea_model
from engibench.problems.thermoelastic2d.model.fea_model import FeaModel
from engibench.problems.thermoelastic2d.utils import get_res_bounds
from engibench.problems.thermoelastic2d.utils import indices_to_binary_matrix

NELY = NELX = 64
LCI, TRI, RCI, BRI = get_res_bounds(NELX + 1, NELY + 1)
FIXED_ELEMENTS = indices_to_binary_matrix([LCI[21], LCI[32], LCI[43]], NELX + 1, NELY + 1)
FORCE_ELEMENTS_X = indices_to_binary_matrix([BRI[31]], NELX + 1, NELY + 1)
FORCE_ELEMENTS_Y = indices_to_binary_matrix([BRI[31]], NELX + 1, NELY + 1)
HEATSINK_ELEMENTS = indices_to_binary_matrix([LCI[31], LCI[32], LCI[33]], NELX + 1, NELY + 1)


class ThermoElastic2D(Problem[npt.NDArray]):
    r"""Truss 2D integer optimization problem.

    This is 2D topology optimization problem for minimizing weakly coupled thermo-elastic compliance subject to boundary conditions and a volume fraction constraint.
    """

    version = 0
    objectives: tuple[tuple[str, ObjectiveDirection], ...] = (
        ("structural_compliance", ObjectiveDirection.MINIMIZE),
        ("thermal_compliance", ObjectiveDirection.MINIMIZE),
        ("volume_fraction", ObjectiveDirection.MINIMIZE),
    )

    @dataclass
    class Conditions:
        """Conditions."""

        fixed_elements: Annotated[npt.NDArray[np.int64], bounded(lower=0.0, upper=1.0).category(THEORY)] = field(
            default_factory=lambda: FIXED_ELEMENTS
        )
        """Binary NxN matrix of the structurally fixed elements in the domain"""
        force_elements_x: Annotated[npt.NDArray[np.int64], bounded(lower=0.0, upper=1.0).category(THEORY)] = field(
            default_factory=lambda: FORCE_ELEMENTS_X
        )
        """Binary NxN matrix specifying elements that have a structural load in the x-direction"""
        force_elements_y: Annotated[npt.NDArray[np.int64], bounded(lower=0.0, upper=1.0).category(THEORY)] = field(
            default_factory=lambda: FORCE_ELEMENTS_Y
        )
        """Binary NxN matrix specifying elements that have a structural load in the y-direction"""
        heatsink_elements: Annotated[npt.NDArray[np.int64], bounded(lower=0.0, upper=1.0).category(THEORY)] = field(
            default_factory=lambda: HEATSINK_ELEMENTS
        )
        """Binary NxN matrix specifying elements that have a heat sink"""
        volfrac: Annotated[float, bounded(lower=0.0, upper=1.0).category(THEORY)] = 0.3
        """Target volume fraction for the volume fraction constraint"""
        rmin: Annotated[
            float, bounded(lower=1.0).category(THEORY), bounded(lower=0.0, upper=3.0).warning().category(IMPL)
        ] = 1.1
        """Filter size used in the optimization routine"""
        weight: Annotated[float, bounded(lower=0.0, upper=1.0).category(THEORY)] = 0.5
        """Control which objective is optimized for. 1.0 is pure structural optimization, while 0.0 is pure thermal optimization"""

    conditions = Conditions()
    design_space = spaces.Box(low=0.0, high=1.0, shape=(NELX, NELY), dtype=np.float32)
    dataset_id = "IDEALLab/thermoelastic_2d_v0"
    container_id = None

    @dataclass
    class Config(Conditions):
        """Structured representation of configuration parameters for a numerical computation."""

        nelx: ClassVar[Annotated[int, bounded(lower=1).category(THEORY)]] = NELX
        nely: ClassVar[Annotated[int, bounded(lower=1).category(THEORY)]] = NELX
        max_iter: int = fea_model.MAX_ITERATIONS
        """Maximal number of iterations for optimize."""

        @constraint
        @staticmethod
        def rmin_bound(rmin: float, nelx: int, nely: int) -> None:
            """Constraint for rmin ∈ (0.0, max{ nelx, nely }]."""
            assert 0.0 < rmin <= max(nelx, nely), f"Params.rmin: {rmin} ∉ (0, max(nelx, nely)]"

    def __init__(self, seed: int = 0) -> None:
        """Initializes the thermoelastic2D problem.

        Args:
            seed (int): The random seed for the problem.
        """
        super().__init__(seed=seed)

    def reset(self, seed: int | None = None) -> None:
        """Resets the simulator and numpy random to a given seed.

        Args:
            seed (int, optional): The seed to reset to. If None, a random seed is used.
        """
        super().reset(seed)

    def simulate(self, design: npt.NDArray, config: dict[str, Any] | None = None) -> npt.NDArray:
        """Simulates the performance of a design topology.

        Args:
            design (np.ndarray): The design to simulate.
            config (dict): A dictionary with configuration (e.g., boundary conditions, filenames) for the simulation.

        Returns:
            dict: The performance of the design - each entry of the dict corresponds to a named objective value.
        """
        boundary_dict = dataclasses.asdict(self.conditions)
        for key, value in (config or {}).items():
            if key in boundary_dict:
                if isinstance(value, list):
                    boundary_dict[key] = np.array(value)
                else:
                    boundary_dict[key] = value

        results = FeaModel(plot=False, eval_only=True).run(boundary_dict, x_init=design)
        return np.array([results["structural_compliance"], results["thermal_compliance"], results["volume_fraction"]])

    def optimize(
        self, starting_point: npt.NDArray, config: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, list[OptiStep]]:
        """Optimizes a topology for the current problem. Note that an appropriate starting_point for the optimization is defined by a uniform material distribution equal to the volume fraction constraint.

        Args:
            starting_point (np.ndarray): The starting point for the optimization.
            config (dict): A dictionary with configuration (e.g., boundary conditions, filenames) for the optimization.

        Returns:
            Tuple[np.ndarray, dict]: The optimized design and its performance.
        """
        boundary_dict = dataclasses.asdict(self.conditions)
        boundary_dict.update({k: v for k, v in (config or {}).items() if k in boundary_dict})
        max_iter = (config or {}).get("max_iter", self.Config.max_iter)
        results = FeaModel(plot=False, eval_only=False, max_iter=max_iter).run(boundary_dict, x_init=starting_point)
        design = np.array(results["design"]).astype(np.float32)
        opti_steps = results["opti_steps"]
        return design, opti_steps

    def render(self, design: np.ndarray, *, open_window: bool = False) -> Figure:
        """Renders the design in a human-readable format.

        Args:
            design (np.ndarray): The design to render.
            open_window (bool): If True, opens a window with the rendered design.

        Returns:
            Figure: The rendered design.
        """
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.imshow(-design, cmap="gray", interpolation="none", norm=colors.Normalize(vmin=-1, vmax=0))
        ax.axis("off")
        plt.tight_layout()
        if open_window is True:
            plt.show()

        return fig

    def random_design(self, dataset_split: str = "train", design_key: str = "optimal_design") -> tuple[npt.NDArray, int]:
        """Samples a valid random design.

        Args:
            dataset_split (str): The key for the dataset to sample from.
            design_key (str): The key for the design to sample from.

        Returns:
            Tuple of:
                np.ndarray: The valid random design.
                int: The random index selected.
        """
        rnd = self.np_random.integers(low=0, high=len(self.dataset[dataset_split]), dtype=int)
        return np.array(self.dataset[dataset_split][design_key][rnd]), rnd


if __name__ == "__main__":
    # --- Create a new problem
    problem = ThermoElastic2D(seed=0)

    # --- Load the problem dataset
    dataset = problem.dataset
    first_item = dataset["train"][0]
    first_item_design = np.array(first_item["optimal_design"])
    problem.render(first_item_design, open_window=True)

    # --- Render the design
    design, _ = problem.random_design()
    problem.render(design, open_window=True)

    # --- Optimize a design ---
    design, _ = problem.random_design()
    design, objectives = problem.optimize(design)
    problem.render(design, open_window=True)

    # --- Evaluate a design ---
    problem.reset(seed=0)
    design, _ = problem.random_design()
    print(problem.simulate(design))
