"""Thermo Elastic 3D Problem."""

import dataclasses
from dataclasses import dataclass
from typing import Annotated, Any

from datasets import Dataset
from gymnasium import spaces
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
from engibench.core import SimulationResult
from engibench.problems.thermoelastic3d.model import fem_model
from engibench.problems.thermoelastic3d.model.fem_model import FeaModel3D
from engibench.utils.upcast import upcast

NELX = NELY = NELZ = 16
DENSITY_THRESHOLD = 0.5  # voxels denser than this are drawn solid by render()
# design is indexed [y, x, z]; transpose to [x, y, z] so the axes read naturally.


def _default_force_elements(nelx: int, nely: int, nelz: int) -> npt.NDArray[np.int64]:
    """Build a vertical-load node mask from fractional load coordinates."""
    out = np.zeros((nelx + 1, nely + 1, nelz + 1), dtype=np.int64)
    out[-1, -1, -1] = 1
    return out


def _fixed_elements(nelx: int, nely: int, nelz: int) -> npt.NDArray[np.int64]:
    """Default fixed elements: clamp 3 bottom corners."""
    out = np.zeros((nelx + 1, nely + 1, nelz + 1), dtype=np.int64)
    out[0, 0, 0] = out[0, -1, 0] = out[0, -1, -1] = 1
    return out


def _default_heatsink_elements(nelx: int, nely: int, nelz: int) -> npt.NDArray[np.int64]:
    """Default fixed elements: clamp 3 bottom corners."""
    out = np.zeros((nelx + 1, nely + 1, nelz + 1), dtype=np.int64)
    out[-1, -1, 0] = 1
    return out


class ThermoElastic3D(Problem[npt.NDArray]):
    """Truss 3D integer optimization problem.

    This is 3D topology optimization problem for minimizing weakly coupled thermo-elastic compliance subject to boundary conditions and a volume fraction constraint.
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

        fixed_elements: Annotated[npt.NDArray[np.int64], bounded(lower=0.0, upper=1.0).category(THEORY)]
        """Binary NxNxN array of the structurally fixed elements in the domain"""
        force_elements_x: Annotated[npt.NDArray[np.int64], bounded(lower=0.0, upper=1.0).category(THEORY)]
        """Binary NxNxN array specifying elements that have a structural load in the x-direction"""
        force_elements_y: Annotated[npt.NDArray[np.int64], bounded(lower=0.0, upper=1.0).category(THEORY)]
        """Binary NxNxN array specifying elements that have a structural load in the y-direction"""
        force_elements_z: Annotated[npt.NDArray[np.int64], bounded(lower=0.0, upper=1.0).category(THEORY)]
        """Binary NxNxN array specifying elements that have a structural load in the z-direction"""
        heatsink_elements: Annotated[npt.NDArray[np.int64], bounded(lower=0.0, upper=1.0).category(THEORY)]
        """Binary NxNxN array specifying elements that have a heat sink"""
        volfrac: Annotated[float, bounded(lower=0.0, upper=1.0).category(THEORY)] = 0.3
        """Target volume fraction for the volume fraction constraint"""
        rmin: Annotated[
            float, bounded(lower=1.0).category(THEORY), bounded(lower=0.0, upper=3.0).warning().category(IMPL)
        ] = 1.5
        """Filter size used in the optimization routine"""
        penal: Annotated[
            float, bounded(lower=1.0).category(THEORY), bounded(lower=0.0, upper=10.0).warning().category(IMPL)
        ] = 3.0
        weight: Annotated[float, bounded(lower=0.0, upper=1.0).category(THEORY)] = 0.5
        """Control which objective is optimized for. 1.0 is pure structural optimization, while 0.0 is pure thermal optimization"""

    design_space = spaces.Box(low=0.0, high=1.0, shape=(NELX, NELY, NELZ), dtype=np.float32)
    dataset_id = "IDEALLab/thermoelastic_3d_v0"
    container_id = None

    @dataclass
    class Config(Conditions):
        """Structured representation of configuration parameters for a numerical computation."""

        nelx: Annotated[int, bounded(lower=1).category(THEORY)] = NELX
        nely: Annotated[int, bounded(lower=1).category(THEORY)] = NELY
        nelz: Annotated[int, bounded(lower=1).category(THEORY)] = NELZ
        max_iter: int = fem_model.MAX_ITERATIONS
        """Maximal number of iterations for optimize."""

        @constraint
        @staticmethod
        def rmin_bound(rmin: float, nelx: int, nely: int, nelz: int) -> None:
            """Constraint for rmin ∈ (0.0, max{ nelx, nely, nelz }]."""
            assert 0.0 < rmin <= max(nelx, nely, nelz), f"Params.rmin: {rmin} ∉ (0, max(nelx, nely, nelz)]"

        @constraint
        @staticmethod
        def bc_check(
            *,
            nelx: int,
            nely: int,
            nelz: int,
            fixed_elements: npt.NDArray[np.int64],
            force_elements_x: npt.NDArray[np.int64],
            force_elements_y: npt.NDArray[np.int64],
            force_elements_z: npt.NDArray[np.int64],
            heatsink_elements: npt.NDArray[np.int64],
        ) -> None:
            """Constraint to ensure boundary conditions are valid."""
            assert fixed_elements.shape == (nelx + 1, nely + 1, nelz + 1), "Params.fixed_elements has invalid shape."
            assert force_elements_x.shape == (nelx + 1, nely + 1, nelz + 1), "Params.force_elements_x has invalid shape."
            assert force_elements_y.shape == (nelx + 1, nely + 1, nelz + 1), "Params.force_elements_y has invalid shape."
            assert force_elements_z.shape == (nelx + 1, nely + 1, nelz + 1), "Params.force_elements_z has invalid shape."
            assert heatsink_elements.shape == (nelx + 1, nely + 1, nelz + 1), "Params.heatsink_elements has invalid shape."

        def __init__(
            self,
            nelx: int = NELX,
            nely: int = NELY,
            nelz: int = NELZ,
            max_iter: int = fem_model.MAX_ITERATIONS,
            **kwargs: Any,
        ) -> None:
            """Manual __init__ which handles fixed / force / heatsink elements."""
            super().__init__(
                **{
                    "fixed_elements": _fixed_elements(nelx, nely, nelz),
                    "force_elements_x": _default_force_elements(nelx, nely, nelz),
                    "force_elements_y": _default_force_elements(nelx, nely, nelz),
                    "force_elements_z": _default_force_elements(nelx, nely, nelz),
                    "heatsink_elements": _default_heatsink_elements(nelx, nely, nelz),
                    **kwargs,
                }
            )
            self.max_iter = max_iter
            self.nelx = nelx
            self.nely = nely
            self.nelz = nelz

    conditions = upcast(Config())

    @property
    def dataset(self) -> Dataset:
        """Pull a supported cubic dataset, failing early for unsupported grids."""
        if not self.dataset_id:
            raise ValueError(
                "Thermoelastic3D dataset access is implemented only for cubic grids "
                f"with nelx = nely = nelz = {NELX}. Got ({self.nelx}, {self.nely}, {self.nelz})."
            )
        return super().dataset

    def reset(self, seed: int | None = None) -> None:
        """Resets the simulator and numpy random to a given seed.

        Args:
            seed (int, optional): The seed to reset to. If None, a random seed is used.
        """
        super().reset(seed)

    def __init__(self, seed: int = 0, config: dict[str, Any] | None = None) -> None:
        """Initialize the problem, sizing default masks and design space to the grid."""
        super().__init__(seed=seed)
        raw_config: dict[str, Any] = {
            key: (np.asarray(value) if isinstance(value, list) else value) for key, value in (config or {}).items()
        }
        self.config = self.Config(**raw_config)

        self.conditions = upcast(self.config)
        self.nelx, self.nely, self.nelz = self.config.nelx, self.config.nely, self.config.nelz
        self.max_iter = self.config.max_iter
        # Note: nely before nelx, as the solver stores it:
        self.design_space = spaces.Box(low=0.0, high=1.0, shape=(self.nely, self.nelx, self.nelz), dtype=np.float32)
        if (self.nelx, self.nely, self.nelz) != (NELX, NELY, NELZ):
            self.dataset_id = ""

    def simulate_verbose(self, design: npt.NDArray, config: dict[str, Any] | None = None) -> SimulationResult:
        r"""Launch a simulation on the given design topology and return the performance.

        Args:
            design (np.ndarray): The design to simulate.
            config (dict): A dictionary with configuration (e.g., boundary conditions, filenames) for the simulation.

        Returns:
            A `SimulationResult` instance.
        """
        boundary_dict = dataclasses.asdict(self.conditions)
        for key, value in (config or {}).items():
            if key in boundary_dict:
                if isinstance(value, list):
                    boundary_dict[key] = np.array(value)
                else:
                    boundary_dict[key] = value

        results = FeaModel3D(eval_only=True).run(boundary_dict, x_init=design)
        return SimulationResult(
            np.array([results["structural_compliance"], results["thermal_compliance"], results["volume_fraction"]])
        )

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
        results = FeaModel3D(eval_only=False, max_iter=(config or {}).get("max_iter", self.max_iter)).run(
            boundary_dict, x_init=starting_point
        )
        design = np.array(results["design"]).astype(np.float32)
        opti_steps = results["opti_steps"]
        return design, opti_steps

    def render(self, design: np.ndarray, *, open_window: bool = False) -> Any:
        """Renders the design in a human-readable format.

        Args:
            design (np.ndarray): The design to render.
            open_window (bool): If True, opens a window with the rendered design.

        Returns:
            fig (np.ndarray): The rendered design.
        """
        solid = np.asarray(design).transpose(1, 0, 2) > DENSITY_THRESHOLD

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.voxels(solid, edgecolor="gray")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        if open_window:
            plt.show()
        return fig, ax

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
    problem = ThermoElastic3D(seed=0)

    # --- Load the problem dataset
    dataset = problem.dataset
    first_item = dataset["train"][0]
    first_item_design = np.array(first_item["optimal_design"])
    problem.render(first_item_design, open_window=True)

    # --- Render the design
    design, _ = problem.random_design()
    problem.render(design, open_window=True)

    # --- Optimize a design ---
    design = 0.2 * np.ones((NELX, NELY, NELZ), dtype=float)
    design, objectives = problem.optimize(design)
    problem.render(design, open_window=True)

    # --- Evaluate a design ---
    problem.reset(seed=0)
    design, _ = problem.random_design()
    print(problem.simulate(design))
