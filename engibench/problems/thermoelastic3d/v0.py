"""Thermo Elastic 3D Problem."""

import dataclasses
from dataclasses import dataclass
from dataclasses import field
from typing import Annotated, Any

from gymnasium import spaces
import napari
import numpy as np
import numpy.typing as npt

from engibench.constraint import bounded
from engibench.constraint import constraint
from engibench.constraint import IMPL
from engibench.constraint import THEORY
from engibench.core import ObjectiveDirection
from engibench.core import OptiStep
from engibench.core import Problem
from engibench.problems.thermoelastic3d.model import fem_model
from engibench.problems.thermoelastic3d.model.fem_model import FeaModel3D

NELX = NELY = NELZ = 16
FIXED_ELEMENTS = np.zeros((NELX + 1, NELY + 1, NELZ + 1), dtype=int)
FIXED_ELEMENTS[0, 0, 0] = 1
FIXED_ELEMENTS[0, -1, -1] = 1
FIXED_ELEMENTS[0, -1, 0] = 1
FORCE_ELEMENTS_X = np.zeros((NELX + 1, NELY + 1, NELZ + 1), dtype=int)
FORCE_ELEMENTS_X[-1, -1, -1] = 1
FORCE_ELEMENTS_Y = np.zeros((NELX + 1, NELY + 1, NELZ + 1), dtype=int)
FORCE_ELEMENTS_Y[-1, -1, -1] = 1
FORCE_ELEMENTS_Z = np.zeros((NELX + 1, NELY + 1, NELZ + 1), dtype=int)
FORCE_ELEMENTS_Z[-1, -1, -1] = 1
HEATSINK_ELEMENTS = np.zeros((NELX + 1, NELY + 1, NELZ + 1), dtype=int)
HEATSINK_ELEMENTS[-1, -1, 0] = 1


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

        fixed_elements: Annotated[npt.NDArray[np.int64], bounded(lower=0.0, upper=1.0).category(THEORY)] = field(
            default_factory=lambda: FIXED_ELEMENTS
        )
        """Binary NxNxN array of the structurally fixed elements in the domain"""
        force_elements_x: Annotated[npt.NDArray[np.int64], bounded(lower=0.0, upper=1.0).category(THEORY)] = field(
            default_factory=lambda: FORCE_ELEMENTS_X
        )
        """Binary NxNxN array specifying elements that have a structural load in the x-direction"""
        force_elements_y: Annotated[npt.NDArray[np.int64], bounded(lower=0.0, upper=1.0).category(THEORY)] = field(
            default_factory=lambda: FORCE_ELEMENTS_Y
        )
        """Binary NxNxN array specifying elements that have a structural load in the y-direction"""
        force_elements_z: Annotated[npt.NDArray[np.int64], bounded(lower=0.0, upper=1.0).category(THEORY)] = field(
            default_factory=lambda: FORCE_ELEMENTS_Z
        )
        """Binary NxNxN array specifying elements that have a structural load in the z-direction"""
        heatsink_elements: Annotated[npt.NDArray[np.int64], bounded(lower=0.0, upper=1.0).category(THEORY)] = field(
            default_factory=lambda: HEATSINK_ELEMENTS
        )
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

    conditions = Conditions()
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
        def bc_check(  # noqa: PLR0913
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

        results = FeaModel3D(plot=False, eval_only=True).run(boundary_dict, x_init=design)
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
        results = FeaModel3D(
            plot=False, eval_only=False, max_iter=(config or {}).get("max_iter", self.Config.max_iter)
        ).run(boundary_dict, x_init=starting_point)
        design = np.array(results["design"]).astype(np.float32)
        opti_steps = results["opti_steps"]
        return design, opti_steps

    def render(self, design: np.ndarray, *, open_window: bool = False) -> np.ndarray:
        """Renders the design in a human-readable format.

        Args:
            design (np.ndarray): The design to render.
            open_window (bool): If True, opens a window with the rendered design.

        Returns:
            fig (np.ndarray): The rendered design.
        """
        design = np.array(design)
        design = np.transpose(design, (2, 0, 1))

        viewer = napari.Viewer()
        viewer.add_image(design, name="rho", rendering="attenuated_mip")
        viewer.dims.ndisplay = 3  # switch to 3D view
        if open_window is True:
            napari.run()
        return viewer.export_figure(flash=False)

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
