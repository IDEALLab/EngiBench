"""Heat Conduction 3D Topology Optimization Problem.

This module defines a 3D heat conduction topology optimization problem using the SIMP method.
The problem is solved using the dolfin-adjoint software within a Docker container.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from gymnasium import spaces
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from engibench.constraint import bounded
from engibench.constraint import constraint
from engibench.constraint import Criticality
from engibench.constraint import IMPL
from engibench.constraint import THEORY
from engibench.core import ObjectiveDirection
from engibench.core import OptiStep
from engibench.core import Problem
from engibench.problems.heatconduction2d.shared import load_float
from engibench.problems.heatconduction2d.shared import run_container_script
from engibench.utils import cli


@constraint(categories=THEORY, criticality=Criticality.Warning)
def volume_fraction_bound(design: npt.NDArray, volume: float) -> None:
    """Constraint for volume fraction of the design."""
    actual_volfrac = design.mean()
    tolerance = 0.01
    assert abs(actual_volfrac - volume) <= tolerance, (
        f"Volume fraction of the design {actual_volfrac:.4f} does not match target {volume:.4f} specified in the conditions. While the optimizer might fix it, this is likely to affect objective values as the initial design is not feasible given the constraints."
    )


class HeatConduction3D(Problem[npt.NDArray]):
    r"""HeatConduction 3D topology optimization problem.

    Problem Description
    This problem simulates the performance of a Topology optimisation of heat conduction problems governed by the Poisson equation (https://github.com/dolfin-adjoint/pyadjoint/blob/master/examples/poisson-topology/poisson-topology.py)
    """

    version = 0
    objectives: tuple[tuple[str, ObjectiveDirection], ...] = (("c", ObjectiveDirection.MINIMIZE),)

    @dataclass
    class Conditions:
        """Structured representation of the conditions."""

        volume: Annotated[
            float,
            bounded(lower=0.0, upper=1.0).category(THEORY),
            bounded(lower=0.3, upper=0.6).warning().category(IMPL),
        ] = 0.3
        """Volume limits on the material distributions"""
        area: Annotated[float, bounded(lower=0.0, upper=1.0).category(THEORY)] = 0.5
        """Area of the adiabatic region on the bottom side of the design domain"""

    @dataclass
    class Config(Conditions):
        """Structured representation of configuration parameters for a numerical computation."""

        resolution: Annotated[
            int, bounded(lower=1).category(THEORY), bounded(lower=10, upper=1000).warning().category(IMPL)
        ] = 51
        """Resolution of the design space"""
        max_iter: int = 100
        """Maximum number of iterations for the solver in `optimize()`."""

    config: Config

    design_constraints = (volume_fraction_bound,)
    design_space = spaces.Box(low=0.0, high=1.0, shape=(51, 51, 51), dtype=np.float64)
    dataset_id = "IDEALLab/heat_conduction_3d_v0"
    container_id = "quay.io/dolfinadjoint/pyadjoint:master"

    def __init__(self, seed: int = 0, **kwargs) -> None:
        """Initialize the HeatConduction3D problem.

        Args:
            seed (int): The random seed for the problem.
            kwargs: Arguments are passed to :class:`HeatConduction3D.Config`.
        """
        super().__init__(seed=seed)
        self.config = self.Config(**kwargs)
        resolution = self.config.resolution
        self.conditions = self.Conditions(self.config.volume, self.config.area)
        self.design_space = spaces.Box(low=0.0, high=1.0, shape=(resolution, resolution, resolution), dtype=np.float64)

    def simulate(self, design: npt.NDArray | None = None, config: dict[str, Any] | None = None) -> npt.NDArray:
        """Simulate the design.

        Args:
            design (Optional[np.ndarray]): The design to simulate.
            config (dict): A dictionary with configuration (e.g., volume (float): Volume constraint,area (float): Area constraint,resolution (int): Resolution of the design space) for the simulation.

        Returns:
            float: The thermal compliance of the design.
        """
        config = config or {}
        volume = config.get("volume", self.config.volume)
        area = config.get("area", self.config.area)
        resolution = config.get("resolution", self.config.resolution)
        if design is None:
            design = self.initialize_design(volume, resolution)

        perf = load_float(
            run_container_script(
                self.container_id,
                Path(__file__).parent / "templates" / "simulate_heat_conduction_3d.py",
                args=(resolution - 1, volume, area),
                stdin=cli.np_array_to_bytes(design),
                output_path="RES_SIM/Performance.txt",
            )
        )

        return np.array([float(perf)])

    def optimize(
        self, starting_point: npt.NDArray | None = None, config: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray, list[OptiStep]]:
        """Optimizes the design.

        Args:
            starting_point (npt.NDArray | None): The initial design for optimization.
            config (dict): A dictionary with configuration (e.g., volume (float): Volume constraint,Area (float): Area constraint,resolution (int): Resolution of the design space) for the simulation.

        Returns:
            Tuple[OptimalDesign, list[OptiStep]]: The optimized design and the optimization history.
        """
        config = config or {}
        volume = config.get("volume", self.config.volume)
        area = config.get("area", self.config.area)
        resolution = config.get("resolution", self.config.resolution)
        max_iter = config.get("max_iter", self.config.max_iter)
        if starting_point is None:
            starting_point = self.initialize_design(volume, resolution)

        output = np.load(
            run_container_script(
                self.container_id,
                Path(__file__).parent / "templates" / "optimize_heat_conduction_3d.py",
                args=(resolution - 1, volume, area, max_iter),
                stdin=cli.np_array_to_bytes(starting_point),
                output_path=f"RES_OPT/OUTPUT={volume}_w={area}.npz",
            )
        )

        steps = output["OptiStep"]
        optisteps = [OptiStep(step, it) for it, step in enumerate(steps)]

        return output["design"], optisteps

    def reset(self, seed: int | None = None, **kwargs) -> None:
        """Reset the problem to a given seed."""
        super().reset(seed, **kwargs)

    def initialize_design(self, volume: float | None = None, resolution: int | None = None) -> npt.NDArray:
        """Initialize the design based on SIMP method.

        Args:
            volume (Optional[float]): Volume constraint.
            resolution (Optional[int]): Resolution of the design space.

        Returns:
            HeatConduction3D: The initialized design.
        """
        volume = volume if volume is not None else self.config.volume
        resolution = resolution if resolution is not None else self.config.resolution

        return np.load(
            run_container_script(
                self.container_id,
                Path(__file__).parent / "templates" / "initialize_design_3d.py",
                args=(resolution - 1, volume),
                output_path=f"initialize_design/initial_v={volume}_resol={resolution}.npy",
            )
        )

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
        rnd = self.np_random.integers(low=0, high=len(self.dataset[dataset_split][design_key]), dtype=int)
        return np.array(self.dataset[dataset_split][design_key][rnd]), rnd

    def render(self, design: npt.NDArray, *, open_window: bool = False) -> Figure:
        """Renders the design in a human-readable format.

        Args:
            design (np.ndarray): The design to render.
            open_window (bool): If True, opens a window with the rendered design.

        Returns:
            Figure: The rendered design.
        """
        print("For better visualization, also consider dedicated software like ParaView.")
        if design is None:
            design = self.initialize_design()

        size = len(design) + 1

        fig = plt.figure()
        ax: Any = fig.add_subplot(111, projection="3d")
        x, y, z = np.indices((size + 1, size + 1, size + 1)) / size  # Normalize to [0,1]
        # Define which voxels to plot
        threshold = 0.7
        filled = design > threshold
        # Adjust voxel positions by shifting their centers
        ax.voxels(x[:-1, :-1, :-1], y[:-1, :-1, :-1], z[:-1, :-1, :-1], filled, edgecolor="k", alpha=0.7)
        cube_vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])

        cube_edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # Bottom face
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),  # Top face
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]  # Side edges

        for edge in cube_edges:
            ax.plot(*zip(*cube_vertices[list(edge)], strict=True), color="red", linewidth=2)

        if open_window:
            plt.show()
        return fig


# Check if the script is run directly
if __name__ == "__main__":
    # Create a HeatConduction3D problem instance
    problem = HeatConduction3D(seed=0)
    design_as_list = problem.dataset["train"]["optimal_design"][0]
    numpy_array = np.array(design_as_list)
    des, traj = problem.optimize(starting_point=numpy_array)
    problem.render(design=numpy_array, open_window=True)
