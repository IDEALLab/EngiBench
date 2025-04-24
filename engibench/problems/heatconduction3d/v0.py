"""Heat Conduction 3D Topology Optimization Problem.

This module defines a 3D heat conduction topology optimization problem using the SIMP method.
The problem is solved using the dolfin-adjoint software within a Docker container.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
from typing import Annotated, Any

from gymnasium import spaces
import numpy as np
import numpy.typing as npt

from engibench.constraint import bounded
from engibench.constraint import IMPL
from engibench.constraint import THEORY
from engibench.core import ObjectiveDirection
from engibench.core import OptiStep
from engibench.core import Problem
from engibench.utils import container


class HeatConduction3D(Problem[npt.NDArray]):
    r"""HeatConduction 3D topology optimization problem.

    ## Problem Description
    This problem simulates the performance of a Topology optimisation of heat conduction problems governed by the Poisson equation (https://github.com/dolfin-adjoint/pyadjoint/blob/master/examples/poisson-topology/poisson-topology.py)

    ## Design space
    The design space is represented by a 3D numpy array which indicates the resolution.

    ## Objectives
    The objective is defined and indexed as follows:

    0. `c`: Thermal compliance coefficient to minimize.

    ## Conditions
    The conditions are defined by the following parameters:
    - `volume`: the volume limits on the material distributions
    - `area`: The area of the adiabatic region on the bottom side of the design domain.

    ## Simulator
    The simulator is a docker container with the dolfin-adjoint software that computes the thermal compliance of the design.
    We convert use intermediary files to convert from and to the simulator that is run from a Docker image.

    ## Dataset
    The dataset has been generated the dolfin-adjoint software. It is hosted on the [Hugging Face Datasets Hub](https://huggingface.co/datasets/IDEALLab/heat_conduction_3d_v0).

    ### v0

    #### Fields
    The dataset only contains conditions and optimal designs (no objective).

    #### Creation Method
    The creation method for the dataset is specified in the reference paper.

    ## References
    If you use this problem in your research, please cite the following paper:
    Habibi, Milad, Shai Bernard, Jun Wang, and Mark Fuge, "Mean squared error may lead you astray when optimizing your inverse design methods" in JMD 2025. doi: https://doi.org/10.1115/1.4066102

    ## Lead
    Milad Habibi @MIladHB
    """

    version = 0
    objectives: tuple[tuple[str, ObjectiveDirection], ...] = (("c", ObjectiveDirection.MINIMIZE),)
    conditions: tuple[tuple[str, Any], ...] = (
        ("volume", 0.3),
        ("area", 0.5),
    )
    design_space = spaces.Box(low=0.0, high=1.0, shape=(51, 51, 51), dtype=np.float64)
    dataset_id = "IDEALLab/heat_conduction_3d_v0"
    container_id = "quay.io/dolfinadjoint/pyadjoint:master"

    def __init__(self, volume: float = 0.3, area: float = 0.5, resolution: int = 51) -> None:
        """Initialize the HeatConduction3D problem.

        Args:
            volume (float): Volume constraint.
            area (float): Area constraint.
            resolution (int): Resolution of the design space.
        """
        super().__init__()
        self.volume = volume
        self.area = area
        self.resolution = resolution
        self.conditions = (
            ("volume", self.volume),
            ("area", self.area),
        )
        self.design_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.resolution, self.resolution, self.resolution), dtype=np.float64
        )

        @dataclass
        class Config:
            """Structured representation of configuration parameters for a numerical computation."""

            resolution: Annotated[
                int, bounded(lower=1).category(THEORY), bounded(lower=10, upper=1000).warning().category(IMPL)
            ] = self.resolution
            volume: Annotated[
                float,
                bounded(lower=0.0, upper=1.0).category(THEORY),
                bounded(lower=0.3, upper=0.6).warning().category(IMPL),
            ] = self.volume
            area: Annotated[float, bounded(lower=0.0, upper=1.0).category(THEORY)] = self.area

        self.Config = Config

    def simulate(self, design: npt.NDArray | None = None, config: dict[str, Any] | None = None) -> npt.NDArray:
        """Simulate the design.

        Args:
            design (Optional[np.ndarray]): The design to simulate.
            config (dict): A dictionary with configuration (e.g., volume (float): Volume constraint,area (float): Area constraint,resolution (int): Resolution of the design space) for the simulation.

        Returns:
            float: The thermal compliance of the design.
        """
        config = config or {}
        volume = config.get("volume", self.volume)
        area = config.get("area", self.area)
        resolution = config.get("resolution", self.resolution)
        if design is None:
            design = self.initialize_design(volume, resolution)

        self.__copy_templates()
        with open("templates/sim_var.txt", "w") as f:
            f.write(f"{volume}\t{area}\t{resolution}")

        filename = "templates/hr_data_v=" + str(volume) + "_w=" + str(area) + "_.npy"
        np.save(filename, design)

        current_dir = os.getcwd()
        container.run(
            command=["python3", "/home/fenics/shared/templates/simulate_heat_conduction_3d.py"],
            image=self.container_id,
            name="dolfin",
            mounts=[(current_dir, "/home/fenics/shared")],
        )

        with open(r"templates/RES_SIM/Performance.txt") as fp:
            perf = fp.read()
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
        volume = config.get("volume", self.volume)
        area = config.get("area", self.area)
        resolution = config.get("resolution", self.resolution)
        if starting_point is None:
            starting_point = self.initialize_design(volume, resolution)

        self.__copy_templates()
        with open("templates/OPT_var.txt", "w") as f:
            f.write(f"{volume}\t{area}\t{resolution}")

        filename = "templates/hr_data_OPT_v=" + str(volume) + "_w=" + str(area) + "_.npy"
        np.save(filename, starting_point)

        current_dir = os.getcwd()
        container.run(
            command=["python3", "/home/fenics/shared/templates/optimize_heat_conduction_3d.py"],
            image=self.container_id,
            name="dolfin",
            mounts=[(current_dir, "/home/fenics/shared")],
        )
        output = np.load("templates/RES_OPT/OUTPUT=" + str(volume) + "_w=" + str(area) + "_.npz")

        steps = output["OptiStep"]
        optisteps = [OptiStep(step, it) for it, step in enumerate(steps)]

        return output["design"], optisteps

    def reset(self, seed: int | None = None, **kwargs) -> None:
        """Reset the problem to a given seed."""
        super().reset(seed, **kwargs)

    def __copy_templates(self) -> None:
        """Copy the templates from the installation location to the current working directory."""
        if not os.path.exists("templates"):
            os.mkdir("templates")
        templates_location = os.path.dirname(os.path.abspath(__file__)) + "/templates/"
        subprocess.run(["cp", "-r", f"{templates_location}/*", "templates/"], check=True)

    def initialize_design(self, volume: float | None = None, resolution: int | None = None) -> npt.NDArray:
        """Initialize the design based on SIMP method.

        Args:
            volume (Optional[float]): Volume constraint.
            resolution (Optional[int]): Resolution of the design space.

        Returns:
            HeatConduction3D: The initialized design.
        """
        volume = volume if volume is not None else self.volume
        resolution = resolution if resolution is not None else self.resolution

        self.__copy_templates()
        with open("templates/Des_var.txt", "w") as f:
            f.write(f"{volume}\t{resolution}")

        # Run the Docker command
        current_dir = os.getcwd()
        container.run(
            command=["python3", "/home/fenics/shared/templates/initialize_design_3d.py"],
            image=self.container_id,
            name="dolfin",
            mounts=[(current_dir, "/home/fenics/shared")],
        )

        # Load the generated design data from the numpy file
        design_file = f"templates/initialize_design/initial_v={volume}_resol={resolution}_.npy"
        if not os.path.exists(design_file):
            error_msg = f"Design file {design_file} not found."
            raise FileNotFoundError(error_msg)

        return np.load(design_file)

    def random_design(self) -> tuple[npt.NDArray, int]:
        """Samples a valid random design.

        Returns:
            Tuple of:
                np.ndarray: The valid random design.
                int: The random index selected.
        """
        rnd = self.np_random.integers(low=0, high=len(self.dataset["train"]["optimal_design"]))
        return np.array(self.dataset["train"]["optimal_design"][rnd]), int(rnd)

    def render(self, design: npt.NDArray, *, open_window: bool = False) -> Any:
        """Renders the design in a human-readable format.

        Args:
            design (np.ndarray): The design to render.
            open_window (bool): If True, opens a window with the rendered design.

        Returns:
            Any: The rendered design.
        """
        print("For better visualization, also consider dedicated software like ParaView.")
        if design is None:
            design = self.initialize_design()

        import matplotlib.pyplot as plt

        size = len(design) + 1

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
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
            ax.plot(*zip(*cube_vertices[list(edge)]), color="red", linewidth=2)

        if open_window:
            plt.show()
        return fig, ax


# Check if the script is run directly
if __name__ == "__main__":
    # Create a HeatConduction3D problem instance
    problem = HeatConduction3D()
    design_as_list = problem.dataset["train"]["optimal_design"][0]
    numpy_array = np.array(design_as_list)
    des, traj = problem.optimize(starting_point=numpy_array)
    problem.render(design=numpy_array, open_window=True)
