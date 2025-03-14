"""Heat Conduction 3D Topology Optimization Problem.

This module defines a 3D heat conduction topology optimization problem using the SIMP method.
The problem is solved using the dolfin-adjoint software within a Docker container.
"""

from __future__ import annotations

import os
from typing import Any

from gymnasium import spaces
import numpy as np
import numpy.typing as npt

from engibench.core import ObjectiveDirection
from engibench.core import OptiStep
from engibench.core import Problem
from engibench.utils import container


class HeatConduction3D(Problem[npt.NDArray, str]):
    r"""HeatConduction 3D topology optimization problem.

    ## Problem Description
    This problem simulates the performance of a Topology optimisation of heat conduction problems governed by the Poisson equation (https://github.com/dolfin-adjoint/pyadjoint/blob/master/examples/poisson-topology/poisson-topology.py)

    ## Design space
    The design space is represented by a 3D numpy array which indicates the resolution.

    ## Objectives
    The objective is defined and indexed as follows:
    0. `C`: Thermal compliance coefficient to minimize.

    ## Boundary conditions
    The boundary conditions are defined by the following parameters:
    - `v`: the volume limits on the material distributions
    - `a`: The area of the adiabatic region on the bottom side of the design domain.

    ## Simulator
    The simulator is a docker container with the dolfin-adjoint software that computes the thermal compliance of the design.
    We convert use intermediary files to convert from and to the simulator that is run from a Docker image.

    ## Dataset
    The dataset has been generated the dolfin-adjoint software. It is hosted on the [Hugging Face Datasets Hub](https://huggingface.co/datasets/IDEALLab/heat_conduction_3d_v0).

    ### v0

    #### Fields
    The dataset contains the following fields:
    - `volume`: The volume constraint.
    - `area`: The area of adiabatic surface on the bottom side constraint.
    - `Optimal_Design`: The optimal design.

    #### Creation Method
    The creation method for the dataset is specified in the reference paper.

    ## References
    If you use this problem in your research, please cite the following paper:
    Habibi, Milad, Shai Bernard, Jun Wang, and Mark Fuge, “Mean squared error may lead you astray when optimizing your inverse design methods” in JMD 2025. doi: https://doi.org/10.1115/1.4066102

    ## Lead
    Milad Habibi @MIladHB
    """

    version = 0
    objectives: tuple[tuple[str, ObjectiveDirection], ...] = (("c", ObjectiveDirection.MINIMIZE),)
    conditions: frozenset[tuple[str, Any]] = frozenset(
        {
            ("volume", 0.3),
            ("area", 0.5),
        }
    )
    design_space = spaces.Box(low=0.0, high=1.0, shape=(51, 51, 51), dtype=np.float32)
    dataset_id = "IDEALLab/heat_conduction_3d_v0"
    container_id = "quay.io/dolfinadjoint/pyadjoint:master"
    _dataset = None

    def __init__(self, config: dict[str, Any] = {}) -> None:
        """Initialize the HeatConduction3D problem.

        Args:
            config (dict): A dictionary with configuration (e.g., volume (float): Volume constraint,Area (float): Area constraint,resolution (int): Resolution of the design space) for the initialization.
        """
        super().__init__()
        self.volume = config.get("volume", 0.3)
        self.area = config.get("area", 0.5)
        self.resolution = config.get("resolution", 51)
        self.conditions = frozenset(
            {
                ("volume", self.volume),
                ("area", self.area),
            }
        )
        self.design_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.resolution, self.resolution, self.resolution), dtype=np.float32
        )

    def simulate(self, design: npt.NDArray | None = None, config: dict[str, Any] = {}) -> npt.NDArray:
        """Simulate the design.

        Args:
            design (Optional[np.ndarray]): The design to simulate.
            config (dict): A dictionary with configuration (e.g., volume (float): Volume constraint,area (float): Area constraint,resolution (int): Resolution of the design space) for the simulation.

        Returns:
            float: The thermal compliance of the design.
        """
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
        return np.array(perf)

    def optimize(
        self, starting_point: npt.NDArray | None = None, config: dict[str, Any] = {}
    ) -> tuple[npt.NDArray, list[OptiStep]]:
        """Optimizes the design.

        Args:
            starting_point (npt.NDArray | None): The initial design for optimization.
            config (dict): A dictionary with configuration (e.g., volume (float): Volume constraint,Area (float): Area constraint,resolution (int): Resolution of the design space) for the simulation.

        Returns:
            Tuple[OptimalDesign, list[OptiStep]]: The optimized design and the optimization history.
        """
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
        os.system(f"cp -r {templates_location}/* templates/")

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
            raise FileNotFoundError(error_msg)  # ruff: noqa: TRY003

        file_npy = np.load(design_file)

        return file_npy

    def random_design(self) -> tuple[npt.NDArray, int]:
        """Samples a valid random design.

        Returns:
            Tuple of:
                np.ndarray: The valid random design.
                int: The random index selected.
        """
        rnd = np.random.randint(low=0, high=len(self.dataset["train"]["Optimal_Design"]), dtype=int)  # type: ignore
        return np.array(self.dataset["train"]["Optimal_Design"][rnd]), rnd  # type: ignore

    def render(self, design: npt.NDArray, open_window: bool = False) -> Any:
        """Renders the design in a human-readable format.

        Args:
            design (np.ndarray): The design to render.
            open_window (bool): If True, opens a window with the rendered design.

        Returns:
            Any: The rendered design.
        """
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
    string_array = problem.dataset["train"]["Optimal_Design"][0]
    numpy_array = np.array(string_array)
    des, traj = problem.optimize(starting_point=numpy_array)
    problem.render(design=numpy_array, open_window=True)
