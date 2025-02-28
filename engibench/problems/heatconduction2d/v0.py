"""Heat Conduction 2D Topology Optimization Problem.

This module defines a 2D heat conduction topology optimization problem using the SIMP method.
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


class HeatConduction2D(Problem[npt.NDArray, str]):
    r"""HeatConduction 2D topology optimization problem.

    ## Problem Description
    This problem simulates the performance of a Topology optimisation of heat conduction problems governed by the Poisson equation (https://www.dolfin-adjoint.org/en/stable/documentation/poisson-topology/poisson-topology.html)

    ## Design space
    The design space is represented by a 2D numpy array which indicates the resolution.

    ## Objectives
    The objective is defined and indexed as follows:
    0. `C`: Thermal compliance coefficient to minimize.

    ## Boundary conditions
    The boundary conditions are defined by the following parameters:
    - `v`: the volume limits on the material distributions
    - `l`: The length of the adiabatic region on the bottom side of the design domain.

    ## Simulator
    The simulator is a docker container with the dolfin-adjoint software that computes the thermal compliance of the design.
    We convert use intermediary files to convert from and to the simulator that is run from a Docker image.

    ## Dataset
    The dataset has been generated the dolfin-adjoint software. It is hosted on the [Hugging Face Datasets Hub](https://huggingface.co/datasets/IDEALLab/heat_conduction_2d_v0).

    ### v0

    #### Fields
    The dataset contains the following fields:
    - `volume`: The volume constraint.
    - `length`: The length constraint.
    - `Optimal_Design`: The optimal design.

    #### Creation Method
    The creation method for the dataset is specified in the reference paper.

    ## References
    # TODO add Milad's paper here

    ## Lead
    Milad Habibi @MIladHB
    """

    version = 0
    possible_objectives: tuple[tuple[str, ObjectiveDirection], ...] = (("c", ObjectiveDirection.MINIMIZE),)
    boundary_conditions: frozenset[tuple[str, Any]] = frozenset(
        {
            ("volume", 0.5),
            ("length", 0.5),
        }
    )
    design_space = spaces.Box(low=0.0, high=1.0, shape=(50, 50), dtype=np.float32)
    dataset_id = "IDEALLab/heat_conduction_2d_v0"
    container_id = "quay.io/dolfinadjoint/pyadjoint:master"
    _dataset = None

    def __init__(self, config: dict[str, Any] = {}) -> None:
        """Initialize the HeatConduction2D problem.

        Args:
            config (dict): A dictionary with configuration (e.g., volume (float): Volume constraint,length (float): Length constraint,resolution (int): Resolution of the design space) for the initilization.
        """
        super().__init__()
        self.volume = config.get("volume", 0.5)
        self.length = config.get("length", 0.5)
        self.resolution = config.get("resolution", 50)
        self.boundary_conditions = frozenset(
            {
                ("volume", self.volume),
                ("length", self.length),
            }
        )
        self.design_space = spaces.Box(low=0.0, high=1.0, shape=(self.resolution, self.resolution), dtype=np.float32)

    def simulate(self, design: npt.NDArray | None = None, config: dict[str, Any] = {}) -> npt.NDArray:
        """Simulate the design.

        Args:
            design (Optional[np.ndarray]): The design to simulate.
            config (dict): A dictionary with configuration (e.g., volume (float): Volume constraint,length (float): Length constraint,resolution (int): Resolution of the design space) for the simulation.

        Returns:
            float: The thermal compliance of the design.
        """
        volume = config.get("volume", self.volume)
        length = config.get("length", self.length)
        resolution = config.get("resolution", self.resolution)
        if design is None:
            des = self.initialize_design(volume, resolution)
            design = des[:, 2].reshape(resolution + 1, resolution + 1)

        self.__copy_templates()
        with open("templates/sim_var.txt", "w") as f:
            f.write(f"{volume}\t{length}\t{resolution}")

        filename = "templates/hr_data_v=" + str(volume) + "_w=" + str(length) + "_.npy"
        np.save(filename, design)

        current_dir = os.getcwd()
        container.run(
            command=["python3", "/home/fenics/shared/templates/simulate_heat_conduction_2d.py"],
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
            design (Optional[np.ndarray]): The design to optimize.
            config (dict): A dictionary with configuration (e.g., volume (float): Volume constraint,length (float): Length constraint,resolution (int): Resolution of the design space) for the simulation.

        Returns:
            Tuple[OptimalDesign, list[OptiStep]]: The optimized design and the optimization history.
        """
        volume = config.get("volume", self.volume)
        length = config.get("length", self.length)
        resolution = config.get("resolution", self.resolution)
        if starting_point is None:
            des = self.initialize_design(volume, resolution)
            design = des[:, 2].reshape(resolution + 1, resolution + 1)
        else:
            design = starting_point

        self.__copy_templates()
        with open("templates/OPT_var.txt", "w") as f:
            f.write(f"{volume}\t{length}\t{resolution}")

        filename = "templates/hr_data_OPT_v=" + str(volume) + "_w=" + str(length) + "_.npy"
        np.save(filename, design)

        current_dir = os.getcwd()
        container.run(
            command=["python3", "/home/fenics/shared/templates/optimize_heat_conduction_2d.py"],
            image=self.container_id,
            name="dolfin",
            mounts=[(current_dir, "/home/fenics/shared")],
        )
        output = np.load("templates/RES_OPT/OUTPUT=" + str(volume) + "_w=" + str(length) + "_.npz")

        return output["design"], output["OptiStep"]

    def reset(self, seed: int | None = None, **kwargs) -> None:
        """Reset the problem to a given seed."""
        super().reset(seed, **kwargs)

    def __copy_templates(self):
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
            HeatConduction2D: The initialized design.
        """
        volume = volume if volume is not None else self.volume
        resolution = resolution if resolution is not None else self.resolution

        self.__copy_templates()
        with open("templates/Des_var.txt", "w") as f:
            f.write(f"{volume}\t{resolution}")

        # Run the Docker command
        current_dir = os.getcwd()
        container.run(
            command=["python3", "/home/fenics/shared/templates/initialize_design_2d.py"],
            image=self.container_id,
            name="dolfin",
            mounts=[(current_dir, "/home/fenics/shared")],
        )

        # Load the generated design data from the numpy file
        design_file = f"templates/initialize_design/initial_v={volume}_resol={resolution}_.npy"
        if not os.path.exists(design_file):
            raise FileNotFoundError(f"Design file {design_file} not found.")  # ruff: noqa: TRY003

        file_npy = np.load(design_file)

        return file_npy

    def random_design(self) -> tuple[npt.NDArray, int]:
        """Generate a random design."""
        return self.initialize_design(), -1  # use random volume and resolution?

    def render(self, design: npt.NDArray, open_window: bool = False) -> Any:
        """Renders the design in a human-readable format.

        Args:
            design (np.ndarray): The design to render.
            open_window (bool): If True, opens a window with the rendered design.

        Returns:
            Any: The rendered design.
        """
        if design is None:
            des = self.initialize_design()
            design = des[:, 2].reshape(self.resolution + 1, self.resolution + 1)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ax.imshow(design)
        if open_window:
            plt.show()
        return fig, ax


# Check if the script is run directly
if __name__ == "__main__":
    # Create a HeatConduction2D problem instance
    problem = HeatConduction2D()

    # Call the design method and print the result
    design, _ = problem.random_design()
    print(design)
    problem.render(design, open_window=True)

    # print(problem.simulate())
    # print(problem.optimize())
