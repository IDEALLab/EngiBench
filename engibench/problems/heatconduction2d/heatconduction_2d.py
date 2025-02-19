"""Heat Conduction 2D Topology Optimization Problem.

This module defines a 2D heat conduction topology optimization problem using the SIMP method.
The problem is solved using the dolfin-adjoint software within a Docker container.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

import numpy as np
import numpy.typing as npt

from engibench.core import Problem


def build(**kwargs) -> HeatConduction2D:
    """Builds a HeatConduction2D problem."""
    return HeatConduction2D(**kwargs)


class HeatConduction2D(Problem):
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

    ## Lead
    Milad Habibi @MIladHB
    """

    def __init__(self, volume: float = 0.5, length: float = 0.5, resolution: int = 50) -> None:
        """Initialize the HeatConduction2D problem.

        Args:
            volume (float): Volume constraint.
            length (float): Length constraint.
            resolution (int): Resolution of the design space.
        """
        super().__init__()
        self.volume = volume
        self.length = length
        self.resolution = resolution

    def initialize_design(
        self, volume: float | None = None, length: float | None = None, resolution: int | None = None
    ) -> npt.NDArray:
        """Initialize the design based on SIMP method.

        Args:
            volume (Optional[float]): Volume constraint.
            length (Optional[float]): Length constraint.
            resolution (Optional[int]): Resolution of the design space.

        Returns:
            HeatConduction2D: The initialized design.
        """
        volume = volume if volume is not None else self.volume
        length = length if length is not None else self.length
        resolution = resolution if resolution is not None else self.resolution
        with open("templates/Des_var.txt", "w") as f:
            f.write(f"{volume}\t{resolution}")

        # Run the Docker command
        # Define Docker image and script path
        docker_image = "quay.io/dolfinadjoint/pyadjoint:master"
        script_path = "/home/fenics/shared/templates/initialize_design2d.py"
        current_dir = os.getcwd()
        docker_command = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{current_dir}:/home/fenics/shared",
            docker_image,
            "python3",
            script_path,
        ]

        try:
            subprocess.run(docker_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing Docker command: {e}")
            return None

        # Load the generated design data from the numpy file
        design_file = f"templates/initialize_design/initial_v={volume}_resol={resolution}_.npy"
        if not os.path.exists(design_file):
            print(f"Error: Design file {design_file} not found.")
            return None
        file_npy = np.load(design_file)

        return file_npy

    def simulate(
        self,
        design: np.ndarray | None = None,
        volume: float | None = None,
        length: float | None = None,
        resolution: int | None = None,
    ) -> npt.NDArray:
        """Simulate the design.

        Args:
            design (Optional[np.ndarray]): The design to simulate.
            volume (Optional[float]): Volume constraint.
            length (Optional[float]): Length constraint.
            resolution (Optional[int]): Resolution of the design space.

        Returns:
            float: The thermal compliance of the design.
        """
        volume = volume if volume is not None else self.volume
        length = length if length is not None else self.length
        resolution = resolution if resolution is not None else self.resolution
        if design is None:
            des = self.initialize_design(volume, length, resolution)
            design = des[:, 2].reshape(resolution + 1, resolution + 1)
        with open("templates/sim_var.txt", "w") as f:
            f.write(f"{volume}\t{length}\t{resolution}")

        filename = "templates/hr_data_v=" + str(volume) + "_w=" + str(length) + "_.npy"
        np.save(filename, design)
        docker_image = "quay.io/dolfinadjoint/pyadjoint:master"
        script_path = "/home/fenics/shared/templates/simulateHeatconduction2d.py"
        current_dir = os.getcwd()
        docker_command = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{current_dir}:/home/fenics/shared",
            docker_image,
            "python3",
            script_path,
        ]
        # Execute the Docker command
        try:
            subprocess.run(docker_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing Docker command: {e}")
            return None

        with open(r"templates/RES_SIM/Performance.txt") as fp:
            self.PERF = fp.read()
        return np.array(self.PERF)

    def optimize(
        self,
        design: np.ndarray | None = None,
        volume: float | None = None,
        length: float | None = None,
        resolution: int | None = None,
    ) -> tuple:
        """Optimizes the design.

        Args:
            design (Optional[np.ndarray]): The design to optimize.
            volume (Optional[float]): Volume constraint.
            length (Optional[float]): Length constraint.
            resolution (Optional[int]): Resolution of the design space.

        Returns:
            Tuple[OptimalDesign, list[OptiStep]]: The optimized design and the optimization history.
        """
        volume = volume if volume is not None else self.volume
        length = length if length is not None else self.length
        resolution = resolution if resolution is not None else self.resolution
        if design is None:
            des = self.initialize_design(volume, length, resolution)
            design = des[:, 2].reshape(resolution + 1, resolution + 1)

        with open("templates/OPT_var.txt", "w") as f:
            f.write(f"{volume}\t{length}\t{resolution}")

        filename = "templates/hr_data_OPT_v=" + str(volume) + "_w=" + str(length) + "_.npy"
        np.save(filename, design)
        docker_image = "quay.io/dolfinadjoint/pyadjoint:master"
        script_path = "/home/fenics/shared/templates/optimizeHeatconduction2d.py"
        current_dir = os.getcwd()
        docker_command = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{current_dir}:/home/fenics/shared",
            docker_image,
            "python3",
            script_path,
        ]
        # Execute the Docker command
        try:
            subprocess.run(docker_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing Docker command: {e}")
            return None
        output = np.load("templates/RES_OPT/OUTPUT=" + str(volume) + "_w=" + str(length) + "_.npz")

        return (output["design"], output["OptiStep"])

    def render(self, design: np.ndarray, open_window: bool = False) -> Any:
        """Renders the design in a human-readable format.

        Args:
            design (np.ndarray): The design to render.
            open_window (bool): If True, opens a window with the rendered design.

        Returns:
            Any: The rendered design.
        """
        if design is None:
            des = self.initialize_design()
            design = des.npy[:, 2].reshape(self.resolution + 1, self.resolution + 1)
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
    design_values = np.random.rand(100, 100)
    problem.render(design_values, open_window=False)
    print(problem.simulate())
