"""Airfoil 2D problem.

This problem simulates the performance of an airfoil in a 2D environment. An airfoil is represented by a set of 192 points that define its shape. The performance is evaluated by a simulator that computes the lift and drag coefficients of the airfoil.
"""

import os
import subprocess
from typing import Any, ClassVar

from datasets import load_dataset
from gymnasium import spaces
import numpy as np

from engibench.core import Problem


class Airfoil2D(Problem):
    r"""Airfoil 2D problem.

    This problem simulates the performance of an airfoil in a 2D environment. The airfoil is represented by a set of 192 points that define its shape. The performance is evaluated by a simulator that computes the lift and drag coefficients of the airfoil.

    The design space is represented by a 3D numpy array (vector of 192 x,y coordinates per design) that define the airfoil shape.

    TODO complete documentation
    """

    input_space = str
    possible_objectives: ClassVar[dict[str, str]] = {
        "lift": "maximize",
        "drag": "minimize",
    }
    design_space = spaces.Box(low=0.0, high=1.0, shape=(2, 192), dtype=np.float32)
    dataset_id = "ffelten/test2"

    def __init__(self, objectives: tuple[str, str] = ("lift", "drag")) -> None:
        """Initialize the problem."""
        super().__init__()
        self.reset()
        self.objectives: set[str] = set(objectives)
        self.dataset = load_dataset(self.dataset_id, split="train")

    def design_to_simulator_input(self, design: np.ndarray, filename: str = "design") -> str:
        """Converts a design to a simulator input.

        The simulator inputs are two files: a mesh file (.cgns) and a FFD file (.xyz). This function generates these files from the design.
        The files are saved in the current directory with the name "$filename.cgns" and "$filename_ffd.xyz".

        Args:
            design (np.ndarray): The design to convert.
            filename (str): The filename to save the design to.
        """
        # TODO decide on this
        with open("tmp.npy", "wb") as f:
            np.save(f, design)

        # Launches a docker container with the pre_process.py script
        # The script generates the mesh and FFD files
        # Bash command:
        command = [
            "docker",
            "run",
            "-it",
            "--rm",
            "--name",
            "machaero",
            "--mount",
            f"type=bind,src={os.getcwd()},target=/home/mdolabuser/mount/engibench",
            "mdolab/public:u22-gcc-ompi-stable",
            "/bin/bash",
            "/home/mdolabuser/mount/engibench/engibench/problems/airfoil2d/pre_process.sh",
            "n0012.dat", # TODO when decided on input, change this
            filename,
        ]

        subprocess.run(command, check=True)
        return filename

    def optimize(self, starting_point: str, conditions: dict[str, Any]) -> tuple[Any, dict[str, float]]:
        """Optimize the design starting from `starting_point`.

        It assumes the designs are saved in the current directory with the name "$starting_point.cgns" and "$starting_point_ffd.xyz".
        """
        # TODO check if we need to propagate the conditions
        cond = conditions
        # Launches a docker container with the optimize_airfoil.py script
        # The script takes a mesh and ffd and performs an optimization
        # Bash command:
        command = [
            "docker",
            "run",
            "-it",
            "--rm",
            "--name",
            "machaero",
            "--mount",
            f"type=bind,src={os.getcwd()},target=/home/mdolabuser/mount/engibench",
            "mdolab/public:u22-gcc-ompi-stable",
            "/bin/bash",
            "/home/mdolabuser/mount/engibench/engibench/problems/airfoil2d/optimize.sh",
            starting_point + ".cgns",
            starting_point + "_ffd.xyz",
        ]

        subprocess.run(command, check=True)
        # TODO extract the design and performance from the output files
        return (starting_point, {"lift": 0.0, "drag": 0.0})


if __name__ == "__main__":
    problem = Airfoil2D()
    problem.reset(seed=0)
    dataset = problem.dataset

    first_design = np.array(dataset["features"][0])  # type: ignore
    print("Design: ", first_design)
    print("Shape: ", first_design.shape)
    print(problem.design_to_simulator_input(first_design, "initial_design"))
    print(problem.optimize("initial_design", {}))
