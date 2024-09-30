"""Airfoil 2D problem.

This problem simulates the performance of an airfoil in a 2D environment. An airfoil is represented by a set of 192 points that define its shape. The performance is evaluated by a simulator that computes the lift and drag coefficients of the airfoil.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, ClassVar, overload

from datasets import load_dataset
from gymnasium import spaces
import numpy as np

from engibench.core import Problem
from engibench.utils.files import clone_template
from engibench.utils.files import replace_template_values


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
    container_id = "mdolab/public:u22-gcc-ompi-stable"

    common_dir = "engibench/problems/airfoil2d/"
    template_dir = common_dir + "templates"
    study_dir = common_dir + "study"

    def __init__(self, objectives: tuple[str, str] = ("lift", "drag")) -> None:
        """Initialize the problem."""
        super().__init__()
        self.objectives: set[str] = set(objectives)
        self.dataset = load_dataset(self.dataset_id, split="train")

    @overload
    def reset(self, seed: int | None = None, *, cleanup: bool = False) -> None:
        if cleanup:
            os.removedirs(self.study_dir + f"_{self.seed}")
        super().reset(seed)
        self.current_study_dir = self.study_dir + f"_{self.seed}/"
        clone_template(template_dir=self.template_dir, study_dir=self.current_study_dir)

    @overload
    def design_to_simulator_input(self, design: np.ndarray, filename: str = "design") -> str:
        """Converts a design to a simulator input.

        The simulator inputs are two files: a mesh file (.cgns) and a FFD file (.xyz). This function generates these files from the design.
        The files are saved in the current directory with the name "$filename.cgns" and "$filename_ffd.xyz".

        Args:
            design (np.ndarray): The design to convert.
            filename (str): The filename to save the design to.
        """
        d = design
        d = "n0012.dat"
        # Prepares the preprocess.py script with the design
        replace_template_values(
            self.current_study_dir + "/pre_process.py",
            {
                "design_fname": f"'{d}'",
                "tmp_xyz_fname": "'" + self.current_study_dir + "tmp.xyz'",
                "mesh_fname": "'" + self.current_study_dir + filename + ".cgns'",
                "ffd_fname": "'" + self.current_study_dir + filename + "_ffd.xyz'",
            },
        )

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
            self.container_id,
            "/bin/bash",
            "/home/mdolabuser/mount/engibench/engibench/problems/airfoil2d/pre_process.sh",
            self.current_study_dir,
        ]

        subprocess.run(command, check=True)
        return filename

    @overload
    def optimize(self, starting_point: str, config: dict[str, Any], mpicores: int = 4) -> tuple[Any, dict[str, float]]:
        """Optimize the design starting from `starting_point`.

        It assumes the designs are saved in the current directory with the name "$starting_point.cgns" and "$starting_point_ffd.xyz".
        """
        # Prepares the optimize_airfoil.py script with the optimization configuration
        base_config = {
            "cl": 0.5,
            "alpha": 1.5,
            "mach": 0.75,
            "altitude": 10000,
            "opt": "'SLSQP'",
            "opt_options": {},
            "output_dir": "'" + self.current_study_dir + "output/'",
            "ffd_fname": "'" + self.current_study_dir + starting_point + "_ffd.xyz'",
            "mesh_fname": "'" + self.current_study_dir + starting_point + ".cgns'",
        }
        base_config.update(config)

        replace_template_values(
            self.current_study_dir + "/airfoil_opt.py",
            base_config,
        )

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
            self.container_id,
            "/bin/bash",
            "/home/mdolabuser/mount/engibench/engibench/problems/airfoil2d/optimize.sh",
            str(mpicores),
            self.current_study_dir,
        ]

        subprocess.run(command, check=True)
        # TODO extract the design and performance from the output files
        return (starting_point, {"lift": 0.0, "drag": 0.0})


if __name__ == "__main__":
    problem = Airfoil2D()
    problem.reset(seed=0, cleanup=False)
    dataset = problem.dataset

    first_design = np.array(dataset["features"][0])  # type: ignore
    print("Design: ", first_design)
    print("Shape: ", first_design.shape)
    print(problem.design_to_simulator_input(first_design, filename="initial_design"))
    print(problem.optimize(starting_point="initial_design", config={}, mpicores=8))
