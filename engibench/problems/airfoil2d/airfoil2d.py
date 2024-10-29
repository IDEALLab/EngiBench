"""Airfoil 2D problem.

This problem simulates the performance of an airfoil in a 2D environment. An airfoil is represented by a set of 192 points that define its shape. The performance is evaluated by the MACHAERO simulator that computes the lift and drag coefficients of the airfoil.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, overload

from datasets import load_dataset
from gymnasium import spaces
import numpy as np
import pyoptsparse

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
    possible_objectives: frozenset[[str, str]] = frozenset(
        {
            ("lift", "maximize"),
            ("drag", "minimize"),
        }
    )
    design_space = spaces.Box(low=0.0, high=1.0, shape=(2, 192), dtype=np.float32)
    dataset_id = "ffelten/test2"
    container_id = "mdolab/public:u22-gcc-ompi-stable"

    def __init__(self, objectives: tuple[str, str] = ("lift", "drag")) -> None:
        super().__init__()
        self.seed = None
        # docker pull image
        subprocess.run(["docker", "pull", self.container_id], check=True)

        # This is used for intermediate files
        self.__common_dir = "engibench/problems/airfoil2d/"
        self.__template_dir = self.__common_dir + "templates"
        self.__study_dir = self.__common_dir + "study"
        self.current_study_dir = self.__study_dir + f"_{self.seed}/"

        self.objectives: set[str] = set(objectives)
        self.dataset = load_dataset(self.dataset_id, split="train")

    def reset(self, seed: int | None = None, *, cleanup: bool = False) -> None:
        # Cleanup the previous study directory -- requires seed number
        if cleanup:
            shutil.rmtree(self.__study_dir + f"_{self.seed}")

        super().reset(seed)
        self.current_study_dir = self.__study_dir + f"_{self.seed}/"
        clone_template(template_dir=self.__template_dir, study_dir=self.current_study_dir)

    def __design_to_simulator_input(self, design: np.ndarray, filename: str = "design") -> str:
        """Converts a design to a simulator input.

        The simulator inputs are two files: a mesh file (.cgns) and a FFD file (.xyz). This function generates these files from the design.
        The files are saved in the current directory with the name "$filename.cgns" and "$filename_ffd.xyz".

        Args:
            design (np.ndarray): The design to convert.
            filename (str): The filename to save the design to.
        """
        d
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
    def simulate(self, design: np.ndarray, config: dict[str, Any] = {}, mpicores: int = 4) -> dict[str, float]:
        # pre-process the design and run the simulation
        filename = "candidate_design"
        self.__design_to_simulator_input(design, filename)

        # Prepares the airfoil_analysis.py script with the simulation configuration
        base_config = {  # TODO Cashen Check those default values (in optimize too)
            "alpha": 1.5,
            "mach": 0.8,
            "reynolds": 1e6,
            "altitude": 10000,
            "temperature": 1.0,
            "output_dir": "'" + self.current_study_dir + "output/'",
            "mesh_fname": "'" + self.current_study_dir + filename + ".cgns'",
            "task": "'analysis'",
        }

        base_config.update(config)
        replace_template_values(
            self.current_study_dir + "/airfoil_analysis.py",
            base_config,
        )

        # Launches a docker container with the airfoil_analysis.py script
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
            "/home/mdolabuser/mount/engibench/engibench/problems/airfoil2d/analyze.sh",
            str(mpicores),
            self.current_study_dir,
        ]

        subprocess.run(command, check=True)
        return {"lift": 0.0, "drag": 0.0}

    def optimize(
        self, starting_point: np.ndarray, config: dict[str, Any], mpicores: int = 4
    ) -> tuple[Any, dict[str, float]]:
        # pre-process the design and run the simulation
        filename = "candidate_design"
        self.__design_to_simulator_input(starting_point, filename)

        # Prepares the optimize_airfoil.py script with the optimization configuration
        base_config = {
            "cl": 0.5,
            "alpha": 1.5,
            "mach": 0.75,
            "altitude": 10000,
            "opt": "'SLSQP'",
            "opt_options": {},
            "output_dir": "'" + self.current_study_dir + "output/'",
            "ffd_fname": "'" + self.current_study_dir + filename + "_ffd.xyz'",
            "mesh_fname": "'" + self.current_study_dir + filename + ".cgns'",
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

        # post process -- extract the shape and objective values
        history = pyoptsparse.History(self.current_study_dir + "output/opt.hst")
        objective = history.getValues(names=["obj"], callCounters=None, allowSens=False, major=False, scale=True)["obj"][
            -1, -1
        ]

        return starting_point, {"obj": objective}


if __name__ == "__main__":
    problem = Airfoil2D()
    problem.reset(seed=0, cleanup=False)
    dataset = problem.dataset

    first_design = np.array(dataset["features"][0])  # type: ignore
    # print("Design: ", first_design)
    # print("Shape: ", first_design.shape)
    print(problem.__design_to_simulator_input(first_design, filename="initial_design"))
    # print(problem.optimize(starting_point="initial_design", config={}, mpicores=8))
    print(problem.simulate("initial_design", config={}, mpicores=8))
    # history = pyoptsparse.History(problem._current_study_dir + "output/opt.hst")
    # print(history.getObjNames())
    # print(history.getValues(names=["obj"], callCounters=None, allowSens=True, major=False, scale=True)["obj"][-1, -1])

    # Get the file named fc_x_slices.dat with the highest x value
    # last_slice_file = None
    # largest_x = -1
    # for file in os.listdir(problem._current_study_dir + "output/"):
    #     if file.startswith("fc_") and file.endswith("_slices.dat"):
    #         x = int(file.split("_")[1])
    #         if last_slice_file is None or x > largest_x:
    #             last_slice_file = problem._current_study_dir + "output/" + file
    #             largest_x = x
    # var_names = [
    #     "CoordinateX",
    #     "CoordinateY",
    #     "CoordinateZ",
    #     "XoC",
    #     "YoC",
    #     "ZoC",
    #     "VelocityX",
    #     "VelocityY",
    #     "VelocityZ",
    #     "CoefPressure",
    #     "Mach",
    # ]
    # print("Last slice file: ", last_slice_file)
    # # TODO see with Cashen how to extract shape from the file and convert to his format
    # print(
    #     pd.read_csv(
    #         last_slice_file, sep=r"\s+", names=["fill1", "Nodes", "fill2", "Elements", "ZONETYPE"], skiprows=3, nrows=1
    #     )
    # )
    # slice = pd.read_csv(last_slice_file, sep=r"\s+", names=var_names, skiprows=5)[["CoordinateX", "CoordinateY"]]
    # slice = pd.read_csv(last_slice_file, sep=r"\s+", names=["NodeC1", "NodeC2"], skiprows=5)
    # print(slice)
