"""Airfoil 2D problem.

This problem simulates the performance of an airfoil in a 2D environment. An airfoil is represented by a set of 192 points that define its shape. The performance is evaluated by the MACHAERO simulator that computes the lift and drag coefficients of the airfoil.
The Dataset linked to this problem is hosted on the Hugging Face Datasets Hub and is called "IDEALLab/airfoil_2d".
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any

from datasets import load_dataset
from gymnasium import spaces
import numpy as np
import pandas as pd
import pyoptsparse

from engibench.core import Problem
from engibench.utils.files import clone_template
from engibench.utils.files import replace_template_values


class Airfoil2D(Problem):
    r"""Airfoil 2D problem.

    ## Problem Description
    This problem simulates the performance of an airfoil in a 2D environment. The performance is evaluated by a simulator that computes the lift and drag coefficients of the airfoil.

    ## Design space
    The design space is represented by a 3D numpy array (vector of 192 x,y coordinates in [0., 1.) per design) that define the airfoil shape.

    ## Dataset
    The dataset linked to this problem is hosted on the [Hugging Face Datasets Hub](https://huggingface.co/datasets/IDEALLab/airfoil_2d).

    ## Simulator
    The simulator is a docker container with the MACH-Aero software that computes the lift and drag coefficients of the airfoil.

    ## Lead
    Cashen Diniz @cashend
    """

    input_space = str
    possible_objectives: frozenset[tuple[str, str]] = frozenset(
        {
            ("lift", "maximize"),
            ("drag", "minimize"),
        }
    )
    design_space = spaces.Box(low=0.0, high=1.0, shape=(2, 192), dtype=np.float32)
    dataset_id = "IDEALLab/airfoil_2d"
    container_id = "mdolab/public:u22-gcc-ompi-stable"

    def __init__(self, objectives: tuple[str, str] = ("lift", "drag")) -> None:
        """Initializes the Airfoil2D problem."""
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
        """Resets the simulator and numpy random to a given seed.

        Args:
            seed (int, optional): The seed to reset to. If None, a random seed is used.
            cleanup (bool): Deletes the previous study directory if True.
        """
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
        # Save the design to a temporary file
        np.savetxt(self.current_study_dir + filename + ".dat", design.transpose())
        # Prepares the preprocess.py script with the design
        replace_template_values(
            self.current_study_dir + "/pre_process.py",
            {
                "design_fname": f"'{self.current_study_dir}{filename}.dat'",
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

    def simulator_output_to_design(self, simulator_output: str = None) -> np.ndarray:
        """Converts a simulator input to a design.

        Args:
            simulator_output (str): The simulator input to convert.

        Returns:
            np.ndarray: The corresponding design.
        """
        if simulator_output is None:
            # Take latest slice file
            files = os.listdir(self.current_study_dir + "/output")
            files = [f for f in files if f.endswith("_slices.dat")]
            file_numbers = [int(f.split("_")[1]) for f in files]
            simulator_output = files[file_numbers.index(max(file_numbers))]

        slice_file = self.current_study_dir + "/output/" + simulator_output

        # Define the variable names for columns
        var_names = [
            "CoordinateX",
            "CoordinateY",
            "CoordinateZ",
            "XoC",
            "YoC",
            "ZoC",
            "VelocityX",
            "VelocityY",
            "VelocityZ",
            "CoefPressure",
            "Mach",
        ]

        nelems = pd.read_csv(
            slice_file, sep=r"\s+", names=["fill1", "Nodes", "fill2", "Elements", "ZONETYPE"], skiprows=3, nrows=1
        )
        nnodes = int(nelems["Nodes"].iloc[0])

        # Read the main data and node connections
        slice_df = pd.read_csv(slice_file, sep=r"\s+", names=var_names, skiprows=5, nrows=nnodes, engine="c")
        # TODO Cashen is this necessary to extract the shape?
        #  nodes_arr = pd.read_csv(slice_file, sep=r"\s+", names=["NodeC1", "NodeC2"], skiprows=5 + nnodes, engine="c")

        # Concatenate node connections to the main data
        # slice_df = pd.concat([slice_df, nodes_arr], axis=1)

        design = slice_df[["CoordinateX", "CoordinateY"]].values.transpose()
        return design

    def simulate(self, design: np.ndarray, config: dict[str, Any] = {}, mpicores: int = 4) -> dict[str, float]:
        """Simulates the performance of an airfoil design.

        Args:
            design (np.ndarray): The design to simulate.
            config (dict): A dictionary with configuration (e.g., boundary conditions, filenames) for the simulation.
            mpicores (int): The number of MPI cores to use in the simulation.

        Returns:
            dict: The performance of the design - each entry of the dict corresponds to a named objective value.
        """
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
        self, starting_point: np.ndarray, config: dict[str, Any] = {}, mpicores: int = 4
    ) -> tuple[Any, dict[str, float]]:
        """Optimizes the design of an airfoil.

        Args:
            starting_point (np.ndarray): The starting point for the optimization.
            config (dict): A dictionary with configuration (e.g., boundary conditions, filenames) for the optimization.
            mpicores (int): The number of MPI cores to use in the optimization.

        Returns:
            Tuple[np.ndarray, dict]: The optimized design and its performance.
        """
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

        optimized_design = self.__simulator_output_to_design()

        return optimized_design, {"obj": objective}  # TODO Cashen check the objective name


if __name__ == "__main__":
    problem = Airfoil2D()
    problem.reset(seed=0, cleanup=False)

    dataset = problem.dataset

    # Get design and conditions from the dataset
    design = np.array(dataset["initial"][0])  # type: ignore
    config_keys = dataset.features.keys() - ["initial", "optimized"]
    config = {key: dataset[key][0] for key in config_keys}

    print(problem.optimize(design, config=config, mpicores=8))
