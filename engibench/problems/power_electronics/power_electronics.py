"""Airfoil 2D problem."""

from __future__ import annotations
from typing import Any

import os
import shutil
import subprocess
import numpy as np
import networkx as nx

from engibench.core import Problem
from engibench.utils.files import clone_template
from engibench.utils.files import replace_template_values


def build(**kwargs) -> PowerElectronics:
    """Builds an Power Electronics problem.

    Args:
        **kwargs: Arguments to pass to the constructor.
    """
    return PowerElectronics(**kwargs)


class PowerElectronics(Problem):
    r"""Power Electronics parameter optimization problem.

    ## Problem Description
    This problem tries to find the optimal circuit parameters for a given circuit topology. 
    The ngSpice simulator takes the topology and parameters as input and returns DcGain and Voltage Ripple.  Then Efficiency is calculated from
    As a single-objective optimization problem,

    ## Design space
    The design space is represented by a 3D numpy array (vector of 192 x,y coordinates in [0., 1.) per design) that define the airfoil shape.
    Dependent on the dataset. In specific, TODO:

    ## Objectives
    The objectives are defined by the following parameters:
    - `DcGain`: Ratio of load vs. input voltage. It's a preset constant and the simulation result should be as close to the constant as possible.
    - `Voltage Ripple`: Fluctuation of voltage on the load side. The lower the better. 
    - `Efficiency`: Range is (0, 1). The higher the better.

    ## Boundary conditions
    The boundary conditions are defined by the following parameters:
    N/A

    ## Dataset
    TODO: The dataset linked to this problem is hosted on the [Hugging Face Datasets Hub](https://huggingface.co/datasets/IDEALLab/airfoil_2d).
    Networkx objects + netlists.
    New simulation data points can be appended locally. 

    ## Simulator
    TODO: linux ngspice package or Windows ngspice.exe?

    ## Lead
    Xuliang Dong @ liangXD523
    """

    input_space = str
    possible_objectives: frozenset[tuple[str, str]] = frozenset(
        {
            ("cd", "minimize"),
            ("cl", "maximize"),
        }
    )
    boundary_conditions: frozenset[tuple[str, Any]] = frozenset(
        {
            ("s0", 3e-6),
            ("marchDist", 100.0),
        }
    )
    # design_space = spaces.Box(low=0.0, high=1.0, shape=(2, 192), dtype=np.float32)  #TODO
    dataset_id = "IDEALLab/airfoil_2d"
    # container_id = "mdolab/public:u22-gcc-ompi-stable"
    _dataset = None

    def __init__(self) -> None:
        """Initializes the Power Electronics problem."""
        super().__init__()
        self.seed = None

        # This is used for intermediate files
        self.__common_dir = "engibench/problems/power_electronics/"
        self.__template_dir = self.__common_dir + "templates"
        self.__study_dir = self.__common_dir + "study"
        # self.current_study_dir = self.__study_dir + f"_{self.seed}/"

    # def reset(self, seed: int | None = None, *, cleanup: bool = False) -> None:
    #     """Resets the simulator and numpy random to a given seed.

    #     Args:
    #         seed (int, optional): The seed to reset to. If None, a random seed is used.
    #         cleanup (bool): Deletes the previous study directory if True.
    #     """
    #     # docker pull image if not already pulled
    #     subprocess.run(["docker", "pull", self.container_id], check=True)
    #     if cleanup:
    #         shutil.rmtree(self.__study_dir + f"_{self.seed}")

    #     super().reset(seed)
    #     self.current_study_dir = self.__study_dir + f"_{self.seed}/"
    #     clone_template(template_dir=self.__template_dir, study_dir=self.current_study_dir)

    def __design_to_simulator_input(self, design: np.ndarray, filename: str = "design") -> str:
        """Combine the networkx and parameter vector into the simulator input .net file.

        The simulator inputs are two files: a mesh file (.cgns) and a FFD file (.xyz). This function generates these files from the design.
        The files are saved in the current directory with the name "$filename.cgns" and "$filename_ffd.xyz".

        Args:
            design (np.ndarray): The design to convert.
            filename (str): The filename to save the design to.
        """
        # Save the design to a temporary file
        np.savetxt(self.current_study_dir + filename + ".dat", design.transpose())

        base_config = {
            "design_fname": f"'{self.current_study_dir}{filename}.dat'",
            "tmp_xyz_fname": "'" + self.current_study_dir + "tmp'",
            "mesh_fname": "'" + self.current_study_dir + filename + ".cgns'",
            "ffd_fname": "'" + self.current_study_dir + filename + "_ffd'",
            "N_sample": 180,
            "nTEPts": 4,
            "xCut": 0.99,
            "ffd_ymarginu": 0.02,
            "ffd_ymarginl": 0.02,
            "ffd_pts": 10,
            "N_grid": 100,
        }
        # Adds the boundary conditions to the configuration
        base_config.update(self.boundary_conditions)

        # Prepares the preprocess.py script with the design
        replace_template_values(
            self.current_study_dir + "/pre_process.py",
            base_config,
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

    # def __simulator_output_to_design(self, simulator_output: str = None) -> np.ndarray:
    #     """Converts a simulator input to a design.

    #     Args:
    #         simulator_output (str): The simulator input to convert.

    #     Returns:
    #         np.ndarray: The corresponding design.
    #     """
    #     if simulator_output is None:
    #         # Take latest slice file
    #         files = os.listdir(self.current_study_dir + "/output")
    #         files = [f for f in files if f.endswith("_slices.dat")]
    #         file_numbers = [int(f.split("_")[1]) for f in files]
    #         simulator_output = files[file_numbers.index(max(file_numbers))]

    #     slice_file = self.current_study_dir + "/output/" + simulator_output

    #     return design

    def simulate(self, design: np.ndarray, config: dict[str, Any] = {}, mpicores: int = 4) -> dict[str, float]:
        """Simulates the performance of a Power Electronics design.

        Args:
            design (np.ndarray): The design to simulate.

        Returns:
            dict: The performance of the design - each entry of the dict corresponds to a named objective value.
        """
        # pre-process the design and run the simulation
        self.__design_to_simulator_input(design)

        # Prepares the airfoil_analysis.py script with the simulation configuration
        base_config = {
            "alpha": 1.5,
            "mach": 0.8,
            "reynolds": 1e6,
            "altitude": 10000,
            "temperature": 223.150,  # should specify either mach + altitude or mach + reynolds + reynoldsLength (default to 1) + temperature
            "output_dir": "'" + self.current_study_dir + "output/'",
            "mesh_fname": "'" + self.current_study_dir + "design.cgns'",
            "task": "'analysis'",  # TODO: We can add the option to perform a polar analysis.
        }

        base_config.update(self.boundary_conditions)
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

        outputs = np.load(self.current_study_dir + "output/outputs.npy")
        lift = float(outputs[3])
        drag = float(outputs[4])
        return {"cd": drag, "cl": lift}

    def optimize(self):
        return NotImplementedError
    # def optimize(
    #     self, starting_point: np.ndarray, config: dict[str, Any] = {}, mpicores: int = 4
    # ) -> tuple[Any, dict[str, float]]:
        # """Optimizes the design of an airfoil.

        # Args:
        #     starting_point (np.ndarray): The starting point for the optimization.
        #     config (dict): A dictionary with configuration (e.g., boundary conditions, filenames) for the optimization.
        #     mpicores (int): The number of MPI cores to use in the optimization.

        # Returns:
        #     Tuple[np.ndarray, dict]: The optimized design and its performance.
        # """
        # # pre-process the design and run the simulation
        # filename = "candidate_design"
        # self.__design_to_simulator_input(starting_point, filename)

        # # Prepares the optimize_airfoil.py script with the optimization configuration
        # base_config = {
        #     "cl": 0.5,
        #     "alpha": 1.5,
        #     "mach": 0.75,
        #     "altitude": 10000,  # add temperature (default 223.150) + reynolds (default 6.31E6) + reynoldsLength (default 1) option
        #     "opt": "'SLSQP'",
        #     "opt_options": {},
        #     "output_dir": "'" + self.current_study_dir + "output/'",
        #     "ffd_fname": "'" + self.current_study_dir + filename + "_ffd.xyz'",
        #     "mesh_fname": "'" + self.current_study_dir + filename + ".cgns'",
        # }

        # base_config.update(self.boundary_conditions)
        # base_config.update(config)

        # replace_template_values(
        #     self.current_study_dir + "/airfoil_opt.py",
        #     base_config,
        # )

        # # Launches a docker container with the optimize_airfoil.py script
        # # The script takes a mesh and ffd and performs an optimization
        # # Bash command:
        # command = [
        #     "docker",
        #     "run",
        #     "-it",
        #     "--rm",
        #     "--name",
        #     "machaero",
        #     "--mount",
        #     f"type=bind,src={os.getcwd()},target=/home/mdolabuser/mount/engibench",
        #     self.container_id,
        #     "/bin/bash",
        #     "/home/mdolabuser/mount/engibench/engibench/problems/airfoil2d/optimize.sh",
        #     str(mpicores),
        #     self.current_study_dir,
        # ]

        # subprocess.run(command, check=True)

        # # post process -- extract the shape and objective values
        # history = pyoptsparse.History(self.current_study_dir + "output/opt.hst")
        # objective = history.getValues(names=["obj"], callCounters=None, allowSens=False, major=False, scale=True)["obj"][
        #     -1, -1
        # ]

        # optimized_design = self.__simulator_output_to_design()

        # return optimized_design, {"cd": objective}


if __name__ == "__main__":
    problem = PowerElectronics()
    problem.reset(seed=0, cleanup=False)

    dataset = problem.dataset

    # Get design and conditions from the dataset
    design = np.array(dataset["initial"][0])  # type: ignore
    config_keys = dataset.features.keys() - ["initial", "optimized"]
    config = {key: dataset[key][0] for key in config_keys}

    print(problem.optimize(design, config=config, mpicores=8))
