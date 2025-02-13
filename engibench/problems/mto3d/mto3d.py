"""MTO 3D problem.

Filename convention is that folder paths do not end with /. For example, /path/to/folder is correct, but /path/to/folder/ is not.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Optional

from gymnasium import spaces
import numpy as np
import numpy.typing as npt
import pandas as pd

from engibench.core import DesignType
from engibench.core import OptiStep
from engibench.core import Problem
from engibench.utils.files import clone_dir
from engibench.utils.files import replace_template_values
from .utils import xh_npy, warm_up

def build(**kwargs) -> MTO3D:
    """Builds an MTO3D problem.

    Args:
        **kwargs: Arguments to pass to the constructor.
    """
    return MTO3D(**kwargs)


class MTO3D(Problem):
    r"""MTO 3D heat sink optimization problem.

    ## Problem Description
    This problem simulates the performance of an airfoil in a 2D environment. An airfoil is represented by a set of 192 points that define its shape. The performance is evaluated by the [MACH-Aero](https://mdolab-mach-aero.readthedocs-hosted.com/en/latest/) simulator that computes the lift and drag coefficients of the airfoil.

    ## Design space
    The design space is represented by a 3D numpy array (vector of 192 x,y coordinates in [0., 1.) per design) that define the airfoil shape.

    ## Objectives
    The objectives are defined and indexed as follows:
    0. `mnt`: Mean temperature to minimize.

    ## Boundary conditions
    The boundary conditions are defined by the following parameters:
    - `v0`: Absolute value of the inlet velocity.
    - `pwd`: Power dissipation upper bound constraint.
    - `vol`: Fluid volume fraction upper bound constraint.

    ## Dataset
    The dataset linked to this problem is hosted on the [Hugging Face Datasets Hub](https://huggingface.co/datasets/IDEALLab/mto_3d).

    ## Simulator
    The simulator is a docker container with the MACH-Aero software that computes the lift and drag coefficients of the airfoil.

    ## Lead
    Qiuyi Chen @qiuyi
    """

    input_space = str
    possible_objectives: tuple[tuple[str, str]] = (
        ("mnt", "minimize"),
    )
    boundary_conditions: frozenset[tuple[str, Any]] = frozenset(
        {
            ("PowerDissMax", 20), 
            ("voluse", 0.5),
            ("U", 0.15),
        }
    )
    design_space = spaces.Box(low=0.0, high=1.0, shape=(20, 100, 100), dtype=np.float32)
    dataset_id = "IDEALLab/airfoil_2d_v0"
    container_id = "mdolab/public:u22-gcc-ompi-stable"
    _dataset = None

    def __init__(self, base_directory: str | None = None) -> None:
        """Initializes the Airfoil2D problem.

        Args:
            base_directory (str, optional): The base directory for the problem. If None, the current directory is selected.
        """
        super().__init__()
        self.seed = None

        self.current_study = f"study_{self.seed}"
        # This is used for intermediate files
        # Local file are prefixed with self.local_base_directory
        if base_directory is not None:
            self.__local_base_directory = base_directory
        else:
            self.__local_base_directory = os.getcwd()
        self.__local_target_dir = self.__local_base_directory + "/engibench_studies/problems/mto3d"
        self.__local_template_dir = (
            os.path.dirname(os.path.abspath(__file__)) + "/templates"
        )  # These templates are shipped with the lib
        self.__local_scripts_dir = os.path.dirname(os.path.abspath(__file__)) + "/scripts"
        self.__local_study_dir = self.__local_target_dir + "/" + self.current_study

    def reset(self, seed: int | None = None, *, cleanup: bool = False) -> None:
        """Resets the simulator and numpy random to a given seed.

        Args:
            seed (int, optional): The seed to reset to. If None, a random seed is used.
            cleanup (bool): Deletes the previous study directory if True.
        """
        # docker pull image if not already pulled
        subprocess.run(["docker", "pull", self.container_id], check=True)
        if cleanup:
            shutil.rmtree(self.__local_study_dir)

        super().reset(seed)
        self.current_study = f"study_{self.seed}"
        self.__local_study_dir = self.__local_target_dir + "/" + self.current_study
        self.__docker_study_dir = self.__docker_target_dir + "/" + self.current_study

        clone_dir(source_dir=self.__local_template_dir, target_dir=self.__local_study_dir)
        clone_dir(source_dir=self.__local_scripts_dir, target_dir=self.__local_target_dir)

    def __design_to_simulator_input(self, design: npt.NDArray) -> str:
        """Converts a design to a simulator input.

        The simulator inputs are two files: a mesh file (.cgns) and a FFD file (.xyz). This function generates these files from the design.
        The files are saved in the current directory with the name "$filename.cgns" and "$filename_ffd.xyz".

        Args:
            design (np.ndarray): The design to convert. Should have shape [20, 100, 200] (symmetric) or [20, 100, 100] (one-side).
        """
        # Save the design to a temporary file
        if design.shape == (20, 100, 100):
            design = np.concatenate(
                [design, np.flip(design, axis=-1)], 
                axis=-1)
        xh_npy.npy_to_xh(tensor=design,
                         path=self.__local_study_dir + '/0/')

    def __simulator_output_to_design(self, **kwargs) -> npt.NDArray:
        """Converts a simulator output to a design.

        Args:
            simulator_output (str): The simulator output to convert.

        Returns:
            np.ndarray: The corresponding design.
        """
        iters = np.loadtxt(os.path.join(self.__local_study_dir, 'Time.txt'), delimiter='\t')
        final_iter = iters[-1, 0]

        design = xh_npy.xh_to_npy(os.path.join(self.__local_study_dir, final_iter, 'xh.gz'))
        return design

    def __read_history(self, config, **kwargs) -> npt.ArrayLike:
        files = ['Time.txt', 'MeanT.txt', 'PowerDiss.txt', 'Voluse.txt']
        his = np.stack([np.loadtxt(os.path.join(self.__local_study_dir, file))[:, -1] for file in files], axis=-1)
        his[:, -1] += config['voluse']
        return his

    def simulate(self, design: npt.NDArray, config: dict[str, Any] = {}, mpicores: int = 4) -> npt.NDArray:
        raise NotImplementedError("The OpenFOAM optimizer hasn't been adapted for simulation yet.")

    def optimize(
        self, starting_point: Optional[npt.NDArray] = None, config: dict[str, Any] = {}, mpicores: int = 4
    ) -> tuple[np.ndarray, np.ndarray]:
        """Optimizes the design of an airfoil.

        Args:
            starting_point (np.ndarray): The starting point for the optimization.
            config (dict): A dictionary with configuration (e.g., boundary conditions, filenames) for the optimization.
            mpicores (int): The number of MPI cores to use in the optimization.

        Returns:
            Tuple[np.ndarray, dict]: The optimized design and its performance.
        """
        # pre-process the design and run the simulation
        if starting_point is not None:
            self.__design_to_simulator_input(starting_point)

        # Prepares the optimize_airfoil.py script with the optimization configuration
        base_config = {
            "PowerDissMax": None, 
            "voluse": None,
            "U": None,
            "continuation": True, 
            "qU": (None, None),
            "alphaMax": (None, None),
            "Heaviside": (None, None),
            "n_beginning": None,
            "n_levels": None,
            "n_transitionItersAtEachLevel": None,
            "n_restPeriodAtEachLevel": None,
            "n_maxIterAtEach": None,
            "n_consecutiveConverged": None,
            'cdr': self.__docker_study_dir,
            'sif': 'MTO_GEN.sif',
            'ntasks': 16,
        }

        base_config.update(self.boundary_conditions)
        base_config.update(config)

        assert os.path.exists(self.__local_study_dir)
        warm_up.modify_optProperties(
            self.__local_study_dir,
            **base_config
        )
        warm_up.replace_src(self.__local_study_dir)
        
        
        template_sh = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'templates', 'warm-start_3D.sh'
        )
        target_sh = os.path.join(
            self.__local_study_dir, 
            'warm-start_3D.sh'
        )
        shutil.copy(template_sh, target_sh)
        replace_template_values(target_sh, base_config)

        command = [
            "bash",
            target_sh
        ]

        subprocess.run(command, check=True)

        # post process -- extract the shape and objective values
        optisteps_history = self.__read_history(base_config)
        optimized_design = self.__simulator_output_to_design()

        return (optimized_design, optisteps_history)

    def render(self, design: np.ndarray, open_window: bool = False) -> Any:  # noqa: ANN401
        """Renders the design in a human-readable format.
        The current implementation uses matplotlib voxel plot to render the design.

        Args:
            design (np.ndarray): The design to render.
            open_window (bool): If True, opens a window with the rendered design.

        Returns:
            Any: The rendered design.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        def plot_vol3d(ax, vol_data, eps=0.5): # vol_data shape (z, h, w)
            vol = np.transpose(np.where(vol_data > eps, 1, 0), (2, 1, 0))
            z = np.empty_like(vol, dtype=np.float32)
            z[:] = np.linspace(0, 1, z.shape[-1])
            colors = cm.coolwarm_r(z)

            ax.voxels(vol, edgecolor='k', facecolors=colors, linewidth=0.05)
            ax.set_box_aspect(vol.shape, zoom=1.1)
            ax.set_axis_off()
            ax.view_init(elev=50, azim=90)

        fig = plt.figure(figsize=(5*2, 5))
        ax = fig.add_subplot(111, projection='3d')
        plot_vol3d(ax, design)

        if open_window:
            plt.show()
        return fig, ax

    def random_design(self) -> DesignType:
        """Samples a valid random design.

        Returns:
            DesignType: The valid random design.
        """
        rnd = self.np_random.integers(low=0, high=len(self.dataset["train"]["initial"]))  # pyright: ignore[reportOptionalMemberAccess]
        return np.array(self.dataset["train"]["initial"][rnd])  # type: ignore


if __name__ == "__main__":
    problem = MTO3D()
    problem.reset(seed=0, cleanup=False)

    dataset = problem.dataset

    # Get design and conditions from the dataset
    design = problem.random_design()
    fig, ax = problem.render(design, open_window=True)
    fig.savefig(
        "airfoil.png",
        dpi=300,
    )
