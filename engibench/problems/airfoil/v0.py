"""Airfoil 2D problem.

Filename convention is that folder paths do not end with /. For example, /path/to/folder is correct, but /path/to/folder/ is not.
"""

from __future__ import annotations

import os
import shutil
from typing import Any

from gymnasium import spaces
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyoptsparse

from engibench.core import ObjectiveDirection
from engibench.core import OptiStep
from engibench.core import Problem
from engibench.utils import container
from engibench.utils.files import clone_dir
from engibench.utils.files import replace_template_values

DesignType = dict[str, Any]


class Airfoil(Problem[DesignType]):
    r"""Airfoil 2D shape optimization problem.

    ```{note}
    This problem requires `gcc` and `gfortran` to be installed. See the simulator section for more details.
    ```

    ## Problem Description
    This problem simulates the performance of an airfoil in a 2D environment. An airfoil is represented by a set of 192 points that define its shape. The performance is evaluated by the [MACH-Aero](https://mdolab-mach-aero.readthedocs-hosted.com/en/latest/) simulator that computes the lift and drag coefficients of the airfoil.

    ## Design space
    The design space is represented by a dictionary where one element (`coords`) is a numpy array (vector of 192 x,y coordinates in `[0., 1.)` per design) that define the airfoil shape, and the other element (`angle_of_attack`) is a scalar.

    ## Objectives
    The objectives are defined and indexed as follows:

    0. `cd`: Drag coefficient to minimize.

    ## Conditions
    The conditions are defined by the following parameters:
    - `mach`: Mach number.
    - `reynolds`: Reynolds number.
    - `area_ratio_min`: Minimum area ratio (ratio relative to initial area) constraint.
    - `area_initial`: Initial area.
    - `cl_target`: Target lift coefficient to satisfy equality constraint.

    ## Simulator
    The simulator is a docker container with the MACH-Aero software that computes the lift and drag coefficients of the airfoil. You can install gcc and gfortran on your system with your package manager.
    - On Ubuntu: `sudo apt-get install gcc gfortran`
    - On MacOS: `brew install gcc gfortran`
    - On Windows (WSL): `sudo apt install build-essential`

    ## Dataset
    The dataset linked to this problem is hosted on the [Hugging Face Datasets Hub](https://huggingface.co/datasets/IDEALLab/airfoil_v0).

    ### v0

    #### Fields
    The dataset contains optimal design, conditions, objectives and these additional fields:
    - `initial_design`: Design before the adjoint optimization.
    - `cl_con_violation`: # Constraint violation for coefficient of lift.
    - `area_ratio`: # Area ratio for given design.

    #### Creation Method
    Refer to paper in references for details on how the dataset was created.

    ## References
    If you use this problem in your research, please cite the following paper:
    C. Diniz and M. Fuge, "Optimizing Diffusion to Diffuse Optimal Designs," in AIAA SCITECH 2024 Forum, 2024. doi: 10.2514/6.2024-2013.

    ## Lead
    Cashen Diniz @cashend
    """

    version = 0
    objectives: tuple[tuple[str, ObjectiveDirection], ...] = (("cd", ObjectiveDirection.MINIMIZE),)
    conditions: tuple[tuple[str, Any], ...] = (
        ("mach", 0.8),
        ("reynolds", 1e6),
        ("area_initial", None),
        ("area_ratio_min", 0.7),
        ("cl_target", 0.5),
    )

    design_space = spaces.Dict(
        {
            "coords": spaces.Box(low=0.0, high=1.0, shape=(2, 192), dtype=np.float32),
            "angle_of_attack": spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
        }
    )
    dataset_id = "IDEALLab/airfoil_v0"
    container_id = "mdolab/public:u22-gcc-ompi-stable"
    __local_study_dir: str

    def __init__(self, base_directory: str | None = None) -> None:
        """Initializes the Airfoil problem.

        Args:
            base_directory (str, optional): The base directory for the problem. If None, the current directory is selected.
        """
        # This is used for intermediate files
        # Local file are prefixed with self.local_base_directory
        if base_directory is not None:
            self.__local_base_directory = base_directory
        else:
            self.__local_base_directory = os.getcwd()
        self.__local_target_dir = self.__local_base_directory + "/engibench_studies/problems/airfoil"
        self.__local_template_dir = (
            os.path.dirname(os.path.abspath(__file__)) + "/templates"
        )  # These templates are shipped with the lib
        self.__local_scripts_dir = os.path.dirname(os.path.abspath(__file__)) + "/scripts"

        # Docker target directory
        # This is used for files that are mounted into the docker container
        self.__docker_base_dir = "/home/mdolabuser/mount/engibench"
        self.__docker_target_dir = self.__docker_base_dir + "/engibench_studies/problems/airfoil"

        super().__init__()

    def reset(self, seed: int | None = None, *, cleanup: bool = False) -> None:
        """Resets the simulator and numpy random to a given seed.

        Args:
            seed (int, optional): The seed to reset to. If None, a random seed is used.
            cleanup (bool): Deletes the previous study directory if True.
        """
        if cleanup:
            shutil.rmtree(self.__local_study_dir)

        super().reset(seed)
        self.current_study = f"study_{self.seed}"
        self.__local_study_dir = self.__local_target_dir + "/" + self.current_study
        self.__docker_study_dir = self.__docker_target_dir + "/" + self.current_study

        clone_dir(source_dir=self.__local_template_dir, target_dir=self.__local_study_dir)

    def __design_to_simulator_input(self, design: DesignType, config: dict[str, Any], filename: str = "design") -> str:
        """Converts a design to a simulator input.

        The simulator inputs are two files: a mesh file (.cgns) and a FFD file (.xyz). This function generates these files from the design.
        The files are saved in the current directory with the name "$filename.cgns" and "$filename_ffd.xyz".

        Args:
            design (dict): The design to convert.
            config (dict): A dictionary with configuration (e.g., boundary conditions) for the simulation.
            filename (str): The filename to save the design to.
        """
        # Scale the design to fit in the design space
        scaled_design = self._scale_coords(design["coords"])
        # Save the design to a temporary file
        np.savetxt(self.__local_study_dir + "/" + filename + ".dat", scaled_design.transpose())  # type: ignore  # noqa: PGH003
        tmp = os.path.join(self.__docker_study_dir, "tmp")

        base_config = {
            "design_fname": f"'{self.__docker_study_dir}/{filename}.dat'",
            "tmp_xyz_fname": f"'{tmp}'",
            "mesh_fname": "'" + self.__docker_study_dir + "/" + filename + ".cgns'",
            "ffd_fname": "'" + self.__docker_study_dir + "/" + filename + "_ffd'",
            "marchDist": 100.0,  # Distance to march the grid from the airfoil surface
            "N_sample": 180,
            "nTEPts": 4,
            "xCut": 0.99,
            "ffd_ymarginu": 0.05,
            "ffd_ymarginl": 0.05,
            "ffd_pts": 10,
            "N_grid": 100,
            "estimate_s0": True,
            "make_input_design_blunt": True,
            "input_blunted": False,
        }

        # Calculate the off-the-wall distance
        if base_config["estimate_s0"]:
            s0 = self._calc_off_wall_distance(
                mach=config["mach"], reynolds=config["reynolds"], freestreamTemp=config["temperature"]
            )
        else:
            s0 = 1e-5

        base_config["s0"] = s0

        # Adds the boundary conditions to the configuration
        base_config.update(self.conditions)

        # Scale the design to fit in the design space
        scaled_design, input_blunted = self._scale_coords(  # type: ignore  # noqa: PGH003
            design["coords"],
            blunted=base_config["input_blunted"],
            xcut=base_config["xCut"],  # type: ignore  # noqa: PGH003
        )
        base_config["input_blunted"] = input_blunted

        # Save the design to a temporary file. Format to 1e-6 rounding
        np.savetxt(self.__local_study_dir + "/" + filename + ".dat", scaled_design.transpose())  # type: ignore  # noqa: PGH003

        # Prepares the preprocess.py script with the design
        replace_template_values(
            self.__local_study_dir + "/pre_process.py",
            base_config,
        )

        # Launches a docker container with the pre_process.py script
        # The script generates the mesh and FFD files
        try:
            bash_command = (
                "source ~/.bashrc_mdolab && cd /home/mdolabuser/mount/engibench && python "
                + self.__docker_study_dir
                + "/pre_process.py"
                + " > "
                + self.__docker_study_dir
                + "/output_preprocess.log"
            )

            container.run(
                command=["/bin/bash", "-c", bash_command],
                image=self.container_id,
                name="machaero",
                mounts=[(self.__local_base_directory, self.__docker_base_dir)],
            )

        except Exception as e:
            # Verify output files exist
            mesh_file = self.__local_study_dir + "/" + filename + ".cgns"
            ffd_file = self.__local_study_dir + "/" + filename + "_ffd.xyz"
            msg = ""

            if not os.path.exists(mesh_file):
                msg += f"Mesh file not generated: {mesh_file}."
            if not os.path.exists(ffd_file):
                msg += f"FFD file not generated: {ffd_file}."
            raise RuntimeError(f"Pre-processing failed: {e!s}. {msg} Check logs in {self.__local_study_dir}") from e

        return filename

    def __reorder_coords(self, df_slice: pd.DataFrame) -> npt.NDArray[np.float32]:  # noqa: PLR0915
        node_c1 = np.array(df_slice["NodeC1"].dropna().values).astype(int)  # A list of [1,2,3,4,...]
        node_c2 = np.array(df_slice["NodeC2"].dropna().values).astype(int)  # A list of [2,3,4,5,...]
        connectivities = np.concatenate(
            (node_c1.reshape(-1, 1), node_c2.reshape(-1, 1)), axis=1
        )  # A list of [[1,2],[2,3],[3,4],...]

        # plot the x 'CoordinateX' and y 'CoordinateY' coordinates of the slice
        coords_x = df_slice["CoordinateX"].as_numpy()
        coords_y = df_slice["CoordinateY"].as_numpy()

        # We also have XoC YoC ZoC VelocityX VelocityY VelocityZ CoefPressure Mach
        # We would like to reorder these values in the same way as the coordinates, so we keep track of the indices
        indices = np.arange(len(df_slice))
        id_breaks_start = [0]
        id_breaks_end = []
        prev_id = 0
        segment_ids = np.zeros(len(connectivities))
        seg_id = 0
        for j in range(len(connectivities)):
            if connectivities[j][0] - 1 != prev_id:
                # This means that we have a new set of points
                id_breaks_start.append(connectivities[j][0] - 1)
                id_breaks_end.append(prev_id)
                seg_id += 1
            segment_ids[j] = seg_id

            prev_id = connectivities[j][1] - 1

        id_breaks_end.append(j)
        unique_segment_ids = np.arange(seg_id + 1)

        new_seg_order = unique_segment_ids.copy()

        # Loop over and sort the segments such that the end of each x and y coordinate for each segment is the start of the next segment
        # Loop through the segment ids
        seg_coords_start_x = coords_x[id_breaks_start]
        seg_coords_start_y = coords_y[id_breaks_start]
        seg_coords_end_x = coords_x[id_breaks_end]
        seg_coords_end_y = coords_y[id_breaks_end]

        seg_coords_end_x_idx = seg_coords_end_x[0]
        seg_coords_end_y_idx = seg_coords_end_y[0]
        seg_coords_start_x_idx = seg_coords_start_x[0]
        seg_coords_start_y_idx = seg_coords_start_y[0]

        ordered_ids = [unique_segment_ids[0]]

        # Loop through and find the end or start of a segment that matches the start of the current segment
        while len(ordered_ids) < len(unique_segment_ids):
            # Calculate the distance between the end of the current segment and the start of all other segments
            diff_end_idx_start_tot = np.sqrt(
                np.square(seg_coords_end_x_idx - seg_coords_start_x) + np.square(seg_coords_end_y_idx - seg_coords_start_y)
            )
            diff_start_idx_start_tot = np.sqrt(
                np.square(seg_coords_start_x_idx - seg_coords_start_x)
                + np.square(seg_coords_start_y_idx - seg_coords_start_y)
            )
            # Get the minimum distance excluding the current ordered segments)
            diff_end_idx_start_tot[np.abs(ordered_ids)] = np.inf
            diff_start_idx_start_tot[np.abs(ordered_ids)] = np.inf
            diff_end_idx_start_tot_id = np.argmin(diff_end_idx_start_tot)
            diff_end_idx_start_tot_min = diff_end_idx_start_tot[diff_end_idx_start_tot_id]

            # Calculate the distance between the end of the current segment and the end of all other segments
            diff_end_idx_end_tot = np.sqrt(
                np.square(seg_coords_end_x_idx - seg_coords_end_x) + np.square(seg_coords_end_y_idx - seg_coords_end_y)
            )
            # Get the minimum distance excluding the current segment
            diff_end_idx_end_tot[np.abs(ordered_ids)] = np.inf
            diff_end_idx_end_tot_id = np.argmin(diff_end_idx_end_tot)
            diff_end_idx_end_tot_min = diff_end_idx_end_tot[diff_end_idx_end_tot_id]
            # If the end of the current segment matches the start of another segment,
            # we have found the correct order
            if diff_end_idx_start_tot_min < diff_end_idx_end_tot_min:
                # Append the matching segment id to the ordered ids
                ordered_ids.append(diff_end_idx_start_tot_id)
                # Update the current segment end coordinates
                seg_coords_end_x_idx = seg_coords_end_x[diff_end_idx_start_tot_id]
                seg_coords_end_y_idx = seg_coords_end_y[diff_end_idx_start_tot_id]
            else:
                # If the end of the current segment matches the end of another segment,
                # the segment we append must be in reverse order
                # We make the sign of the segment id negative to indicate reverse order
                ordered_ids.append(-diff_end_idx_end_tot_id)
                # Update the current segment end coordinates;
                # Because of reversal, we use the start of the segment we are appending as the new end coordinates
                seg_coords_end_x_idx = seg_coords_start_x[diff_end_idx_end_tot_id]
                seg_coords_end_y_idx = seg_coords_start_y[diff_end_idx_end_tot_id]

        # Now we have the new order of the segments
        new_seg_order = np.array(ordered_ids)

        # Concatenate the segments in the new order
        coords_x_reordered = np.array([])
        coords_y_reordered = np.array([])
        indices_reordered = np.array([])

        for j in range(len(new_seg_order)):
            if new_seg_order[j] < 0:
                segment = np.nonzero(segment_ids == -new_seg_order[j])[0]
                coords_x_segment = coords_x[connectivities[segment] - 1][:, 0][::-1]
                coords_y_segment = coords_y[connectivities[segment] - 1][:, 0][::-1]
                indices_segment = indices[connectivities[segment] - 1][:, 0][::-1]
            else:
                segment = np.nonzero(segment_ids == new_seg_order[j])[0]
                coords_x_segment = coords_x[connectivities[segment] - 1][:, 0]
                coords_y_segment = coords_y[connectivities[segment] - 1][:, 0]
                indices_segment = indices[connectivities[segment] - 1][:, 0]
            coords_x_reordered = np.concatenate((coords_x_reordered, coords_x_segment))
            coords_y_reordered = np.concatenate((coords_y_reordered, coords_y_segment))
            indices_reordered = np.concatenate((indices_reordered, indices_segment))

        err = 1e-4
        err_x = 0.90
        max_x = np.amax(coords_x_reordered) * err_x
        max_x_ids = np.argwhere(np.abs(coords_x_reordered - np.amax(coords_x_reordered)) < err).reshape(-1, 1)
        max_x_ids = max_x_ids[coords_x_reordered[max_x_ids] >= max_x]
        # Get the y values at the maximum x values
        max_x_y_values = coords_y_reordered[max_x_ids]
        # Get the maximum y value
        max_y_value_id = np.argmax(max_x_y_values)
        max_y_value = max_x_y_values[max_y_value_id]
        # Get the minimum y value
        min_y_value_id = np.argmin(max_x_y_values)
        min_y_value = max_x_y_values[min_y_value_id]

        # Get the id of the value closest to the mean of the maximum and minimum y values
        mean_y_value = (max_y_value + min_y_value) / 2
        # Get the id of the value closest to the mean y value at the x value of the maximum y value
        mean_y_value_sub_id = np.argmin(np.abs(max_x_y_values - mean_y_value))
        mean_y_value_id = max_x_ids[mean_y_value_sub_id]
        # Now reorder the coordinates such that the mean y value is first
        coords_x_reordered = np.concatenate((coords_x_reordered[mean_y_value_id:], coords_x_reordered[:mean_y_value_id]))
        coords_y_reordered = np.concatenate((coords_y_reordered[mean_y_value_id:], coords_y_reordered[:mean_y_value_id]))
        indices_reordered = np.concatenate((indices_reordered[mean_y_value_id:], indices_reordered[:mean_y_value_id]))
        # Finally, check the direction of the coordinates and reverse if necessary
        # Randomly get the 20th coordinate and check if the x value is
        # greater than the x value of the first coordinate, and the y value is greater than the y value on the lower surface
        # Get the y value on the lower surface that is approximately at the same x value as the 20th coordinate
        x_lower_surf_values = coords_x_reordered[-20:]
        y_lower_surf_values = coords_y_reordered[-20:]
        y_lower_surf_value_id = np.argmin(np.abs(x_lower_surf_values - coords_x_reordered[20]))
        y_lower_surf_value = y_lower_surf_values[y_lower_surf_value_id]

        # if the upper surface is going right to left, reverse the coordinates.
        # We can also tell whether our assumption about the upper surface is correct by checking if the y value of the 20th coordinate is greater than the y value of the lower surface
        if coords_y_reordered[20] < y_lower_surf_value:
            coords_x_reordered = coords_x_reordered[::-1]
            coords_y_reordered = coords_y_reordered[::-1]
            indices_reordered = indices_reordered[::-1]

        # Lastly, remove any successive duplicate coordinates
        err_remove = 1e-6
        removal_ids = np.where(np.abs(np.diff(coords_x_reordered) + np.diff(coords_y_reordered)) < err_remove)[0]
        indices_reordered = np.delete(indices_reordered, removal_ids)
        coords_x_reordered = np.delete(coords_x_reordered, removal_ids)
        coords_y_reordered = np.delete(coords_y_reordered, removal_ids)

        # Concatenate the the first coordinate to the end of the array
        coords_x_reordered = np.concatenate((coords_x_reordered, np.expand_dims(coords_x_reordered[0], axis=0)))
        coords_y_reordered = np.concatenate((coords_y_reordered, np.expand_dims(coords_y_reordered[0], axis=0)))
        indices_reordered = np.concatenate((indices_reordered, np.expand_dims(indices_reordered[0], axis=0)))

        return np.array([coords_x_reordered, coords_y_reordered])

    def __simulator_output_to_design(self, simulator_output: str | None = None) -> npt.NDArray[np.float32]:
        """Converts a simulator output to a design.

        Args:
            simulator_output (str): The simulator output to convert.

        Returns:
            np.ndarray: The corresponding design.
        """
        if simulator_output is None:
            # Take latest slice file
            files = os.listdir(self.__local_study_dir + "/output")
            files = [f for f in files if f.endswith("_slices.dat")]
            file_numbers = [int(f.split("_")[1]) for f in files]
            simulator_output = files[file_numbers.index(max(file_numbers))]

        slice_file = self.__local_study_dir + "/output/" + simulator_output

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
        nodes_arr = pd.read_csv(slice_file, sep=r"\s+", names=["NodeC1", "NodeC2"], skiprows=5 + nnodes, engine="c")

        # Concatenate node connections to the main data
        slice_df = pd.concat([slice_df, nodes_arr], axis=1)

        return self.__reorder_coords(slice_df)

    def simulate(self, design: DesignType, config: dict[str, Any] | None = None, mpicores: int = 4) -> npt.NDArray:
        """Simulates the performance of an airfoil design.

        Args:
            design (dict): The design to simulate.
            config (dict): A dictionary with configuration (e.g., boundary conditions, filenames) for the simulation.
            mpicores (int): The number of MPI cores to use in the simulation.

        Returns:
            dict: The performance of the design - each entry of the dict corresponds to a named objective value.
        """
        # docker pull image if not already pulled
        if container.RUNTIME is not None:
            container.pull(self.container_id)
        # pre-process the design and run the simulation

        # Prepares the airfoil_analysis.py script with the simulation configuration
        base_config = {
            "alpha": design["angle_of_attack"],
            "mach": 0.8,
            "reynolds": 1e6,
            "altitude": 10000,
            "temperature": 300,
            "use_altitude": False,
            "output_dir": "'" + self.__docker_study_dir + "/output/'",
            "mesh_fname": "'" + self.__docker_study_dir + "/design.cgns'",
            "task": "'analysis'",  # TODO(cashend): We can add the option to perform a polar analysis.
            # https://github.com/IDEALLab/EngiBench/issues/15
        }
        base_config.update(self.conditions)
        base_config.update(config or {})
        self.__design_to_simulator_input(design, base_config)

        replace_template_values(
            self.__local_study_dir + "/airfoil_analysis.py",
            base_config,
        )

        # Launches a docker container with the airfoil_analysis.py script
        # The script takes a mesh and ffd and performs an optimization
        try:
            bash_command = (
                "source ~/.bashrc_mdolab && cd /home/mdolabuser/mount/engibench && mpirun -np "
                + str(mpicores)
                + " python "
                + self.__docker_study_dir
                + "/airfoil_analysis.py > "
                + self.__docker_study_dir
                + "/output.log"
            )
            container.run(
                command=["/bin/bash", "-c", bash_command],
                image=self.container_id,
                name="machaero",
                mounts=[(self.__local_base_directory, self.__docker_base_dir)],
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to run airfoil analysis: {e!s}. Please check logs in {self.__local_study_dir}."
            ) from e

        outputs = np.load(self.__local_study_dir + "/output/outputs.npy")
        lift = float(outputs[3])
        drag = float(outputs[4])
        return np.array([drag, lift])

    def optimize(
        self, starting_point: DesignType, config: dict[str, Any] | None = None, mpicores: int = 4
    ) -> tuple[DesignType, list[OptiStep]]:
        """Optimizes the design of an airfoil.

        Args:
            starting_point (dict): The starting point for the optimization.
            config (dict): A dictionary with configuration (e.g., boundary conditions, filenames) for the optimization.
            mpicores (int): The number of MPI cores to use in the optimization.

        Returns:
            tuple[dict[str, Any], list[OptiStep]]: The optimized design and its performance.
        """
        # docker pull image if not already pulled
        if container.RUNTIME is not None:
            container.pull(self.container_id)
        # pre-process the design and run the simulation
        filename = "candidate_design"

        # Prepares the optimize_airfoil.py script with the optimization configuration
        base_config = {
            "cl_target": 0.5,
            "alpha": starting_point["angle_of_attack"],
            "mach": 0.75,
            "reynolds": 1e6,
            "altitude": 10000,
            "temperature": 300,  # should specify either mach + altitude or mach + reynolds + reynoldsLength (default to 1) + temperature
            "use_altitude": False,
            "area_initial": None,  # actual initial airfoil area
            "area_ratio_min": 0.7,  # Minimum ratio the initial area is allowed to decrease to i.e minimum_area = area_initial*area_target
            "opt": "'SLSQP'",
            "opt_options": {},
            "output_dir": "'" + self.__docker_study_dir + "/output/'",
            "ffd_fname": "'" + self.__docker_study_dir + "/" + filename + "_ffd.xyz'",
            "mesh_fname": "'" + self.__docker_study_dir + "/" + filename + ".cgns'",
            "area_input_design": self._calc_area(starting_point["coords"]),
        }
        base_config.update(self.conditions)
        base_config.update(config or {})
        self.__design_to_simulator_input(starting_point, base_config, filename)
        replace_template_values(
            self.__local_study_dir + "/airfoil_opt.py",
            base_config,
        )

        try:
            # Launches a docker container with the optimize_airfoil.py script
            # The script takes a mesh and ffd and performs an optimization
            bash_command = (
                "source ~/.bashrc_mdolab && cd /home/mdolabuser/mount/engibench && mpirun -np "
                + str(mpicores)
                + " python "
                + self.__docker_study_dir
                + "/airfoil_opt.py > "
                + self.__docker_study_dir
                + "/airfoil_opt.log"
            )
            container.run(
                command=["/bin/bash", "-c", bash_command],
                image=self.container_id,
                name="machaero",
                mounts=[(self.__local_base_directory, self.__docker_base_dir)],
            )
        except Exception as e:
            raise RuntimeError(f"Optimization failed: {e!s}. Check logs in {self.__local_study_dir}") from e

        # post process -- extract the shape and objective values
        optisteps_history = []
        history = pyoptsparse.History(self.__local_study_dir + "/output/opt.hst")
        iters = list(map(int, history.getCallCounters()[:]))

        for i in range(len(iters)):
            vals = history.read(int(iters[i]))
            if vals is not None and "funcs" in vals and "obj" in vals["funcs"] and not vals["fail"]:
                objective = history.getValues(names=["obj"], callCounters=[i], allowSens=False, major=False, scale=True)[
                    "obj"
                ]
                optisteps_history.append(OptiStep(obj_values=np.array([objective]), step=vals["iter"]))

        history.close()

        opt_coords = self.__simulator_output_to_design()

        return {"coords": opt_coords, "angle_of_attack": starting_point["angle_of_attack"]}, optisteps_history

    def render(self, design: DesignType, *, open_window: bool = False) -> Any:
        """Renders the design in a human-readable format.

        Args:
            design (dict): The design to render.
            open_window (bool): If True, opens a window with the rendered design.

        Returns:
            Any: The rendered design.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        coords = design["coords"]

        ax.scatter(coords[0], coords[1], s=10, alpha=0.7)
        plt.ylim(-0.15, 0.15)
        if open_window:
            plt.show()
        return fig, ax

    def random_design(self) -> tuple[dict[str, Any], int]:
        """Samples a valid random initial design.

        Returns:
            tuple[dict[str, Any], int]: The valid random design and the index of the design in the dataset.
        """
        rnd = self.np_random.integers(low=0, high=len(self.dataset["train"]["initial_design"]), dtype=int)
        initial_design = self.dataset["train"]["initial_design"][rnd]
        return {"coords": np.array(initial_design["coords"]), "angle_of_attack": initial_design["angle_of_attack"]}, rnd

    def _calc_off_wall_distance(  # noqa: PLR0913
        self,
        mach: float,
        reynolds: float,
        freestreamTemp: float = 300.0,  # noqa: N803
        reynoldsLength: float = 1.0,  # noqa: N803
        yplus: float = 1,
        R: float = 287.0,  # noqa: N803
        gamma: float = 1.4,
    ) -> float:
        """Estimation of the off-wall distance for a given design.

        The off-wall distance is calculated using the Reynolds number and the freestream temperature.
        """
        # ---------------------------
        a = np.sqrt(gamma * R * freestreamTemp)
        u = mach * a
        # ---------------------------
        # Viscosity from Sutherland's law
        ## Sutherland's law parameters
        mu0 = 1.716e-5
        T0 = 273.15  # noqa: N806
        S = 110.4  # noqa: N806
        mu = mu0 * ((freestreamTemp / T0) ** (3 / 2)) * (T0 + S) / (freestreamTemp + S)
        # ---------------------------
        # Density
        rho = reynolds * mu / (reynoldsLength * u)
        ## Skin friction coefficient
        Cf = (2 * np.log10(reynolds) - 0.65) ** (-2.3)  # noqa: N806
        # Wall shear stress
        tau = Cf * 0.5 * rho * (u**2)
        # Friction velocity
        uTau = np.sqrt(tau / rho)  # noqa: N806
        # Off wall distance
        return yplus * mu / (rho * uTau)

    def _is_blunted(self, coords: npt.NDArray[np.float64], delta_x_tol: float = 1e-5) -> bool:
        """Checks if the coordinates are blunted or not.

        Args:
            coords (np.ndarray): The coordinates to check.
            delta_x_tol (float): The tolerance for the x coordinate difference.

        Returns:
            bool: True if the coordinates are blunted, False otherwise.
        """
        # Check if the coordinates going away from the tip have a small delta y
        coords_x = coords[0, :]
        # Get all of the trailing edge indices, i.e where consecutive x coordinates are the same
        x_gt = np.max(coords_x) * 0.99
        trailing_edge_indices_l = np.where(np.abs(coords_x - np.roll(coords_x, -1)) < delta_x_tol)[0]
        trailing_edge_indices_r = np.where(np.abs(coords_x - np.roll(coords_x, 1)) < delta_x_tol)[0]
        # Include any indices that are in either list
        trailing_edge_indices = np.unique(np.concatenate((trailing_edge_indices_l, trailing_edge_indices_r)))
        trailing_edge_indices = trailing_edge_indices[coords_x[trailing_edge_indices] >= x_gt]

        # check if we have no trailing edge indices
        return not len(trailing_edge_indices) <= 1

    def _calc_area(self, coords: npt.NDArray[np.float32]) -> float:
        """Calculates the area of the airfoil.

        Args:
            coords (np.ndarray): The coordinates of the airfoil.

        Returns:
            float: The area of the airfoil.
        """
        return 0.5 * np.absolute(
            np.sum(coords[0, :] * np.roll(coords[1, :], -1)) - np.sum(coords[1, :] * np.roll(coords[0, :], -1))
        )

    def _scale_coords(
        self,
        coords: npt.NDArray[np.float64],
        blunted: bool = False,  # noqa: FBT001, FBT002
        xcut: float = 0.99,
        min_trailing_edge_indices: float = 6,
    ) -> tuple[npt.NDArray[np.float64], bool]:
        """Scales the coordinates to fit in the design space.

        Args:
            coords (np.ndarray): The coordinates to scale.
            blunted (bool): If True, the coordinates are assumed to be blunted.
            xcut (float): The x coordinate of the cut, if the coordinates are blunted.
            min_trailing_edge_indices (int): The minimum number of trailing edge indices to remove.

        Returns:
            np.ndarray: The scaled coordinates.
        """
        # Test if the coordinates are blunted or not
        if not (blunted) and self._is_blunted(coords):
            blunted = True
            print(
                "The coordinates may be blunted. However, blunted was not set to True. Will set blunted to True and continue, but please check the coordinates."
            )

        if not (blunted):
            xcut = 1.0

        # Scale x coordinates to be xcut in length
        airfoil_length = np.abs(np.max(coords[0, :]) - np.min(coords[0, :]))

        # Center the coordinates around the leading edge and scale them
        coords[0, :] = xcut * (coords[0, :] - np.min(coords[0, :])) / airfoil_length
        airfoil_length = np.abs(np.max(coords[0, :]) - np.min(coords[0, :]))

        # Shift the coordinates to be centered at 0 at the leading edge
        leading_id = np.argmin(coords[0, :])
        y_dist = coords[1, leading_id]
        coords[1, :] += -y_dist
        # Ensure the first and last points are the same
        coords[0, 0] = xcut
        coords[0, -1] = xcut
        coords[1, -1] = coords[1, 0]
        # Set the leading edge location

        if blunted:
            coords_x = coords[0, :]
            # Get all of the trailing edge indices, i.e where consecutive x coordinates are the same
            err = 1e-5
            x_gt = np.max(coords_x) * 0.99
            trailing_edge_indices_l = np.where(np.abs(coords_x - np.roll(coords_x, -1)) < err)[0]
            trailing_edge_indices_r = np.where(np.abs(coords_x - np.roll(coords_x, 1)) < err)[0]
            # Include any indices that are in either list
            trailing_edge_indices = np.unique(np.concatenate((trailing_edge_indices_l, trailing_edge_indices_r)))
            trailing_edge_indices = trailing_edge_indices[coords_x[trailing_edge_indices] >= x_gt]

            err = 1e-4
            err_stop = 1e-3
            while len(trailing_edge_indices) < min_trailing_edge_indices:
                trailing_edge_indices_l = np.where(np.abs(coords_x - np.roll(coords_x, -1)) < err)[0]
                trailing_edge_indices_r = np.where(np.abs(coords_x - np.roll(coords_x, 1)) < err)[0]
                # Include any indices that are in either list
                trailing_edge_indices = np.unique(np.concatenate((trailing_edge_indices_l, trailing_edge_indices_r)))
                trailing_edge_indices = trailing_edge_indices[coords_x[trailing_edge_indices] >= x_gt]
                err *= 1.5
                if err > err_stop:
                    break

            # Remove the trailing edge indices from the coordinates
            coords = np.delete(coords, trailing_edge_indices[1:-1], axis=1)

        return coords, blunted


if __name__ == "__main__":
    problem = Airfoil()
    problem.reset(seed=0, cleanup=False)

    dataset = problem.dataset
    # Get design and conditions from the dataset
    # Print Dataset object keys
    design, idx = problem.random_design()
    config = dataset["train"].select_columns(problem.conditions_keys)[idx]

    print(problem.simulate(design, config=config, mpicores=8))

    # Get design and conditions from the dataset, render design
    design_tuple, optisteps_history = problem.optimize(design, config=config, mpicores=8)
