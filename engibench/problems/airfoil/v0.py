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


class Airfoil(Problem[dict[str, Any]]):
    r"""Airfoil 2D shape optimization problem.

    ## Problem Description
    This problem simulates the performance of an airfoil in a 2D environment. An airfoil is represented by a set of 192 points that define its shape. The performance is evaluated by the [MACH-Aero](https://mdolab-mach-aero.readthedocs-hosted.com/en/latest/) simulator that computes the lift and drag coefficients of the airfoil.

    ## Design space
    The design space is represented by a dictionary where one element (`coords`) is a numpy array (vector of 192 x,y coordinates in `[0., 1.)` per design) that define the airfoil shape, and the other element (`angle_of_attack`) is a scalar.

    ## Objectives
    The objectives are defined and indexed as follows:

    0. `cd`: Drag coefficient to minimize.
    1. `cl`: Lift coefficient to maximize.

    ## Conditions
    The conditions are defined by the following parameters:
    - `mach`: Mach number.
    - `reynolds`: Reynolds number.
    - `area_ratio_min`: Minimum area ratio (ratio relative to initial area) constraint.
    - `area_initial`: Initial area.
    - `cl_target`: Target lift coefficient to satisfy equality constraint.

    ## Simulator
    The simulator is a docker container with the MACH-Aero software that computes the lift and drag coefficients of the airfoil.

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
    objectives: tuple[tuple[str, ObjectiveDirection], ...] = (
        ("cd", ObjectiveDirection.MINIMIZE),
        ("cl", ObjectiveDirection.MAXIMIZE),
    )
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
    _dataset = None

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

    def __design_to_simulator_input(self, design: dict[str, Any], config: dict[str, Any], filename: str = "design") -> str:
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
        np.savetxt(self.__local_study_dir + "/" + filename + ".dat", scaled_design.transpose())

        base_config = {
            "design_fname": f"'{self.__docker_study_dir}/{filename}.dat'",
            "tmp_xyz_fname": "'" + self.__docker_study_dir + "/tmp'",
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
            raise RuntimeError(f"Pre-processing failed: {e!s}. {msg} Check logs in {self.__local_study_dir}") from e  # noqa: TRY003

        return filename

    def __reorder_coords(self, df_slice: pd.DataFrame) -> npt.NDArray[np.float32]:  # noqa: PLR0915
        """Reorder the coordinates of a slice such that the segments are ordered correctly. WIP temporary code.

        Args:
            df_slice (pd.DataFrame): The raw dataframe slice.

        Returns:
            coords_reordered (np.ndarray): The reordered x and y coordinates of the slice.
        """
        # Now get connectivities
        node_c1 = np.array(df_slice["NodeC1"].dropna().values).astype(int)  # A list of [1,2,3,4,...]
        node_c2 = np.array(df_slice["NodeC2"].dropna().values).astype(int)  # A list of [2,3,4,5,...]
        connectivities = np.concatenate(
            (node_c1.reshape(-1, 1), node_c2.reshape(-1, 1)), axis=1
        )  # A list of [[1,2],[2,3],[3,4],...]

        # plot the x 'CoordinateX' and y 'CoordinateY' coordinates of the slice
        coords_x = df_slice["CoordinateX"].values
        coords_y = df_slice["CoordinateY"].values

        # We also have XoC YoC ZoC VelocityX VelocityY VelocityZ CoefPressure Mach
        # We would like to reorder these values in the same way as the coordinates, so we keep track of the indices
        indices = np.arange(len(df_slice))
        id_breaks_start_id = [0]
        id_breaks_end_id = []
        prev_id = 0
        segment_ids = np.zeros(len(connectivities))
        seg_id = 0

        for j in range(len(connectivities)):
            if connectivities[j][0] - 1 != prev_id:
                # This means that we have a new set of points
                id_breaks_start_id.append(connectivities[j][0] - 1)
                id_breaks_end_id.append(prev_id)
                seg_id += 1
            segment_ids[j] = seg_id

            prev_id = connectivities[j][1] - 1

        id_breaks_end_id.append(j)

        unique_segment_ids = np.arange(seg_id + 1)
        new_seg_order = unique_segment_ids.copy()

        # Loop over and sort the segments such that the end of each x and y coordinate for each segment is the start of the next segment
        # Loop through the segment ids
        seg_coords_start_x = coords_x[id_breaks_start_id]
        seg_coords_start_y = coords_y[id_breaks_start_id]
        seg_coords_end_x = coords_x[id_breaks_end_id]
        seg_coords_end_y = coords_y[id_breaks_end_id]

        err = 1e-8
        for j in range(len(unique_segment_ids)):
            seg_coords_start_x_j = seg_coords_start_x[j]
            seg_coords_start_y_j = seg_coords_start_y[j]
            seg_coords_end_x_j = seg_coords_end_x[j]
            seg_coords_end_y_j = seg_coords_end_y[j]
            # Loop through the segment ids
            seg_x_diff_start = np.abs(seg_coords_start_x_j - seg_coords_end_x)
            seg_y_diff_start = np.abs(seg_coords_start_y_j - seg_coords_end_y)
            seg_tot_diff_start = seg_x_diff_start + seg_y_diff_start
            seg_tot_diff_min_id_start = np.argmin(seg_tot_diff_start)
            seg_tot_diff_min_start = seg_tot_diff_start[seg_tot_diff_min_id_start]

            seg_x_diff_end = np.abs(seg_coords_end_x_j - seg_coords_start_x)
            seg_y_diff_end = np.abs(seg_coords_end_y_j - seg_coords_start_y)
            seg_tot_diff_end = seg_x_diff_end + seg_y_diff_end
            seg_tot_diff_min_id_end = np.argmin(seg_tot_diff_end)

            new_seg_order_temp = unique_segment_ids.copy()

            if seg_tot_diff_min_start > err:
                # No segment matches, so we have found the starting segment
                # Now we need to reorder the segments
                # Put the first segment in the first position
                new_seg_order_temp[0] = unique_segment_ids[j]
                new_seg_order_temp[j] = unique_segment_ids[0]
                # Now choose the id that is minimum; i.e we match the end of the first segment to the start of the next most likely segment
                new_seg_order_temp[1] = unique_segment_ids[seg_tot_diff_min_id_end]
                new_seg_order_temp[seg_tot_diff_min_id_end] = unique_segment_ids[1]
                # Now loop through the remaining segments and match the end of the previous segment to the start of the next most likely segment, breaking if we reach the end
                for k in range(2, len(unique_segment_ids)):
                    # Get the previous segment id
                    prev_seg_id = new_seg_order_temp[k - 1]
                    # Get the previous segment end coordinates
                    prev_seg_end_x = seg_coords_end_x[prev_seg_id]
                    prev_seg_end_y = seg_coords_end_y[prev_seg_id]
                    # Get the remaining segment ids
                    remaining_seg_ids = np.setdiff1d(unique_segment_ids, new_seg_order_temp[:k])
                    # Get the remaining segment start coordinates
                    remaining_seg_start_x = seg_coords_start_x[remaining_seg_ids]
                    remaining_seg_start_y = seg_coords_start_y[remaining_seg_ids]
                    # Get the remaining segment end coordinates
                    # Get the difference between the previous segment end coordinates and the remaining segment start coordinates
                    remaining_seg_x_diff = np.abs(prev_seg_end_x - remaining_seg_start_x)
                    remaining_seg_y_diff = np.abs(prev_seg_end_y - remaining_seg_start_y)
                    remaining_seg_tot_diff = remaining_seg_x_diff + remaining_seg_y_diff
                    # Get the minimum difference
                    remaining_seg_tot_diff_min_id = np.argmin(remaining_seg_tot_diff)
                    # Now get the id of the remaining segment that matches the previous segment end coordinates
                    remaining_seg_id = remaining_seg_ids[remaining_seg_tot_diff_min_id]
                    # Now put the remaining segment id in the next position
                    new_seg_order_temp[k] = remaining_seg_id

                # Now we have a new segment order
                new_seg_order = new_seg_order_temp
                break

        # We can use the new order to plot all segments at once
        # Concatenate the segments in the new order
        coords_x_reordered = np.array([])
        coords_y_reordered = np.array([])
        indices_reordered = np.array([])

        for j in range(len(new_seg_order)):
            segment = np.nonzero(segment_ids == new_seg_order[j])[0]
            coords_x_segment = coords_x[connectivities[segment] - 1][:, 0]
            coords_y_segment = coords_y[connectivities[segment] - 1][:, 0]
            indices_segment = indices[connectivities[segment] - 1][:, 0]
            coords_x_reordered = np.concatenate((coords_x_reordered, coords_x_segment))
            coords_y_reordered = np.concatenate((coords_y_reordered, coords_y_segment))
            indices_reordered = np.concatenate((indices_reordered, indices_segment))

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

    def simulate(self, design: dict[str, Any], config: dict[str, Any], mpicores: int = 4) -> dict[str, Any]:
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
            "task": "'analysis'",  # TODO: We can add the option to perform a polar analysis.  # noqa: FIX002
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
            raise RuntimeError(  # noqa: TRY003
                f"Failed to run airfoil analysis: {e!s}. Please check logs in {self.__local_study_dir}."
            ) from e

        outputs = np.load(self.__local_study_dir + "/output/outputs.npy")
        lift = float(outputs[3])
        drag = float(outputs[4])
        return np.array([drag, lift])

    def optimize(
        self, starting_point: dict[str, Any], config: dict[str, Any] | None = None, mpicores: int = 4
    ) -> tuple[dict[str, Any], list[OptiStep]]:
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
            raise RuntimeError(f"Optimization failed: {e!s}. Check logs in {self.__local_study_dir}") from e  # noqa: TRY003

        # post process -- extract the shape and objective values
        optisteps_history = []
        history = pyoptsparse.History(self.__local_study_dir + "/output/opt.hst")

        # TODO return the full history of the optimization instead of just the last step # noqa: FIX002
        # Also, this is inconsistent with the definition of the problem saying we optimize 2 objectives...
        objective = history.getValues(names=["obj"], callCounters=None, allowSens=False, major=False, scale=True)["obj"][
            -1, -1
        ]
        optisteps_history.append(OptiStep(obj_values=np.array([objective]), step=0))
        history.close()

        opt_coords = self.__simulator_output_to_design()

        return {"coords": opt_coords, "angle_of_attack": starting_point["angle_of_attack"]}, optisteps_history

    def render(self, design: dict[str, Any], open_window: bool = False) -> Any:
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
        rnd = self.np_random.integers(low=0, high=len(self.dataset["train"]["initial_design"]), dtype=int)  # pyright: ignore[reportOptionalMemberAccess]
        initial_design = self.dataset["train"]["initial_design"][rnd]
        return {"coords": np.array(initial_design["coords"]), "angle_of_attack": initial_design["angle_of_attack"]}, rnd

    def _calc_off_wall_distance(  # noqa
        self,
        mach: float,
        reynolds: float,
        freestreamTemp: float = 300.0,  # noqa
        reynoldsLength: float = 1.0,  # noqa
        yplus: float = 1,
        R: float = 287.0,  # noqa
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
        T0 = 273.15  # noqa
        S = 110.4  # noqa
        mu = mu0 * ((freestreamTemp / T0) ** (3 / 2)) * (T0 + S) / (freestreamTemp + S)
        # ---------------------------
        # Density
        rho = reynolds * mu / (reynoldsLength * u)
        ## Skin friction coefficient
        Cf = (2 * np.log10(reynolds) - 0.65) ** (-2.3)  # noqa
        # Wall shear stress
        tau = Cf * 0.5 * rho * (u**2)
        # Friction velocity
        uTau = np.sqrt(tau / rho)  # noqa
        # Off wall distance
        delta = yplus * mu / (rho * uTau)
        return delta

    def _scale_coords(self, coords: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Scales the coordinates to fit in the design space.

        Args:
            coords (np.ndarray): The coordinates to scale.

        Returns:
            np.ndarray: The scaled coordinates.
        """
        # Align coordinates to [0,1]
        y_dist = coords[1, 0]
        x_dist = 1 - coords[0, 0]
        coords[0, :] += x_dist
        coords[1, :] += -y_dist

        coords[0, :] = coords[0, :] - coords[0, 0] + 1
        coords[1, :] = coords[1, :] - coords[1, 0]

        coords[0, 0] = 1
        coords[1, 0] = 0
        coords[0, -1] = 1
        coords[1, -1] = 0
        return coords

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


if __name__ == "__main__":
    problem = Airfoil()
    problem.reset(seed=0, cleanup=False)

    dataset = problem.dataset
    # Get design and conditions from the dataset
    # Print Dataset object keys
    design, idx = problem.random_design()
    config = dataset["train"].select_columns(problem.conditions_keys)[idx]

    # Get design and conditions from the dataset, render design
    print(problem.optimize(design, config=config, mpicores=8))
