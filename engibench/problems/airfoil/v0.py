"""Airfoil problem.

Filename convention is that folder paths do not end with /. For example, /path/to/folder is correct, but /path/to/folder/ is not.
"""

import dataclasses
from dataclasses import dataclass
from dataclasses import field
from importlib.util import find_spec
import json
import os
import shutil
import sys
from typing import Annotated, Any

from gymnasium import spaces
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from engibench.constraint import bounded
from engibench.constraint import constraint
from engibench.constraint import IMPL
from engibench.constraint import THEORY
from engibench.core import ObjectiveDirection
from engibench.core import OptiStep
from engibench.core import Problem
from engibench.problems.airfoil.pyopt_history import History
from engibench.problems.airfoil.templates import cli_interface
from engibench.problems.airfoil.utils import calc_area
from engibench.problems.airfoil.utils import calc_off_wall_distance
from engibench.problems.airfoil.utils import reorder_coords
from engibench.problems.airfoil.utils import scale_coords
from engibench.utils import container
from engibench.utils.files import clone_dir

# Allow loading pyoptsparse histories even if pyoptsparse is not installed:
if find_spec("pyoptsparse") is None:
    from engibench.problems.airfoil import fake_pyoptsparse

    sys.modules["pyoptsparse"] = fake_pyoptsparse

DesignType = dict[str, Any]


def self_intersect(curve: npt.NDArray[np.float64]) -> tuple[int, npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
    """Determines if two segments a and b intersect."""
    # intersection: find t such that (p + t dp - q) x dq = 0 with 0 <= t <= 1
    # and (q + s dq - p) x dp = 0, 0 <= s <= 1
    # dp x dq = 0 => parallel => no intersection
    #
    # t = (q-p) x dq / dp x dq
    # s = (q-p) x dp / dp x dq
    #
    # Also use the fact that 2 consecutive segments always intersect (at their common point)
    # => never check consecutive segments
    segments = curve[1:] - curve[:-1]
    n = segments.shape[0]
    for i in range(n - 1):
        p, dp = curve[i], segments[i]
        end = n - 1 if i == 0 else n
        q, dq = curve[i + 2 : end], segments[i + 2 : end]
        x = np.cross(dp, dq)
        parallel = x == 0.0
        t = np.cross(q[~parallel] - p, dq[~parallel]) / x[~parallel]
        s = np.cross(q[~parallel] - p, dp) / x[~parallel]
        if np.any((t >= 0.0) & (t <= 1.0) & (s >= 0.0) & (s <= 1.0)):
            return i, p, curve[i + 1]
    return None


@constraint(categories=IMPL)
def does_not_self_intersect(design: DesignType) -> None:
    """Check if a curve has no self intersections."""
    intersection = self_intersect(design["coords"])
    assert intersection is None, (
        f"design: Curve does self intersect at segment {intersection[0]}: {intersection[1]} -- {intersection[2]}"
    )


class Airfoil(Problem[DesignType]):
    r"""Airfoil 2D shape optimization problem.

    This problem simulates the performance of an airfoil in a 2D environment. An airfoil is represented by a set of 192 points that define its shape. The performance is evaluated by the [MACH-Aero](https://mdolab-mach-aero.readthedocs-hosted.com/en/latest/) simulator that computes the lift and drag coefficients of the airfoil.
    """

    version = 0
    objectives: tuple[tuple[str, ObjectiveDirection], ...] = (
        ("cd", ObjectiveDirection.MINIMIZE),
        ("cl", ObjectiveDirection.MAXIMIZE),
    )

    design_space = spaces.Dict(
        {
            "coords": spaces.Box(low=0.0, high=1.0, shape=(2, 192), dtype=np.float32),
            "angle_of_attack": spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
        }
    )
    design_constraints = (does_not_self_intersect,)
    dataset_id = "IDEALLab/airfoil_v0"
    container_id = "mdolab/public:u22-gcc-ompi-stable"
    __local_study_dir: str

    @dataclass
    class Conditions:
        """Conditions."""

        mach: Annotated[
            float, bounded(lower=0.0).category(IMPL), bounded(lower=0.1, upper=1.0).warning().category(IMPL)
        ] = 0.8
        """Mach number"""
        reynolds: Annotated[
            float, bounded(lower=0.0).category(IMPL), bounded(lower=1e5, upper=1e9).warning().category(IMPL)
        ] = 1e6
        """Reynolds number"""
        area_initial: float = float("NAN")
        """actual initial airfoil area"""
        area_ratio_min: Annotated[float, bounded(lower=0.0, upper=1.2).category(THEORY)] = 0.7
        """Minimum ratio the initial area is allowed to decrease to i.e minimum_area = area_initial*area_target"""
        cl_target: float = 0.5
        """Target lift coefficient to satisfy equality constraint"""

    conditions = Conditions()

    @dataclass
    class Config(Conditions):
        """Structured representation of configuration parameters for a numerical computation."""

        alpha: Annotated[float, bounded(lower=0.0, upper=10.0).category(THEORY)] = 0.0
        altitude: float = 10000.0
        temperature: float = 300.0
        use_altitude: bool = False
        output_dir: str | None = None
        mesh_fname: str | None = None
        task: str = "analysis"
        opt: str = "SLSQP"
        opt_options: dict = field(default_factory=dict)
        ffd_fname: str | None = None
        area_input_design: float | None = None

        @constraint(categories=THEORY)
        @staticmethod
        def area_ratio_bound(area_ratio_min: float, area_initial: float, area_input_design: float | None) -> None:
            """Constraint for area_ratio_min <= area_ratio <= 1.2."""
            area_ratio_max = 1.2
            if area_input_design is None:
                return
            assert not np.isnan(area_initial)
            area_ratio = area_input_design / area_initial
            assert area_ratio_min <= area_ratio <= area_ratio_max, (
                f"Config.area_ratio: {area_ratio} ∉ [area_ratio_min={area_ratio_min}, 1.2]"
            )

    def __init__(self, seed: int = 0, base_directory: str | None = None) -> None:
        """Initializes the Airfoil problem.

        Args:
            seed (int): The random seed for the problem.
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

        super().__init__(seed=seed)

    def reset(self, seed: int | None = None, *, cleanup: bool = False) -> None:
        """Resets the simulator and numpy random to a given seed.

        Args:
            seed (int, optional): The seed to reset to. If None, a random seed is used.
            cleanup (bool): Deletes the previous study directory if True.
        """
        if cleanup:
            shutil.rmtree(self.__local_study_dir)

        super().reset(seed)
        self.current_study = f"study_{self.seed}-pid{os.getpid()}"
        self.__local_study_dir = self.__local_target_dir + "/" + self.current_study
        self.__docker_study_dir = self.__docker_target_dir + "/" + self.current_study

    def __design_to_simulator_input(
        self, design: DesignType, mach: float, reynolds: float, temperature: float, filename: str = "design"
    ) -> str:
        """Converts a design to a simulator input.

        The simulator inputs are two files: a mesh file (.cgns) and a FFD file (.xyz). This function generates these files from the design.
        The files are saved in the current directory with the name "$filename.cgns" and "$filename_ffd.xyz".

        Args:
            design (dict): The design to convert.
            mach: mach number
            reynolds: reynolds number
            temperature: temperature
            filename (str): The filename to save the design to.
        """
        # Creates the study directory
        clone_dir(source_dir=self.__local_template_dir, target_dir=self.__local_study_dir)

        tmp = os.path.join(self.__docker_study_dir, "tmp")

        # Calculate the off-the-wall distance
        estimate_s0 = True

        s0 = calc_off_wall_distance(mach=mach, reynolds=reynolds, freestreamTemp=temperature) if estimate_s0 else 1e-5
        # Scale the design to fit in the design space
        x_cut = 0.99
        scaled_design, input_blunted = scale_coords(
            design["coords"],
            blunted=False,
            xcut=x_cut,
        )
        args = cli_interface.PreprocessParameters(
            design_fname=f"{self.__docker_study_dir}/{filename}.dat",
            tmp_xyz_fname=tmp,
            mesh_fname=self.__docker_study_dir + "/" + filename + ".cgns",
            ffd_fname=self.__docker_study_dir + "/" + filename + "_ffd",
            N_sample=180,
            n_tept_s=4,
            x_cut=x_cut,
            ffd_ymarginu=0.05,
            ffd_ymarginl=0.05,
            ffd_pts=10,
            N_grid=100,
            s0=s0,
            input_blunted=input_blunted,
            march_dist=100.0,
        )

        # Save the design to a temporary file. Format to 1e-6 rounding
        np.savetxt(self.__local_study_dir + "/" + filename + ".dat", scaled_design.transpose())

        # Launches a docker container with the pre_process.py script
        # The script generates the mesh and FFD files
        bash_command = f"source /home/mdolabuser/.bashrc_mdolab && cd {self.__docker_base_dir} && python {self.__docker_study_dir}/pre_process.py '{json.dumps(dataclasses.asdict(args))}'"
        assert self.container_id is not None, "Container ID is not set"
        container.run(
            command=["/bin/bash", "-c", bash_command],
            image=self.container_id,
            name="machaero",
            mounts=[(self.__local_base_directory, self.__docker_base_dir)],
            sync_uid=True,
        )

        return filename

    def simulator_output_to_design(self, simulator_output: str | None = None) -> npt.NDArray[np.float32]:
        """Converts a simulator output to a design.

        Args:
            simulator_output (str): The simulator output to convert. If None, the latest slice file is used.

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

        return reorder_coords(slice_df)

    def simulate(self, design: DesignType, config: dict[str, Any] | None = None, mpicores: int = 4) -> npt.NDArray:
        """Simulates the performance of an airfoil design.

        Args:
            design (dict): The design to simulate.
            config (dict): A dictionary with configuration (e.g., boundary conditions, filenames) for the simulation.
            mpicores (int): The number of MPI cores to use in the simulation.

        Returns:
            dict: The performance of the design - each entry of the dict corresponds to a named objective value.
        """
        if isinstance(design["angle_of_attack"], np.ndarray):
            design["angle_of_attack"] = design["angle_of_attack"][0]

        # pre-process the design and run the simulation

        # Prepares the airfoil_analysis.py script with the simulation configuration
        conditions = self.Conditions()
        config = config or {}
        args = cli_interface.AnalysisParameters(
            alpha=design["angle_of_attack"],
            altitude=config.get("altitude", 10000),
            temperature=config.get("temperature", 300),
            reynolds=config.get("reynolds", conditions.reynolds),
            mach=config.get("mach", conditions.mach),
            use_altitude=config.get("use_altitude", False),
            output_dir=config.get("output_dir", self.__docker_study_dir + "/output/"),
            mesh_fname=config.get("mesh_fname", self.__docker_study_dir + "/design.cgns"),
            task=cli_interface.Task[config["task"]] if "task" in config else cli_interface.Task.ANALYSIS,
        )
        self.__design_to_simulator_input(design, mach=args.mach, reynolds=args.reynolds, temperature=args.temperature)

        # Launches a docker container with the airfoil_analysis.py script
        # The script takes a mesh and ffd and performs an optimization
        bash_command = f"source /home/mdolabuser/.bashrc_mdolab && cd {self.__docker_base_dir} && mpirun -np {mpicores} python -m mpi4py {self.__docker_study_dir}/airfoil_analysis.py '{json.dumps(args.to_dict())}'"
        assert self.container_id is not None, "Container ID is not set"
        container.run(
            command=["/bin/bash", "-c", bash_command],
            image=self.container_id,
            name="machaero",
            mounts=[(self.__local_base_directory, self.__docker_base_dir)],
            sync_uid=True,
        )

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
        if isinstance(starting_point["angle_of_attack"], np.ndarray):
            starting_point["angle_of_attack"] = starting_point["angle_of_attack"][0]

        # pre-process the design and run the simulation
        filename = "candidate_design"

        # Prepares the optimize_airfoil.py script with the optimization configuration
        fields = {f.name for f in dataclasses.fields(cli_interface.OptimizeParameters)}
        config = {key: val for key, val in (config or {}).items() if key in fields}
        if "area_initial" not in config:
            raise ValueError("optimize(): config is missing the required parameter 'area_initial'")
        if "opt" in config:
            config["opt"] = cli_interface.Algorithm[config["opt"]]
        args = cli_interface.OptimizeParameters(
            **{
                **dataclasses.asdict(self.Conditions()),
                "alpha": starting_point["angle_of_attack"],
                "altitude": 10000,
                "temperature": 300,  # should specify either mach + altitude or mach + reynolds + reynoldsLength (default to 1) + temperature
                "use_altitude": False,
                "opt": cli_interface.Algorithm.SLSQP,
                "opt_options": {},
                "output_dir": self.__docker_study_dir + "/output",
                "ffd_fname": self.__docker_study_dir + "/" + filename + "_ffd.xyz",
                "mesh_fname": self.__docker_study_dir + "/" + filename + ".cgns",
                "area_input_design": calc_area(starting_point["coords"]),
                **config,
            },
        )
        self.__design_to_simulator_input(
            starting_point, reynolds=args.reynolds, mach=args.reynolds, temperature=args.temperature, filename=filename
        )

        # Launches a docker container with the optimize_airfoil.py script
        # The script takes a mesh and ffd and performs an optimization
        bash_command = f"source /home/mdolabuser/.bashrc_mdolab && cd {self.__docker_base_dir} && mpirun -np {mpicores} python -m mpi4py {self.__docker_study_dir}/airfoil_opt.py '{json.dumps(args.to_dict())}'"
        assert self.container_id is not None, "Container ID is not set"
        container.run(
            command=["/bin/bash", "-c", bash_command],
            image=self.container_id,
            name="machaero",
            mounts=[(self.__local_base_directory, self.__docker_base_dir)],
            sync_uid=True,
        )

        # post process -- extract the shape and objective values
        optisteps_history = []
        history = History(self.__local_study_dir + "/output/opt.hst")
        call_counters = history.getCallCounters()
        iters = list(map(int, call_counters)) if call_counters is not None else []

        for i in range(len(iters)):
            vals = history.read(int(iters[i]))
            if vals is not None and "funcs" in vals and "obj" in vals["funcs"] and not vals["fail"]:
                values = history.getValues(names=["obj"], callCounters=[i], allowSens=False, major=False, scale=True)
                if values is not None and "obj" in values:
                    objective = values["obj"]
                    # flatten objective if it is a list
                    obj_np = np.array(objective)
                    if obj_np.ndim > 1:
                        obj_np = obj_np.flatten()
                    optisteps_history.append(OptiStep(obj_values=obj_np, step=vals["iter"]))

        history.close()

        opt_coords = self.simulator_output_to_design()

        return {"coords": opt_coords, "angle_of_attack": starting_point["angle_of_attack"]}, optisteps_history

    def render(self, design: DesignType, *, open_window: bool = False, save: bool = False) -> Figure:
        """Renders the design in a human-readable format.

        Args:
            design (dict): The design to render.
            open_window (bool): If True, opens a window with the rendered design.
            save (bool): If True, saves the rendered design to a file in the study directory.

        Returns:
            Figure: The rendered design.
        """
        fig, ax = plt.subplots()
        coords = design["coords"]
        alpha = design["angle_of_attack"]
        ax.scatter(coords[0], coords[1], s=10, alpha=0.7)
        ax.set_title(r"$\alpha$=" + str(np.round(alpha, 2)) + r"$^\circ$")
        ax.axis("equal")
        ax.axis("off")
        ax.set_xlim((-0.005, 1.005))

        if open_window:
            plt.show()
        if save:
            plt.savefig(self.__local_study_dir + "/airfoil.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return fig

    def render_optisteps(self, optisteps_history: list[OptiStep], *, open_window: bool = False, save: bool = False) -> Any:
        """Renders the optimization step history.

        Args:
            optisteps_history (list[OptiStep]): The optimization steps to render.
            open_window (bool): If True, opens a window with the rendered design.
            save (bool): If True, saves the rendered design to a file in the study directory.

        Returns:
            Any: Rendered optimization step history.
        """
        fig, ax = plt.subplots()
        steps = np.array([step.step for step in optisteps_history])
        objectives = np.array([step.obj_values[0][0] for step in optisteps_history])
        ax.plot(steps, objectives, label="Drag Coefficient")
        ax.set_title("Optimization Steps")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Drag counts")
        if open_window:
            plt.show()
        if save:
            plt.savefig(self.__local_study_dir + "/optisteps.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return fig, ax

    def random_design(self, dataset_split: str = "train", design_key: str = "initial_design") -> tuple[dict[str, Any], int]:
        """Samples a valid random initial design.

        Args:
            dataset_split (str): The key to use for the dataset. Defaults to "train".
            design_key (str): The key to use for the design in the dataset.
                Defaults to "initial_design".

        Returns:
            tuple[dict[str, Any], int]: The valid random design and the index of the design in the dataset.
        """
        rnd = self.np_random.integers(low=0, high=len(self.dataset[dataset_split][design_key]), dtype=int)
        initial_design = self.dataset[dataset_split][design_key][rnd]
        return {"coords": np.array(initial_design["coords"]), "angle_of_attack": initial_design["angle_of_attack"]}, rnd


if __name__ == "__main__":
    # Initialize the problem

    problem = Airfoil(seed=0)

    # Retrieve the dataset
    dataset = problem.dataset

    # Get random initial design and optimized conditions from the dataset + the index
    design, idx = problem.random_design()

    # Get the config conditions from the dataset
    config = dataset["train"].select_columns(problem.conditions_keys)[idx]

    # Simulate the design
    print("Simulation results: ", problem.simulate(design, config=config, mpicores=8))

    # Cleanup the study directory; will delete the previous contents from simulate in this case
    problem.reset(seed=1, cleanup=True)

    # Get design and conditions from the dataset, render design
    opt_design, optisteps_history = problem.optimize(design, config=config, mpicores=8)
    print("Optimized design: ", opt_design)
    print("Optimization history: ", optisteps_history)

    # Render the final optimized design
    problem.render(opt_design, open_window=False, save=True)
