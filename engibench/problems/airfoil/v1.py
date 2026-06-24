"""Airfoil problem - Version 1 (v1).

v1 aligns the EngiBench airfoil problem with the updated 2D dataset-collection
pipeline. Relative to v0 it:

- adds **temperature** as a sampled conditioning variable,
- runs the optimization with **IPOPT** (gradient-based, adjoint gradients),
- treats lift as an **inequality** constraint (``cl >= cl_target`` with a generous
  upper guardrail) instead of an equality,
- **widens the FFD shape design-variable bounds to +/-0.10** (from +/-0.025),
- applies the reference ADflow solver settings (RANS + QCR, NK finisher, an internal
  per-Mach ANK schedule, richer adjoint settings) so the CFD is configured exactly as
  the dataset was collected, while abstracting the solver tuning away from the user
  (advanced users can still override it),
- emits richer debugging artifacts (full ``opt.hst`` with stored sensitivities,
  per-iteration surface/section files, ``final_abs_volume.npy``, ``IPOPT.out``).

The geometric area / thickness constraints are kept identical to v0 (the volume
constraint is scaled against a fixed baseline ``area_input_design`` so optimized areas
stay comparable across designs).

Note: the reference pipeline's automatic restart/retune logic is coupled to its external
Slurm harness and has no analog in EngiBench's single-shot container model, so it is not
ported here.
"""

import dataclasses
from dataclasses import dataclass
from dataclasses import field
import json
import os
import re
from typing import Annotated, Any

from gymnasium import spaces
import numpy as np
import numpy.typing as npt
import pandas as pd

from engibench.constraint import bounded
from engibench.constraint import constraint
from engibench.constraint import IMPL
from engibench.constraint import THEORY
from engibench.core import ObjectiveDirection
from engibench.core import OptiStep
from engibench.core import SimulationResult
from engibench.problems.airfoil.pyopt_history import History
from engibench.problems.airfoil.templates_v1 import cli_interface
from engibench.problems.airfoil.utils import calc_area
from engibench.problems.airfoil.utils import calc_off_wall_distance
from engibench.problems.airfoil.utils import reorder_coords
from engibench.problems.airfoil.utils import scale_coords
from engibench.problems.airfoil.v0 import Airfoil as Airfoil_v0
from engibench.utils import container
from engibench.utils.files import clone_dir

DesignType = dict[str, Any]


class Airfoil(Airfoil_v0):
    r"""Airfoil 2D shape optimization problem - Version 1 (v1).

    Same MACH-Aero (ADflow + pyOptSparse) backend as v0, run inside the
    ``mdolab/public`` container, but with the updated solver settings, IPOPT optimizer,
    inequality lift constraint, widened FFD bounds, and sampled temperature used to
    collect the v1 dataset. See the module docstring for the full list of changes.
    """

    version = 1
    objectives: tuple[tuple[str, ObjectiveDirection], ...] = (
        ("cd", ObjectiveDirection.MINIMIZE),
        ("cl", ObjectiveDirection.MAXIMIZE),
    )

    design_space = spaces.Dict(
        {
            "coords": spaces.Box(low=-1.0, high=1.0, shape=(2, 207), dtype=np.float32),
            "angle_of_attack": spaces.Box(low=-1.0, high=10.0, shape=(1,), dtype=np.float32),
        }
    )
    dataset_id = "IDEALLab/airfoil_v1"

    @dataclass
    class Conditions:
        """Conditions (sampled per case)."""

        mach: Annotated[
            float, bounded(lower=0.0).category(IMPL), bounded(lower=0.1, upper=1.2).warning().category(IMPL)
        ] = 0.734
        """Mach number"""
        reynolds: Annotated[
            float, bounded(lower=0.0).category(IMPL), bounded(lower=1e5, upper=1e9).warning().category(IMPL)
        ] = 6.5e6
        """Reynolds number"""
        temperature: Annotated[
            float, bounded(lower=0.0).category(IMPL), bounded(lower=200.0, upper=320.0).warning().category(IMPL)
        ] = 300.0
        """Freestream static temperature [K]"""
        area_initial: float = float("NAN")
        """actual initial airfoil area"""
        area_ratio_min: Annotated[float, bounded(lower=0.0, upper=1.2).category(THEORY)] = 0.7
        """Minimum ratio the initial area is allowed to decrease to i.e minimum_area = area_initial*area_ratio_min"""
        cl_target: float = 0.824
        """Target lift coefficient (the realized lift sits at ~cl_target)."""

    conditions = Conditions()  # type: ignore[assignment]  # v1 redefines the nested Conditions

    @dataclass
    class Config(Conditions):
        """Structured representation of configuration parameters for a numerical computation."""

        alpha: Annotated[float, bounded(lower=-1.0, upper=10.0).category(THEORY)] = 2.5
        altitude: float = 10000.0
        use_altitude: bool = False
        output_dir: str | None = None
        mesh_fname: str | None = None
        task: str = "analysis"
        opt: str = "IPOPT"
        opt_options: dict = field(default_factory=dict)
        ffd_fname: str | None = None
        area_input_design: float | None = None
        # Advanced (None -> sensible defaults): restrict written surface fields / override
        # the internal ADflow solver schedule.
        surface_variables: list[str] | None = None
        solver_options: dict | None = None

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
        """Initializes the v1 Airfoil problem.

        Args:
            seed (int): The random seed for the problem.
            base_directory (str, optional): The base directory for the problem. If None, the current directory is used.
        """
        super().__init__(seed=seed, base_directory=base_directory)
        # Point the per-study template clone at the v1 templates. Both v0 and v1 classes
        # are named ``Airfoil`` so the name-mangled attribute target is identical.
        self.__local_template_dir = os.path.dirname(os.path.abspath(__file__)) + "/templates_v1"

    def __design_to_simulator_input(
        self, design: DesignType, mach: float, reynolds: float, temperature: float, filename: str = "design"
    ) -> str:
        """Converts a design to a simulator input (mesh + FFD) using the v1 mesh settings.

        Args:
            design (dict): The design to convert.
            mach: mach number.
            reynolds: reynolds number.
            temperature: temperature.
            filename (str): The filename to save the design to.
        """
        clone_dir(source_dir=self.__local_template_dir, target_dir=self.__local_study_dir)
        os.makedirs(os.path.join(self.__local_study_dir, "mpi_tmp"), exist_ok=True)

        tmp = os.path.join(self.__docker_study_dir, "tmp")
        s0 = calc_off_wall_distance(mach=mach, reynolds=reynolds, freestreamTemp=temperature)

        # The v1 baselines are already blunt (x in [0, 0.98]); feed them through directly.
        scaled_design, input_blunted = scale_coords(design["coords"], blunted=True, xcut=0.98)
        args = cli_interface.PreprocessParameters(
            design_fname=f"{self.__docker_study_dir}/{filename}.dat",
            tmp_xyz_fname=tmp,
            mesh_fname=self.__docker_study_dir + "/" + filename + ".cgns",
            ffd_fname=self.__docker_study_dir + "/" + filename + "_ffd",
            s0=s0,
            input_blunted=input_blunted,
        )

        np.savetxt(self.__local_study_dir + "/" + filename + ".dat", scaled_design.transpose())

        bash_command = (
            f"source /home/mdolabuser/.bashrc_mdolab && cd {self.__docker_base_dir} && "
            f"python {self.__docker_study_dir}/pre_process.py '{json.dumps(args.to_dict())}'"
        )
        assert self.container_id is not None, "Container ID is not set"
        container.run(
            command=["/bin/bash", "-c", bash_command],
            image=self.container_id,
            name="machaero",
            mounts=[(self.__local_base_directory, self.__docker_base_dir)],
            env={"TMPDIR": os.path.join(self.__docker_study_dir, "mpi_tmp")},
            sync_uid=True,
        )
        return filename

    def simulator_output_to_design(self, simulator_output: str | None = None) -> npt.NDArray[np.float32]:
        """Converts a slice file to a design, robust to the v1 (richer) surface-variable set.

        The v1 solver writes many more surface fields than v0, so the slice/section files have
        a variable (and column) count that v0's fixed parser cannot read. This parser instead
        reads the Tecplot FELINESEG ``Nodes``/``Elements`` header, the node-coordinate block, and
        the connectivity block generically, then reuses ``reorder_coords``.

        Args:
            simulator_output (str): Slice filename to read. If None, the latest slice file is used.

        Returns:
            np.ndarray: The reordered (x, y) airfoil coordinates.
        """
        output_dir = self.__local_study_dir + "/output"
        if simulator_output is None:
            files = [f for f in os.listdir(output_dir) if f.endswith("_slices.dat")]
            file_numbers = [int(f.split("_")[1]) for f in files]
            simulator_output = files[file_numbers.index(max(file_numbers))]

        with open(os.path.join(output_dir, simulator_output)) as fh:
            lines = fh.readlines()

        n_nodes = n_elements = None
        data_start = None
        for i, line in enumerate(lines):
            match = re.search(r"Nodes\s*=\s*(\d+).*Elements\s*=\s*(\d+)", line)
            if match:
                n_nodes, n_elements = int(match.group(1)), int(match.group(2))
            if "DATAPACKING" in line:
                data_start = i + 1
                break
        if n_nodes is None or n_elements is None or data_start is None:
            raise ValueError(f"Could not parse slice header in {simulator_output}")

        node_rows = lines[data_start : data_start + n_nodes]
        conn_rows = lines[data_start + n_nodes : data_start + n_nodes + n_elements]
        node_data = np.array([[float(v) for v in row.split()] for row in node_rows])
        conn_data = np.array([[int(float(v)) for v in row.split()] for row in conn_rows])

        # reorder_coords only needs the (x, y) coordinates and the node connectivity.
        slice_df = pd.DataFrame({"CoordinateX": node_data[:, 0], "CoordinateY": node_data[:, 1]})
        nodes_df = pd.DataFrame({"NodeC1": conn_data[:, 0], "NodeC2": conn_data[:, 1]})
        slice_df = pd.concat([slice_df, nodes_df], axis=1)
        return reorder_coords(slice_df)

    def simulate_verbose(
        self, design: DesignType, config: dict[str, Any] | None = None, mpicores: int = 4
    ) -> SimulationResult:
        """Simulates the performance of an airfoil design with the v1 solver settings.

        Args:
            design (dict): The design to simulate.
            config (dict): Boundary conditions / filenames for the simulation.
            mpicores (int): The number of MPI cores to use.

        Returns:
            `SimulationResult` with objective values [drag, lift].
        """
        if isinstance(design["angle_of_attack"], np.ndarray):
            design["angle_of_attack"] = design["angle_of_attack"][0]

        conditions = self.Conditions()
        config = config or {}
        args = cli_interface.AnalysisParameters(
            alpha=design["angle_of_attack"],
            altitude=config.get("altitude", 10000),
            temperature=config.get("temperature", conditions.temperature),
            reynolds=config.get("reynolds", conditions.reynolds),
            mach=config.get("mach", conditions.mach),
            use_altitude=config.get("use_altitude", False),
            output_dir=config.get("output_dir", self.__docker_study_dir + "/output/"),
            mesh_fname=config.get("mesh_fname", self.__docker_study_dir + "/design.cgns"),
            task=cli_interface.Task[config["task"]] if "task" in config else cli_interface.Task.ANALYSIS,
            surface_variables=config.get("surface_variables"),
            solver_options=config.get("solver_options"),
        )
        self.__design_to_simulator_input(design, mach=args.mach, reynolds=args.reynolds, temperature=args.temperature)

        bash_command = (
            f"source /home/mdolabuser/.bashrc_mdolab && cd {self.__docker_base_dir} && "
            f"mpirun -np {mpicores} python -m mpi4py {self.__docker_study_dir}/airfoil_analysis.py "
            f"'{json.dumps(args.to_dict())}'"
        )
        assert self.container_id is not None, "Container ID is not set"
        container.run(
            command=["/bin/bash", "-c", bash_command],
            image=self.container_id,
            name="machaero",
            mounts=[(self.__local_base_directory, self.__docker_base_dir)],
            env={"TMPDIR": os.path.join(self.__docker_study_dir, "mpi_tmp")},
            sync_uid=True,
        )

        outputs = np.load(self.__local_study_dir + "/output/outputs.npy")
        lift = float(outputs[3])
        drag = float(outputs[4])
        return SimulationResult(np.array([drag, lift]))

    def optimize(
        self, starting_point: DesignType, config: dict[str, Any] | None = None, mpicores: int = 4
    ) -> tuple[DesignType, list[OptiStep]]:
        """Optimizes the design of an airfoil with IPOPT and the v1 solver settings.

        Args:
            starting_point (dict): The starting point for the optimization.
            config (dict): Boundary conditions / filenames for the optimization.
            mpicores (int): The number of MPI cores to use.

        Returns:
            tuple[dict[str, Any], list[OptiStep]]: The optimized design and its history.
        """
        if isinstance(starting_point["angle_of_attack"], np.ndarray):
            starting_point["angle_of_attack"] = starting_point["angle_of_attack"][0]

        filename = "candidate_design"

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
                "use_altitude": False,
                "opt": cli_interface.Algorithm.IPOPT,
                "opt_options": {},
                "output_dir": self.__docker_study_dir + "/output",
                "ffd_fname": self.__docker_study_dir + "/" + filename + "_ffd.xyz",
                "mesh_fname": self.__docker_study_dir + "/" + filename + ".cgns",
                "area_input_design": calc_area(starting_point["coords"]),
                **config,
            },
        )
        self.__design_to_simulator_input(
            starting_point, reynolds=args.reynolds, mach=args.mach, temperature=args.temperature, filename=filename
        )

        bash_command = (
            f"source /home/mdolabuser/.bashrc_mdolab && cd {self.__docker_base_dir} && "
            f"mpirun -np {mpicores} python -m mpi4py {self.__docker_study_dir}/airfoil_opt.py "
            f"'{json.dumps(args.to_dict())}'"
        )
        assert self.container_id is not None, "Container ID is not set"
        container.run(
            command=["/bin/bash", "-c", bash_command],
            image=self.container_id,
            name="machaero",
            mounts=[(self.__local_base_directory, self.__docker_base_dir)],
            env={"TMPDIR": os.path.join(self.__docker_study_dir, "mpi_tmp")},
            sync_uid=True,
        )

        # Post-process: extract the optimization history and optimized shape.
        optisteps_history = []
        history = History(self.__local_study_dir + "/output/opt.hst")
        call_counters = history.getCallCounters()
        iters = list(map(int, call_counters)) if call_counters is not None else []

        for i in range(len(iters)):
            vals = history.read(int(iters[i]))
            if vals is not None and "funcs" in vals and "obj" in vals["funcs"] and not vals["fail"]:
                values = history.getValues(names=["obj"], callCounters=[i], allowSens=False, major=False, scale=True)
                if values is not None and "obj" in values:
                    obj_np = np.array(values["obj"])
                    if obj_np.ndim > 1:
                        obj_np = obj_np.flatten()
                    optisteps_history.append(OptiStep(obj_values=obj_np, step=vals["iter"]))

        opt_alpha_values = history.getValues(names=["alpha_fc"], callCounters=["last"], major=True)
        opt_alpha = (
            float(opt_alpha_values["alpha_fc"].flatten()[0])
            if opt_alpha_values and "alpha_fc" in opt_alpha_values and len(opt_alpha_values["alpha_fc"]) > 0
            else starting_point["angle_of_attack"]
        )
        history.close()

        opt_coords = self.simulator_output_to_design()
        return {"coords": opt_coords, "angle_of_attack": opt_alpha}, optisteps_history


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

    # Optimize the design
    opt_design, optisteps_history = problem.optimize(design, config=config, mpicores=8)
    print("Optimized design: ", opt_design)
    print("Optimization history: ", optisteps_history)

    # Render the final optimized design
    problem.render(opt_design, open_window=False, save=True)
