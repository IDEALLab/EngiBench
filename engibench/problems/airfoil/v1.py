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
import tarfile
from typing import Annotated, Any, ClassVar

from datasets import load_dataset
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
from engibench.core import SimulationResult
from engibench.problems.airfoil.pyopt_history import History
from engibench.problems.airfoil.templates_v1 import cli_interface
from engibench.problems.airfoil.utils import calc_area
from engibench.problems.airfoil.utils import calc_off_wall_distance
from engibench.problems.airfoil.utils import reorder_coords
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

    # Coordinates follow the dataset convention: 192 points (x/c, y/c) ordered
    # TE -> upper -> LE -> lower -> TE, i.e. shape (192, 2).
    design_space = spaces.Dict(
        {
            "coords": spaces.Box(low=-1.0, high=1.0, shape=(192, 2), dtype=np.float32),
            "angle_of_attack": spaces.Box(low=-1.0, high=10.0, shape=(1,), dtype=np.float32),
        }
    )
    # The three released dataset tiers share the same case ids / conditions / objectives.
    # ``dataset_id`` (the basic tier) drives ``self.dataset``; ``load_variant`` loads the others.
    dataset_variants: ClassVar[dict[str, str]] = {
        "basic": "Cashen/optiwing-airfoil-2d-v0",
        "surface": "Cashen/optiwing-airfoil-2d-surface-v0",
        "full": "Cashen/optiwing-airfoil-2d-full-v0",
    }
    dataset_id = dataset_variants["basic"]

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

    def load_variant(self, variant: str = "basic", **load_kwargs: Any) -> Any:
        """Load one of the released v1 dataset tiers from the Hugging Face Hub.

        The three tiers share the same ``case_id`` key, conditions and scalar objectives:

        - ``"basic"`` -- initial/optimal geometry + conditions + scalar objectives (this is also
          ``self.dataset``);
        - ``"surface"`` -- adds per-surface-node flow fields (cp, pressure, skin friction, ...) on
          the initial and optimal designs;
        - ``"full"`` -- the complete optimization record: every iteration with full surface fields
          and the adjoint shape/AoA sensitivities (large, ~3.6 GB).

        Args:
            variant: One of ``"basic"``, ``"surface"``, ``"full"``.
            **load_kwargs: Forwarded to ``datasets.load_dataset`` (e.g. ``split=...``, ``streaming=True``).

        Returns:
            The loaded Hugging Face dataset.
        """
        if variant not in self.dataset_variants:
            raise ValueError(f"Unknown dataset variant {variant!r}; choose from {sorted(self.dataset_variants)}.")
        return load_dataset(self.dataset_variants[variant], **load_kwargs)

    @staticmethod
    def _coords_2xn(coords: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return airfoil coordinates as a ``(2, N)`` array (x row, y row).

        Accepts either the dataset ``(N, 2)`` convention or an already-``(2, N)`` array.
        """
        arr = np.asarray(coords, dtype=float)
        return arr if arr.shape[0] == 2 else arr.T  # noqa: PLR2004  -- 2 rows == (2, N)

    @staticmethod
    def _prep_coords_for_mesh(coords: npt.ArrayLike, tol: float = 1e-9) -> npt.NDArray[np.float64]:
        """Return a mesh-ready ``(2, N)`` open contour from any airfoil coordinate array.

        The released dataset stores each section as a *closed* contour (first point repeated
        at the end) on a 192-point cosine grid; such coincident points make prefoil's spline
        fit singular. This removes coincident consecutive points and any closing duplicate so
        the section can be re-meshed, while preserving the geometry (no rescaling / shifting).
        """
        pts = Airfoil._coords_2xn(coords).T  # (N, 2)
        seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        keep = np.concatenate(([True], seg > tol))  # drop points coincident with the previous one
        pts = pts[keep]
        if len(pts) > 1 and np.hypot(*(pts[0] - pts[-1])) <= tol:
            pts = pts[:-1]  # drop a closing duplicate (open the contour)
        return pts.T

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

        # Coords arrive in the dataset (N, 2) convention (a closed contour). Open it and remove
        # coincident points so prefoil can build the section spline. The section is already
        # chord-normalized/blunt, so pre_process feeds it directly without re-blunting
        # (input_blunted=True) -- this best reproduces the stored aerodynamics.
        prepped = self._prep_coords_for_mesh(design["coords"])
        args = cli_interface.PreprocessParameters(
            design_fname=f"{self.__docker_study_dir}/{filename}.dat",
            tmp_xyz_fname=tmp,
            mesh_fname=self.__docker_study_dir + "/" + filename + ".cgns",
            ffd_fname=self.__docker_study_dir + "/" + filename + "_ffd",
            s0=s0,
            input_blunted=True,
        )

        np.savetxt(self.__local_study_dir + "/" + filename + ".dat", prepped.transpose())

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

    @property
    def study_output_dir(self) -> str:
        """Local path to the current study's ``output`` directory.

        Holds the per-run artifacts (``opt.hst``, ``IPOPT.out``, ``final_abs_volume.npy``,
        the loose baseline/optimized section files and the tarred intermediates).
        """
        return self.__local_study_dir + "/output"

    @staticmethod
    def _parse_slice_lines(
        lines: list[str],
    ) -> tuple[list[str], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """Parse a Tecplot FELINESEG slice file generically (robust to the variable count).

        Args:
            lines: The lines of a ``fc_*_slices.dat`` file.

        Returns:
            A tuple ``(var_names, node_data, conn_data)`` where ``node_data`` is
            ``(n_nodes, n_vars)`` and ``conn_data`` is ``(n_elements, 2)``.
        """
        var_names: list[str] = []
        n_nodes = n_elements = None
        data_start = None
        for i, line in enumerate(lines):
            if "variables" in line.lower() and '"' in line:
                var_names = re.findall(r'"([^"]+)"', line)
            match = re.search(r"Nodes\s*=\s*(\d+).*Elements\s*=\s*(\d+)", line)
            if match:
                n_nodes, n_elements = int(match.group(1)), int(match.group(2))
            if "DATAPACKING" in line:
                data_start = i + 1
                break
        if n_nodes is None or n_elements is None or data_start is None:
            raise ValueError("Could not parse slice header (Nodes/Elements/DATAPACKING).")

        node_rows = lines[data_start : data_start + n_nodes]
        conn_rows = lines[data_start + n_nodes : data_start + n_nodes + n_elements]
        node_data = np.array([[float(v) for v in row.split()] for row in node_rows])
        conn_data = np.array([[int(float(v)) for v in row.split()] for row in conn_rows])
        if not var_names or len(var_names) != node_data.shape[1]:
            var_names = [f"var_{j}" for j in range(node_data.shape[1])]
        return var_names, node_data, conn_data

    @staticmethod
    def _coords_from_parsed(
        node_data: npt.NDArray[np.float64], conn_data: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float32]:
        """Reorder parsed slice data into clean ``(N, 2)`` airfoil coordinates (dataset convention)."""
        slice_df = pd.DataFrame({"CoordinateX": node_data[:, 0], "CoordinateY": node_data[:, 1]})
        nodes_df = pd.DataFrame({"NodeC1": conn_data[:, 0], "NodeC2": conn_data[:, 1]})
        return reorder_coords(pd.concat([slice_df, nodes_df], axis=1)).T  # reorder_coords gives (2, N)

    def simulator_output_to_design(self, simulator_output: str | None = None) -> npt.NDArray[np.float32]:
        """Converts a slice file to a design, robust to the v1 (richer) surface-variable set.

        The v1 solver writes many more surface fields than v0, so the slice/section files have
        a variable (and column) count that v0's fixed parser cannot read. This parser instead
        reads the Tecplot FELINESEG header, node-coordinate block, and connectivity block
        generically, then reuses ``reorder_coords``.

        Args:
            simulator_output (str): Slice filename to read. If None, the latest slice file is used.

        Returns:
            np.ndarray: The reordered ``(N, 2)`` airfoil coordinates (dataset convention).
        """
        output_dir = self.study_output_dir
        if simulator_output is None:
            files = [f for f in os.listdir(output_dir) if f.endswith("_slices.dat")]
            file_numbers = [int(f.split("_")[1]) for f in files]
            simulator_output = files[file_numbers.index(max(file_numbers))]

        with open(os.path.join(output_dir, simulator_output)) as fh:
            _, node_data, conn_data = self._parse_slice_lines(fh.readlines())
        return self._coords_from_parsed(node_data, conn_data)

    def optimization_trajectory(self, *, include_surface: bool = False) -> list[dict[str, Any]]:
        """Returns the per-iteration designs (and optionally surface fields) from the last ``optimize()``.

        ``optimize()`` writes one section file per evaluated design: the baseline (``fc_000``) and
        the optimized design stay loose, while the intermediate ones are bundled in
        ``fc_intermediate_slices.dat.tar.gz``. This reads all of them (loose + tarred) and returns
        the full optimization trajectory of geometries -- rich data for trajectory/multimodal models.
        It complements the scalar objective trajectory returned by ``optimize()`` (``optisteps``).

        Args:
            include_surface: If True, also include the full per-node surface fields (cp, Mach,
                skin friction, separation sensors, ...) as a DataFrame in raw slice order.

        Returns:
            A list (ordered by design index) of dicts with keys ``index`` (int), ``coords``
            (``(N, 2)`` reordered airfoil coordinates, dataset convention) and, if
            ``include_surface`` is True, ``surface`` (a ``pandas.DataFrame`` of every surface
            variable in the slice file).
        """
        output_dir = self.study_output_dir
        slices: dict[int, list[str]] = {}

        # Loose slice files (baseline + optimized).
        for fname in os.listdir(output_dir):
            if fname.startswith("fc_") and fname.endswith("_slices.dat"):
                with open(os.path.join(output_dir, fname)) as fh:
                    slices[int(fname.split("_")[1])] = fh.readlines()

        # Tarred intermediate slice files.
        tar_path = os.path.join(output_dir, "fc_intermediate_slices.dat.tar.gz")
        if os.path.exists(tar_path):
            with tarfile.open(tar_path) as tar:
                for member in tar.getmembers():
                    if member.name.endswith("_slices.dat"):
                        extracted = tar.extractfile(member)
                        if extracted is not None:
                            idx = int(os.path.basename(member.name).split("_")[1])
                            slices[idx] = extracted.read().decode().splitlines(keepends=True)

        trajectory: list[dict[str, Any]] = []
        for idx in sorted(slices):
            var_names, node_data, conn_data = self._parse_slice_lines(slices[idx])
            entry: dict[str, Any] = {"index": idx, "coords": self._coords_from_parsed(node_data, conn_data)}
            if include_surface:
                entry["surface"] = pd.DataFrame(node_data, columns=var_names)
            trajectory.append(entry)
        return trajectory

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
                "area_input_design": calc_area(self._coords_2xn(starting_point["coords"]).astype(np.float32)),
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

    def render(self, design: DesignType, *, open_window: bool = False, save: bool = False) -> Figure:
        """Renders an airfoil design (handles the dataset ``(N, 2)`` coordinate convention).

        Args:
            design (dict): The design to render.
            open_window (bool): If True, opens a window with the rendered design.
            save (bool): If True, saves the rendered design under the study directory.

        Returns:
            Figure: The rendered design.
        """
        fig, ax = plt.subplots()
        coords = self._coords_2xn(design["coords"])
        alpha = float(np.asarray(design["angle_of_attack"]).ravel()[0])
        ax.plot(coords[0], coords[1], lw=1)
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
