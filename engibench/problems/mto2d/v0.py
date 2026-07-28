"""Two-dimensional multiphysics topology optimization problem."""

import argparse
import dataclasses
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Annotated, Any
import warnings

from gymnasium import spaces
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from engibench.constraint import bounded
from engibench.constraint import check_field_constraints
from engibench.constraint import constraint
from engibench.constraint import Criticality
from engibench.constraint import greater_than
from engibench.constraint import IMPL
from engibench.constraint import less_than
from engibench.constraint import THEORY
from engibench.core import ObjectiveDirection
from engibench.core import OptiStep
from engibench.core import Problem
from engibench.core import SimulationResult
from engibench.problems.mto2d.model.design_io import FIXED_CELL_COUNT
from engibench.problems.mto2d.model.design_io import GAMMA_CELL_COUNT
from engibench.problems.mto2d.model.design_io import HALF_DESIGN_SHAPE
from engibench.problems.mto2d.model.design_io import half_to_full
from engibench.problems.mto2d.model.design_io import legacy_256_to_half
from engibench.problems.mto2d.model.design_io import LEGACY_DESIGN_SHAPE
from engibench.problems.mto2d.model.runner import MTO2DRunner
from engibench.problems.mto2d.model.runner import POWER_BOUND_DECREMENT_PER_ITERATION
from engibench.problems.mto2d.model.runner import RunnerSettings
from engibench.problems.mto2d.model.runner import SolverRun
from engibench.utils.upcast import upcast

MIN_VOLUME_FRACTION = FIXED_CELL_COUNT / GAMMA_CELL_COUNT
"""Smallest feasible all-cell volume fraction when the design domain is solid."""

SOURCE_CHECKOUT_PATH = Path(__file__).resolve().parents[3]
"""EngiBench source checkout containing this module."""

REPOSITORY_DATASET_PATH = SOURCE_CHECKOUT_PATH / "dataset_output" / "mto_2d_v0"
"""Preferred local converted dataset for the source-tree demonstration."""

LOCAL_RUNTIME_CONFIG_PATH = SOURCE_CHECKOUT_PATH.parent / ".artifacts" / "mto2d-docker.json"
"""Optional private Docker runtime config produced beside this source checkout."""

SOLVER_CONFIG_ENV_VAR = "ENGIBENCH_MTO2D_SOLVER_CONFIG"
"""Environment variable selecting a solver configuration JSON file."""


@dataclass
class MTO2DSimulationResult(SimulationResult):
    """Objectives and constraint diagnostics from a frozen MTO2D evaluation."""

    volume_constraint_residual: float
    """Solver-reported ``mean(gamma) - volume_fraction`` over all cells."""

    power_constraint_residual: float
    """Relative power residual ``power_dissipation / max_power_dissipation - 1``."""

    elapsed_time: float
    """Cumulative solver wall time in seconds."""

    status: str
    """Simulation completion status."""

    artifacts_path: str | None = None
    """Retained isolated run directory, when requested."""


@dataclass
class MTO2DOptimizationResult:
    """Detailed result for the MTO2D-specific :meth:`MTO2D.optimize_verbose` extension."""

    design: npt.NDArray[np.float32]
    history: list[OptiStep]
    volume_constraint_residuals: npt.NDArray[np.float64]
    active_power_bounds: npt.NDArray[np.float64]
    active_power_constraint_residuals: npt.NDArray[np.float64]
    power_constraint_residuals: npt.NDArray[np.float64]
    elapsed_times: npt.NDArray[np.float64]
    artifacts_path: str | None


class MTO2D(Problem[npt.NDArray]):
    """OpenFOAM-based 2D thermofluid topology optimization.

    A design is the non-redundant, visually oriented ``(400, 200)``
    fluid-density half-domain. ``gamma=0`` is solid and ``gamma=1`` is fluid.
    The right half is implied by symmetry and is created only for rendering.

    The built-in optimizer minimizes mean temperature with MMA while treating
    fluid volume and power dissipation as constraints. Power dissipation is
    nevertheless exposed as a second EngiBench objective for Pareto analysis.
    """

    version = 0
    objectives: tuple[tuple[str, ObjectiveDirection], ...] = (
        ("mean_temperature", ObjectiveDirection.MINIMIZE),
        ("power_dissipation", ObjectiveDirection.MINIMIZE),
    )

    @dataclass
    class Conditions:
        """Physical conditions represented in the dataset."""

        inlet_velocity: Annotated[
            float,
            less_than(0.0).category(THEORY),
            bounded(lower=-0.095, upper=-0.025).warning().category(IMPL),
        ] = -0.074
        """Signed inlet y-velocity in m/s; the nominal dataset range is [-0.095, -0.025]."""

        max_power_dissipation: Annotated[
            float,
            greater_than(0.0).category(THEORY),
            bounded(lower=50.0, upper=75.0).warning().category(IMPL),
        ] = 63.1
        """Dimensionless normalized power-dissipation bound.

        The retained solver divides physical dissipation by its exact
        ``D_normalization = 1.57572e-7``. The paper denotes the rounded
        reference scale ``J1 ≈ 1.58e-7``.
        """

        volume_fraction: Annotated[
            float,
            bounded(lower=MIN_VOLUME_FRACTION, upper=1.0).category(THEORY),
            bounded(lower=0.25, upper=0.70).warning().category(IMPL),
        ] = 0.61
        """Maximum all-cell fluid volume fraction."""

    conditions = Conditions()
    design_space = spaces.Box(low=0.0, high=1.0, shape=HALF_DESIGN_SHAPE, dtype=np.float32)
    dataset_id = "IDEALLab/mto_2d_v0"
    container_id = None

    @dataclass
    class Config(Conditions):
        """Solver configuration not treated as a physical dataset condition."""

        max_iter: Annotated[int, greater_than(0).category(IMPL)] = 200
        mode: str = "cold"
        mpi_cores: Annotated[int, greater_than(0).category(IMPL)] = 1
        case_template: str | None = None
        backend: str = "local"
        container_image: str | None = None
        driver_command: tuple[str, ...] = ()
        solver_executable: str = "../src_TF/EXEC"
        build_solver: bool = False
        timeout: float | None = None
        work_dir: str | None = None
        retain_artifacts: bool = False
        retain_on_failure: bool = True
        continuation_steps: int | None = None
        continuation_profile: str = "geometric"
        power_bound_start: float | None = None
        qu_start: float | None = None
        qu_final: Annotated[float, greater_than(0.0).category(IMPL)] = 0.019
        alpha_max_start: float | None = None
        alpha_max_final: Annotated[float, greater_than(0.0).category(IMPL)] = 5.0252e6
        heaviside_start: float | None = None
        heaviside_final: Annotated[float, greater_than(0.0).category(IMPL)] = 59.8
        movement_limit: Annotated[float, bounded(lower=0.0, upper=1.0).category(IMPL)] = 0.4

        @constraint(categories=IMPL)
        @staticmethod
        def valid_mode(mode: str) -> None:
            """Require a supported continuation initialization mode."""
            assert mode in {"cold", "warm"}, "Config.mode must be 'cold' or 'warm'"

        @constraint(categories=IMPL)
        @staticmethod
        def valid_backend(backend: str) -> None:
            """Require a supported runner backend."""
            assert backend in {"local", "container", "command"}, "Config.backend must be 'local', 'container', or 'command'"

        @constraint(categories=IMPL)
        @staticmethod
        def valid_continuation(max_iter: int, continuation_steps: int | None) -> None:
            """Require continuation intervals supported by the legacy solver."""
            if continuation_steps is None:
                return
            assert 1 <= continuation_steps <= max_iter, "Config.continuation_steps must be between 1 and max_iter"
            assert max_iter % continuation_steps == 0, "Config.max_iter must be divisible by continuation_steps"

        @constraint(categories=IMPL)
        @staticmethod
        def valid_timeout(timeout: float | None) -> None:
            """Require a positive optional process timeout."""
            assert timeout is None or timeout > 0.0, "Config.timeout must be positive"

        @constraint(categories=IMPL)
        @staticmethod
        def valid_power_bound_start(power_bound_start: float | None) -> None:
            """Require a positive optional initial power-dissipation bound."""
            assert power_bound_start is None or power_bound_start > 0.0, "Config.power_bound_start must be positive"

    def __init__(
        self,
        seed: int = 0,
        config: dict[str, Any] | None = None,
        *,
        dataset: Any | None = None,
        runner: MTO2DRunner | None = None,
    ) -> None:
        """Initialize MTO2D with optional local dataset and runner injection."""
        values = dict(config or {})
        if isinstance(values.get("driver_command"), list):
            values["driver_command"] = tuple(values["driver_command"])
        super().__init__(seed=seed)
        self.config = self.Config(**values)
        self._raise_config_errors(self.config)
        self.conditions = upcast(self.config)
        self._runner = runner or MTO2DRunner()
        self.last_solver_run: SolverRun | None = None
        if dataset is not None:
            self._dataset = dataset

    def reset(self, seed: int | None = None) -> None:
        """Reset the EngiBench random-number generator."""
        super().reset(seed)

    def simulate_verbose(
        self,
        design: npt.NDArray,
        config: dict[str, Any] | None = None,
    ) -> MTO2DSimulationResult:
        """Evaluate final physics once while bypassing sensitivity and MMA updates."""
        density = self._coerce_design(design)
        resolved = self._resolve_config(config)
        settings = self._runner_settings(resolved, max_iter=1)
        run = self._runner.run(density, settings, kind="simulate")
        self.last_solver_run = run
        mean_temperature = float(run.mean_temperature[-1])
        power_dissipation = float(run.power_dissipation[-1])
        return MTO2DSimulationResult(
            objective_values=np.array([mean_temperature, power_dissipation], dtype=np.float64),
            volume_constraint_residual=float(run.volume_residual[-1]),
            power_constraint_residual=power_dissipation / resolved.max_power_dissipation - 1.0,
            elapsed_time=float(run.elapsed_time[-1]),
            status="success",
            artifacts_path=run.artifacts_path,
        )

    def optimize(
        self,
        starting_point: npt.NDArray,
        config: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.float32], list[OptiStep]]:
        """Optimize a topology with adjoint sensitivities and MMA."""
        detailed = self.optimize_verbose(starting_point, config)
        return detailed.design, detailed.history

    def optimize_verbose(
        self,
        starting_point: npt.NDArray,
        config: dict[str, Any] | None = None,
    ) -> MTO2DOptimizationResult:
        """Optimize and return MTO2D-specific constraint and runtime details.

        Solver history row ``k`` describes the pre-update design evaluated in
        iteration ``k``. The returned final gamma is the subsequent MMA update,
        so call :meth:`simulate` when exact objectives for that returned field
        are required.
        """
        density = self._coerce_design(starting_point)
        resolved = self._resolve_config(config)
        active_power_bounds = self._active_power_bounds(resolved)
        if active_power_bounds[-1] > resolved.max_power_dissipation:
            warnings.warn(
                "Optimization stops before the legacy power-bound continuation reaches "
                f"max_power_dissipation={resolved.max_power_dissipation:.8g}; its final active bound is "
                f"{active_power_bounds[-1]:.8g}. Use more iterations, set power_bound_start to the final bound, "
                "or treat this as a runtime smoke test.",
                UserWarning,
                stacklevel=2,
            )
        run = self._runner.run(density, self._runner_settings(resolved), kind="optimize")
        self.last_solver_run = run
        history = [
            OptiStep(
                obj_values=np.array([mean_temperature, power_dissipation], dtype=np.float64),
                step=step,
            )
            for step, (mean_temperature, power_dissipation) in enumerate(
                zip(run.mean_temperature, run.power_dissipation, strict=True),
                start=1,
            )
        ]
        return MTO2DOptimizationResult(
            design=np.asarray(run.final_design, dtype=np.float32),
            history=history,
            volume_constraint_residuals=run.volume_residual.copy(),
            active_power_bounds=active_power_bounds,
            active_power_constraint_residuals=run.power_dissipation / active_power_bounds - 1.0,
            power_constraint_residuals=run.power_dissipation / resolved.max_power_dissipation - 1.0,
            elapsed_times=run.elapsed_time.copy(),
            artifacts_path=run.artifacts_path,
        )

    @staticmethod
    def _active_power_bounds(config: Config) -> npt.NDArray[np.float64]:
        """Return the power bound actually supplied to MMA at every iteration."""
        power_bound_start = config.power_bound_start
        if power_bound_start is None:
            power_bound_start = config.max_power_dissipation if config.mode == "warm" else 90.0
        iterations = np.arange(1, config.max_iter + 1, dtype=np.float64)
        return np.maximum(
            power_bound_start - POWER_BOUND_DECREMENT_PER_ITERATION * iterations,
            config.max_power_dissipation,
        )

    def random_design(
        self,
        dataset_split: str = "train",
        design_key: str = "optimal_design",
    ) -> tuple[npt.NDArray[np.float32], int]:
        """Sample a native design from a formatted dataset.

        Legacy ``256 x 256`` half-domain rows are accepted for migration and
        reconstructed lossily. New datasets should store native ``400 x 200``
        designs directly.
        """
        split = self.dataset[dataset_split]
        index = int(self.np_random.integers(0, len(split)))
        return self.design_from_dataset_value(split[index][design_key]), index

    @staticmethod
    def design_from_dataset_value(value: Any) -> npt.NDArray[np.float32]:
        """Convert a native, flattened, or legacy dataset value to the design space."""
        design = np.asarray(value, dtype=np.float32)
        if design.shape == (1, *LEGACY_DESIGN_SHAPE):
            design = design[0]
        if design.shape == LEGACY_DESIGN_SHAPE:
            warnings.warn(
                "Reconstructing a native MTO2D design from the lossy legacy 256 x 256 half-domain.",
                UserWarning,
                stacklevel=2,
            )
            design = legacy_256_to_half(design)
        elif design.shape != HALF_DESIGN_SHAPE and design.size == int(np.prod(HALF_DESIGN_SHAPE)):
            design = design.reshape(HALF_DESIGN_SHAPE)
        return MTO2D._coerce_design(design)

    def render(self, design: npt.NDArray, *, open_window: bool = False) -> tuple[Figure, Any]:
        """Render the symmetric ``400 x 400`` fluid-density field."""
        full = half_to_full(self._coerce_design(design))
        fig, ax = plt.subplots(figsize=(6, 6))
        image = ax.imshow(full, cmap="Blues", vmin=0.0, vmax=1.0, origin="upper")
        ax.axvline(HALF_DESIGN_SHAPE[1] - 0.5, color="black", linewidth=0.6, linestyle="--")
        ax.set_title("MTO2D density (0 = solid, 1 = fluid)")
        ax.set_xlabel("x cell")
        ax.set_ylabel("y cell")
        fig.colorbar(image, ax=ax, label="gamma")
        fig.tight_layout()
        if open_window:
            plt.show()
        return fig, ax

    @staticmethod
    def design_volume_residual(design: npt.NDArray, volume_fraction: float) -> float:
        """Estimate the raw all-cell volume residual before filtering/projection."""
        density = np.asarray(design, dtype=np.float64)
        if density.shape != HALF_DESIGN_SHAPE:
            raise ValueError(f"design must have shape {HALF_DESIGN_SHAPE}")
        return float((density.sum() + FIXED_CELL_COUNT) / GAMMA_CELL_COUNT - volume_fraction)

    @staticmethod
    def uniform_starting_design(volume_fraction: float) -> npt.NDArray[np.float32]:
        """Create a uniform design whose all-cell mean equals the requested volume.

        The correction accounts for the 6,400 fixed-fluid cells outside the
        80,000-cell design domain.
        """
        design_fraction = (GAMMA_CELL_COUNT * volume_fraction - FIXED_CELL_COUNT) / int(np.prod(HALF_DESIGN_SHAPE))
        if not 0.0 <= design_fraction <= 1.0:
            raise ValueError("volume_fraction is incompatible with the fixed-fluid region")
        return np.full(HALF_DESIGN_SHAPE, design_fraction, dtype=np.float32)

    @staticmethod
    def _raise_config_errors(config: Config) -> None:
        errors = check_field_constraints(config).by_criticality(Criticality.Error)
        if errors:
            raise ValueError(str(errors))

    def _resolve_config(self, overrides: dict[str, Any] | None) -> Config:
        if self.config is None:
            raise RuntimeError("MTO2D solver configuration is not initialized")
        values = dataclasses.asdict(self.config)
        values.update(overrides or {})
        if isinstance(values.get("driver_command"), list):
            values["driver_command"] = tuple(values["driver_command"])
        resolved = self.Config(**values)
        self._raise_config_errors(resolved)
        return resolved

    @staticmethod
    def _coerce_design(design: npt.NDArray) -> npt.NDArray[np.float32]:
        density = np.asarray(design, dtype=np.float32)
        if density.shape != HALF_DESIGN_SHAPE:
            raise ValueError(f"MTO2D design must have shape {HALF_DESIGN_SHAPE}; got {density.shape}")
        if not np.all(np.isfinite(density)):
            raise ValueError("MTO2D design must contain only finite values")
        if np.any((density < 0.0) | (density > 1.0)):
            raise ValueError("MTO2D design values must lie in [0, 1]")
        return np.ascontiguousarray(density)

    @staticmethod
    def _runner_settings(config: Config, *, max_iter: int | None = None) -> RunnerSettings:
        values = dataclasses.asdict(config)
        fields = {field.name for field in dataclasses.fields(RunnerSettings)}
        values = {key: value for key, value in values.items() if key in fields}
        if max_iter is not None:
            values["max_iter"] = max_iter
        values["container_image"] = values["container_image"] or os.environ.get("ENGIBENCH_MTO2D_IMAGE")
        return RunnerSettings(**values)


def _load_demo_dataset(
    source: str | Path | None,
    *,
    default_dataset_id: str,
) -> tuple[Any, str]:
    """Load a local saved DatasetDict or a dataset from the Hugging Face Hub."""
    from datasets import load_dataset  # noqa: PLC0415
    from datasets import load_from_disk  # noqa: PLC0415

    resolved_source: str | Path
    if source is None:
        resolved_source = REPOSITORY_DATASET_PATH if REPOSITORY_DATASET_PATH.is_dir() else default_dataset_id
    else:
        resolved_source = source

    local_path = Path(resolved_source).expanduser()
    if local_path.is_dir():
        resolved_path = local_path.resolve()
        return load_from_disk(str(resolved_path)), str(resolved_path)
    if isinstance(resolved_source, Path):
        raise FileNotFoundError(f"local dataset directory does not exist: {local_path}")
    return load_dataset(resolved_source), resolved_source


def _load_solver_config_file(path: str | Path) -> dict[str, Any]:
    """Read one JSON object containing solver-only configuration."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"solver config must contain a JSON object: {config_path}")
    return config


def _read_solver_config(
    path: str | Path | None,
    *,
    auto_discover: bool = False,
) -> dict[str, Any]:
    """Resolve CLI solver configuration without changing :class:`MTO2D` defaults.

    An explicit path wins. For opt-in CLI simulation, a path selected through
    ``ENGIBENCH_MTO2D_SOLVER_CONFIG`` is next, followed by the private runtime
    config beside a Git source checkout. Individual case/image environment
    variables may override only that automatically discovered local config.
    """
    if path is not None:
        return _load_solver_config_file(path)
    if not auto_discover:
        return {}

    environment_path = os.environ.get(SOLVER_CONFIG_ENV_VAR)
    if environment_path:
        return _load_solver_config_file(environment_path)

    if not (SOURCE_CHECKOUT_PATH / ".git").exists() or not LOCAL_RUNTIME_CONFIG_PATH.is_file():
        return {}

    config = _load_solver_config_file(LOCAL_RUNTIME_CONFIG_PATH)
    if case_template := os.environ.get("ENGIBENCH_MTO2D_CASE_TEMPLATE"):
        config["case_template"] = case_template
    if container_image := os.environ.get("ENGIBENCH_MTO2D_IMAGE"):
        config["container_image"] = container_image
    return config


def main(  # noqa: PLR0913
    problem_type: type[MTO2D] = MTO2D,
    *,
    dataset: Any | None = None,
    dataset_source: str | Path | None = None,
    split: str = "train",
    index: int = 0,
    seed: int = 0,
    solver_config: dict[str, Any] | None = None,
    run_simulation: bool = False,
    render_output: str | Path | None = None,
    open_window: bool = False,
    runner: MTO2DRunner | None = None,
) -> MTO2DSimulationResult | None:
    """Render one real dataset design and optionally evaluate it.

    Simulation is deliberately opt-in because it requires the external
    OpenFOAM runtime and can be expensive. Dataset-row conditions always
    override the physical-condition defaults in ``solver_config``.
    """
    if dataset is not None and dataset_source is not None:
        raise ValueError("pass either dataset or dataset_source, not both")

    source_label = "<injected dataset>"
    if dataset is None:
        dataset, source_label = _load_demo_dataset(dataset_source, default_dataset_id=problem_type.dataset_id)

    problem = problem_type(seed=seed, config=solver_config, dataset=dataset, runner=runner)
    if split not in problem.dataset:
        available = ", ".join(problem.dataset.keys())
        raise KeyError(f"dataset split {split!r} is unavailable; choose from: {available}")
    selected_split = problem.dataset[split]
    if not 0 <= index < len(selected_split):
        raise IndexError(f"dataset index must be in [0, {len(selected_split)}); got {index}")

    row = selected_split[index]
    design = problem.design_from_dataset_value(row["optimal_design"])
    row_conditions = {key: float(row[key]) for key in problem.conditions_keys}
    stored_objectives = {key: float(row[key]) for key in problem.objectives_keys}

    print(f"Dataset: {source_label}")
    print(f"Selected split={split!r}, index={index}")
    print("Conditions: " + ", ".join(f"{key}={value:.8g}" for key, value in row_conditions.items()))
    print("Stored objectives: " + ", ".join(f"{key}={value:.8g}" for key, value in stored_objectives.items()))
    provenance = row.get("design_provenance")
    if provenance:
        print(f"Design provenance: {provenance}")
    if row.get("design_is_exact") is False or row.get("objectives_evaluated_on_design") is False:
        print(
            "WARNING: this design was reconstructed lossily; its stored objectives "
            "belong to the source solver case and may not match a new simulation."
        )

    figure, _axes = problem.render(design, open_window=False)
    if render_output is not None:
        output_path = Path(render_output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved rendering to {output_path}")
    if open_window:
        plt.show()
    plt.close(figure)

    if not run_simulation:
        print("Simulation skipped. Pass --simulate with a solver config to run the external CFD solver.")
        return None

    print("Running frozen one-step simulation with the selected row conditions...")
    result = problem.simulate_verbose(design, config=row_conditions)
    print(
        "Simulated objectives: "
        + ", ".join(
            f"{key}={float(value):.8g}" for key, value in zip(problem.objectives_keys, result.objective_values, strict=True)
        )
    )
    print(
        "Simulation diagnostics: "
        f"volume_residual={result.volume_constraint_residual:.8g}, "
        f"power_residual={result.power_constraint_residual:.8g}, "
        f"elapsed_time={result.elapsed_time:.8g}s"
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a real MTO2D dataset row and optionally simulate it.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--dataset",
        dest="dataset_source",
        help=(
            "local DatasetDict directory or Hugging Face dataset ID; defaults to "
            "dataset_output/mto_2d_v0 when present, otherwise IDEALLab/mto_2d_v0"
        ),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-output", type=Path)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument(
        "--solver-config",
        type=Path,
        help=(
            "JSON object with case_template, backend, MPI, timeout, and related solver settings; "
            f"with --simulate, defaults to ${SOLVER_CONFIG_ENV_VAR} or the prepared local runtime"
        ),
    )
    return parser


def _cli(argv: list[str] | None = None) -> MTO2DSimulationResult | None:
    args = _parser().parse_args(argv)
    return main(
        dataset_source=args.dataset_source,
        split=args.split,
        index=args.index,
        seed=args.seed,
        solver_config=_read_solver_config(args.solver_config, auto_discover=args.simulate),
        run_simulation=args.simulate,
        render_output=args.render_output,
        open_window=args.show,
    )


if __name__ == "__main__":
    _cli()
