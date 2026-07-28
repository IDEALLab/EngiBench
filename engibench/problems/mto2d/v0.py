"""Two-dimensional multiphysics topology optimization problem."""

import dataclasses
from dataclasses import dataclass
import os
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
from engibench.problems.mto2d.model.runner import RunnerSettings
from engibench.problems.mto2d.model.runner import SolverRun
from engibench.utils.upcast import upcast

J1 = 1.58e-7
"""Published reference scale for normalized fluid power dissipation."""

MIN_VOLUME_FRACTION = FIXED_CELL_COUNT / GAMMA_CELL_COUNT
"""Smallest feasible all-cell volume fraction when the design domain is solid."""


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
    """Detailed result used by :meth:`MTO2D.optimize_verbose`."""

    design: npt.NDArray[np.float32]
    history: list[OptiStep]
    volume_constraint_residuals: npt.NDArray[np.float64]
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
        """Power-dissipation bound as a multiple of ``J1``."""

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
        """Evaluate a fixed topology for one final-physics solver iteration."""
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
        """Optimize and also return volume/time histories and artifact location.

        Solver history row ``k`` describes the pre-update design evaluated in
        iteration ``k``. The returned final gamma is the subsequent MMA update,
        so call :meth:`simulate` when exact objectives for that returned field
        are required.
        """
        density = self._coerce_design(starting_point)
        resolved = self._resolve_config(config)
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
            power_constraint_residuals=run.power_dissipation / resolved.max_power_dissipation - 1.0,
            elapsed_times=run.elapsed_time.copy(),
            artifacts_path=run.artifacts_path,
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
        design = np.asarray(split[index][design_key], dtype=np.float32)
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
        return self._coerce_design(design), index

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


def main(problem_type: type[MTO2D] = MTO2D, *, open_window: bool = False) -> None:
    """Render one condition-aware starting design without launching the solver."""
    problem = problem_type(seed=0)
    design = problem.uniform_starting_design(problem.conditions.volume_fraction)
    problem.render(design, open_window=open_window)


if __name__ == "__main__":
    main(open_window=True)
