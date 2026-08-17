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
from engibench.problems.mto2d.model.runner import LEGACY_OPTIMIZATION_ITERATIONS
from engibench.problems.mto2d.model.runner import MTO2DRunner
from engibench.problems.mto2d.model.runner import POWER_BOUND_DECREMENT_PER_ITERATION
from engibench.problems.mto2d.model.runner import RunnerSettings
from engibench.problems.mto2d.model.runner import SolverRun
from engibench.utils.upcast import upcast

MIN_VOLUME_FRACTION = FIXED_CELL_COUNT / GAMMA_CELL_COUNT
"""Smallest feasible all-cell volume fraction when the design domain is solid."""

VOLUME_FRACTION_TOLERANCE = 0.01
"""Warning tolerance for the raw all-cell fluid-volume constraint."""

DEFAULT_CONTAINER_IMAGE = (
    "ghcr.io/arthurdrake1/engibench-mto2d@sha256:2887a5c8eaa3fba2d2738188757aeb66fe69d8ef7060698cb8512252aacaa131"
)
"""Published OCI image pinned by immutable manifest digest."""


@constraint(categories=THEORY, criticality=Criticality.Warning)
def fluid_volume_bound(design: npt.NDArray, volfrac: float) -> None:
    """Warn when the raw design exceeds the maximum all-cell fluid fraction."""
    density = np.asarray(design, dtype=np.float64)
    if density.shape != HALF_DESIGN_SHAPE:
        return
    actual_volume_fraction = float((density.sum() + FIXED_CELL_COUNT) / GAMMA_CELL_COUNT)
    assert actual_volume_fraction <= volfrac + VOLUME_FRACTION_TOLERANCE, (
        f"All-cell fluid fraction {actual_volume_fraction:.6g} exceeds maximum volfrac={volfrac:.6g} "
        f"by more than the {VOLUME_FRACTION_TOLERANCE:.2g} warning tolerance."
    )


@dataclass
class MTO2DSimulationResult(SimulationResult):
    """Objectives and constraint diagnostics from a frozen MTO2D evaluation."""

    volume_constraint_residual: float
    """Solver-reported ``mean(gamma) - volfrac`` over all cells."""

    power_constraint_residual: float
    """Relative power residual ``power_dissipation / max_power_dissipation - 1``."""

    elapsed_time: float
    """Cumulative solver wall time in seconds."""

    artifacts_path: str | None = None
    """Retained isolated run directory, when requested."""


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
            bounded(lower=47.7, upper=75.0).warning().category(IMPL),
        ] = 63.1
        """Dimensionless normalized power-dissipation bound.

        The retained solver divides physical dissipation by its exact
        ``D_normalization = 1.57572e-7``. The paper denotes the rounded
        reference scale ``J1 ≈ 1.58e-7``.

        The nominal dataset range is [47.7, 75]. The second sweep targeted
        [50, 75], but 17 of the 5,666 retained rows fall below 50.
        """

        volfrac: Annotated[
            float,
            bounded(lower=MIN_VOLUME_FRACTION, upper=1.0).category(THEORY),
            bounded(lower=0.25, upper=0.70).warning().category(IMPL),
        ] = 0.61
        """Maximum all-cell fluid volume fraction."""

    conditions = Conditions()
    design_constraints = (fluid_volume_bound,)
    design_space = spaces.Box(low=0.0, high=1.0, shape=HALF_DESIGN_SHAPE, dtype=np.float32)
    dataset_id = "IDEALLab/mto_2d_v0"
    container_id = DEFAULT_CONTAINER_IMAGE

    @dataclass
    class Config(Conditions):
        """Solver configuration not treated as a physical dataset condition.

        Every field here describes the *physics or schedule* of a run. Host
        concerns -- image, working directory, timeout, artifact retention -- are
        constructor arguments of :class:`~engibench.problems.mto2d.model.runner.MTO2DRunner`
        and are deliberately not part of the benchmark's configuration surface.
        """

        max_iter: Annotated[int, greater_than(0).category(IMPL)] = 200
        mode: str = "cold"
        optimization_schedule: str = "legacy"
        mpi_cores: Annotated[int, greater_than(0).category(IMPL)] = 1
        continuation_steps: int | None = None
        continuation_profile: str = "geometric"
        power_bound_start: float | None = None
        qu_start: float | None = None
        qu_final: Annotated[float, greater_than(0.0).category(IMPL)] = 0.01
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
        def valid_optimization_schedule(optimization_schedule: str) -> None:
            """Require a supported optimization schedule."""
            assert optimization_schedule in {
                "legacy",
                "strict",
            }, "Config.optimization_schedule must be 'legacy' or 'strict'"

        @constraint(categories=IMPL)
        @staticmethod
        def valid_continuation_profile(continuation_profile: str) -> None:
            """Require a profile implemented by the warm-ready solver."""
            assert continuation_profile in {
                "constant",
                "linear",
                "geometric",
            }, "Config.continuation_profile must be 'constant', 'linear', or 'geometric'"

        @constraint(categories=IMPL)
        @staticmethod
        def valid_power_bound_start(power_bound_start: float | None) -> None:
            """Require a positive optional initial power-dissipation bound."""
            assert power_bound_start is None or power_bound_start > 0.0, "Config.power_bound_start must be positive"

    def __init__(
        self,
        seed: int = 0,
        *,
        dataset: Any | None = None,
        runner: MTO2DRunner | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the MTO2D problem.

        Args:
            seed: The random seed for the problem.
            dataset: Optional in-memory dataset, used instead of the Hub dataset.
            runner: Optional solver runner. Defaults to running the pinned
                :attr:`container_id` image through `engibench.utils.container`.
            kwargs: Arguments are passed to :class:`MTO2D.Config`.
        """
        super().__init__(seed=seed)
        self.config = self.Config(**kwargs)
        self._raise_config_errors(self.config)
        self.conditions = upcast(self.config)
        self._runner = runner if runner is not None else MTO2DRunner(image=self.resolved_container_image())
        self.last_solver_run: SolverRun | None = None
        if dataset is not None:
            self._dataset = dataset

    @classmethod
    def resolved_container_image(cls) -> str | None:
        """Return the image the default runner uses, honouring `$ENGIBENCH_MTO2D_IMAGE`.

        This is the single place the environment is consulted, so that the
        override applies to the demo, the release verifier, and user code alike.
        """
        return os.environ.get("ENGIBENCH_MTO2D_IMAGE") or cls.container_id

    def reset(self, seed: int | None = None) -> None:
        """Reset the EngiBench random-number generator.

        Required by the shared problem suite, which checks that every problem
        declares `reset` on its own class rather than inheriting it.
        """
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
            artifacts_path=run.artifacts_path,
        )

    def optimize(
        self,
        starting_point: npt.NDArray,
        config: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.float32], list[OptiStep]]:
        """Optimize a topology with adjoint sensitivities and MMA.

        Solver history row ``k`` describes the pre-update design evaluated in
        iteration ``k``. The returned final gamma is the subsequent MMA update,
        so call :meth:`simulate` when exact objectives for that returned field
        are required.

        Per-iteration constraint residuals, elapsed times and the retained
        artifacts directory are available on :attr:`last_solver_run` afterwards;
        :meth:`active_power_bounds` reproduces the bound MMA actually saw.
        """
        density = self._coerce_design(starting_point)
        resolved = self._resolve_config(config)
        settings = self._runner_settings(resolved)
        MTO2DRunner.validate_settings(settings, "optimize")
        active_power_bounds = self.active_power_bounds(resolved)
        if resolved.optimization_schedule == "legacy" and resolved.max_iter < LEGACY_OPTIMIZATION_ITERATIONS:
            warnings.warn(
                f"A {resolved.max_iter}-step legacy optimization is an exact prefix of the "
                "published 200-step schedule, not a converged source reproduction.",
                UserWarning,
                stacklevel=2,
            )
        if active_power_bounds[-1] > resolved.max_power_dissipation:
            warnings.warn(
                "Optimization stops before the legacy power-bound continuation reaches "
                f"max_power_dissipation={resolved.max_power_dissipation:.8g}; its final active bound is "
                f"{active_power_bounds[-1]:.8g}. Use more iterations, set power_bound_start to the final bound, "
                "or treat this as a runtime smoke test.",
                UserWarning,
                stacklevel=2,
            )
        run = self._runner.run(density, settings, kind="optimize")
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
        return np.asarray(run.final_design, dtype=np.float32), history

    def active_power_bounds(self, config: Config | dict[str, Any] | None = None) -> npt.NDArray[np.float64]:
        """Return the power bound actually supplied to MMA at every iteration.

        The legacy schedule walks the bound down from ``power_bound_start``
        towards ``max_power_dissipation``, so a short run never reaches the
        requested bound. Combine with `last_solver_run.power_dissipation` to
        recover per-iteration constraint residuals after :meth:`optimize`.
        """
        resolved = config if isinstance(config, self.Config) else self._resolve_config(config)
        return self._active_power_bounds(resolved)

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
        """Sample a native design from the published dataset."""
        split = self.dataset[dataset_split]
        index = int(self.np_random.integers(0, len(split)))
        return self.design_from_dataset_value(split[index][design_key]), index

    @staticmethod
    def design_from_dataset_value(value: Any) -> npt.NDArray[np.float32]:
        """Convert a native or flattened dataset value to the design space."""
        design = np.asarray(value, dtype=np.float32)
        if design.shape != HALF_DESIGN_SHAPE and design.size == int(np.prod(HALF_DESIGN_SHAPE)):
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
    def design_volume_residual(design: npt.NDArray, volfrac: float) -> float:
        """Estimate the raw all-cell volume residual before filtering/projection."""
        density = np.asarray(design, dtype=np.float64)
        if density.shape != HALF_DESIGN_SHAPE:
            raise ValueError(f"design must have shape {HALF_DESIGN_SHAPE}")
        return float((density.sum() + FIXED_CELL_COUNT) / GAMMA_CELL_COUNT - volfrac)

    @staticmethod
    def uniform_starting_design(volfrac: float) -> npt.NDArray[np.float32]:
        """Create a uniform design whose all-cell mean equals the requested volume.

        The correction accounts for the 6,400 fixed-fluid cells outside the
        80,000-cell design domain.
        """
        design_fraction = (GAMMA_CELL_COUNT * volfrac - FIXED_CELL_COUNT) / int(np.prod(HALF_DESIGN_SHAPE))
        if not 0.0 <= design_fraction <= 1.0:
            raise ValueError("volfrac is incompatible with the fixed-fluid region")
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
        values["volume_fraction"] = values.pop("volfrac")
        fields = {field.name for field in dataclasses.fields(RunnerSettings)}
        values = {key: value for key, value in values.items() if key in fields}
        if max_iter is not None:
            values["max_iter"] = max_iter
        return RunnerSettings(**values)


def main(problem_type: type[MTO2D] = MTO2D, *, open_window: bool = False) -> None:
    """Sample a dataset design, render it, and re-evaluate it in the published container."""
    problem = problem_type(seed=0)
    design, index = problem.random_design()
    row = problem.dataset["train"][index]
    conditions = {key: float(row[key]) for key in problem.conditions_keys}
    stored_objectives = {key: float(row[key]) for key in problem.objectives_keys}

    print(f"Sampled train row {index}.")
    print("Conditions: " + ", ".join(f"{key}={value:.8g}" for key, value in conditions.items()))
    print("Stored objectives: " + ", ".join(f"{key}={value:.8g}" for key, value in stored_objectives.items()))
    problem.render(design, open_window=open_window)

    result = problem.simulate_verbose(design, config=conditions)
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


if __name__ == "__main__":
    main(open_window=True)
