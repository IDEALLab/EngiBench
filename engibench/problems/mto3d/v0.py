"""Three-dimensional multiphysics topology optimization problem."""

import dataclasses
from dataclasses import dataclass
from typing import Annotated, Any

from gymnasium import spaces
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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
from engibench.problems.mto3d.model.design_io import DESIGN_CELL_COUNT
from engibench.problems.mto3d.model.design_io import FLUID_CELL_COUNT
from engibench.problems.mto3d.model.design_io import HALF_DESIGN_SHAPE
from engibench.problems.mto3d.model.design_io import half_to_full
from engibench.problems.mto3d.model.design_io import XH_CELL_COUNT
from engibench.problems.mto3d.model.runner import DECOMPOSITIONS
from engibench.problems.mto3d.model.runner import DEFAULT_FROZEN_NS_MAX_ITER
from engibench.problems.mto3d.model.runner import DEFAULT_MPI_CORES
from engibench.problems.mto3d.model.runner import MTO3DRunner
from engibench.problems.mto3d.model.runner import RunnerSettings
from engibench.problems.mto3d.model.runner import SolverRun
from engibench.utils.upcast import upcast

MIN_VOLUME_FRACTION = FLUID_CELL_COUNT / XH_CELL_COUNT
"""Smallest all-cell fluid fraction: every design cell solid, the 24,000 fixed fluid cells fluid."""
MAX_VOLUME_FRACTION = (FLUID_CELL_COUNT + DESIGN_CELL_COUNT) / XH_CELL_COUNT
"""Largest all-cell fluid fraction: every design cell fluid, the 40,000 fixed solid cells solid."""

VOLUME_FRACTION_TOLERANCE = 0.01
"""Warning tolerance for the raw all-cell fluid-volume constraint."""

DEFAULT_CONTAINER_IMAGE = (
    "ghcr.io/ideallab/engibench-mto3d@sha256:e1df3b29d462c3a4ba81bd1355fccc26574141842a3a78bc38b4c2cbec9f50e0"
)
"""Published OCI image pinned by immutable manifest digest."""

RENDER_VIEW = (22.0, -72.0)
"""Elevation and azimuth (degrees) of the 3D render; the heated wall faces the viewer."""

RENDER_FLUID_THRESHOLD = 0.5
"""Cells with ``xh`` above this value are drawn as fluid."""

_VOXEL_FACES = {  # outward normal (axis, sign) -> corners of that face of the unit cube
    (0, -1): ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),
    (0, 1): ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)),
    (1, -1): ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),
    (1, 1): ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)),
    (2, -1): ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)),
    (2, 1): ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)),
}
_FACE_SHADE = np.array([0.78, 1.0, 0.9])
"""Brightness of side (x), front (depth) and top (z) faces in the 3D render."""


def _render_volume(design: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
    """Fluid cells of the mirrored design as ``(x, depth, z)``: heated wall at depth 0, inlet at the top."""
    return (half_to_full(design) > RENDER_FLUID_THRESHOLD).transpose(2, 0, 1)[:, ::-1, ::-1]


def _camera_facing_voxel_faces(
    filled: npt.NDArray[np.bool_], camera: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int_]]:
    """Exposed faces of ``filled`` cells that face ``camera``: quads, their mean depth, and their normal axis.

    A face is exposed when the cell across it is empty or outside the grid. Faces turned away from the
    camera lie behind a closed surface, so they are skipped.
    """
    padded = np.pad(filled, 1)
    quads, depth, axes = [], [], []
    for (axis, sign), corners in _VOXEL_FACES.items():
        if sign * camera[axis] <= 0:
            continue
        neighbour = np.roll(padded, -sign, axis=axis)[1:-1, 1:-1, 1:-1]
        cells = np.argwhere(filled & ~neighbour).astype(np.float32)
        face = cells[:, None, :] + np.asarray(corners, dtype=np.float32)[None]
        quads.append(face)
        depth.append(face[:, :, 1].mean(axis=1))
        axes.append(np.full(len(face), axis))
    return np.concatenate(quads), np.concatenate(depth), np.concatenate(axes)


@constraint(categories=THEORY, criticality=Criticality.Warning)
def fluid_volume_bound(design: npt.NDArray, volfrac: float) -> None:
    """Warn when the raw design exceeds the maximum all-cell fluid fraction (equal cell volumes)."""
    density = np.asarray(design, dtype=np.float64)
    if density.shape != HALF_DESIGN_SHAPE:
        return
    actual = float((density.sum() + FLUID_CELL_COUNT) / XH_CELL_COUNT)
    assert actual <= volfrac + VOLUME_FRACTION_TOLERANCE, (
        f"All-cell fluid fraction {actual:.6g} exceeds maximum volfrac={volfrac:.6g} "
        f"by more than the {VOLUME_FRACTION_TOLERANCE:.2g} warning tolerance."
    )


@dataclass
class MTO3DSimulationResult(SimulationResult):
    """Objectives and diagnostics from a frozen MTO3D evaluation."""

    volume_constraint_residual: float
    """Solver-reported ``mean(xh) - volfrac`` over all 264,000 equal-volume cells."""

    power_constraint_residual: float
    """Relative power residual ``power_dissipation / max_power_dissipation - 1``."""

    elapsed_time: float
    """Solver wall time in seconds."""

    ns_passes: int
    """SIMPLE passes the flow solve from rest needed to reach its residual tolerance (1e-6)."""

    artifacts_path: str | None = None
    """Retained isolated run directory, when requested."""


@dataclass(kw_only=True)
class MTO3DOptiStep(OptiStep):
    """One MTO3D optimization step with its constraint diagnostics."""

    volume_constraint_residual: float
    """Solver-reported ``mean(xh) - volfrac`` over all cells for the evaluated design."""

    power_constraint_residual: float
    """Relative residual against ``max_power_dissipation``."""

    elapsed_time: float
    """Cumulative solver wall time in seconds."""


class MTO3D(Problem[npt.NDArray]):
    """OpenFOAM-based 3D thermofluid topology optimization of a liquid-cooled heat sink.

    A design is the non-redundant ``(20, 100, 100)`` fluid-density half-domain in ``(y, z, x)``
    order. ``xh=0`` is solid and ``xh=1`` is fluid. The other half is implied by the symmetry plane
    at the last x index and is created only for rendering.

    The built-in optimizer minimizes mean temperature on the heated wall with MMA, treating fluid
    volume and power dissipation as constraints. Power dissipation is also exposed as a second
    EngiBench objective for Pareto analysis.
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
            bounded(lower=-0.170, upper=-0.100).warning().category(IMPL),
        ] = -0.121
        """Signed inlet z-velocity in m/s (flow along -z); the dataset range is [-0.170, -0.100].

        The published cases effectively used this value rounded to 3 significant figures; see
        ``Config.inlet_significant_figures``.
        """

        max_power_dissipation: Annotated[
            float,
            greater_than(0.0).category(THEORY),
            bounded(lower=5.0, upper=25.0).warning().category(IMPL),
        ] = 21.6
        """Bound on the normalized fluid power dissipation (physical value / 1e-9); dataset range [5, 25]."""

        volfrac: Annotated[
            float,
            bounded(lower=MIN_VOLUME_FRACTION, upper=MAX_VOLUME_FRACTION).category(THEORY),
            bounded(lower=0.25, upper=0.55).warning().category(IMPL),
        ] = 0.49
        """Maximum all-cell fluid volume fraction; dataset range [0.25, 0.55]."""

    conditions = Conditions()
    design_constraints = (fluid_volume_bound,)
    design_space = spaces.Box(low=0.0, high=1.0, shape=HALF_DESIGN_SHAPE, dtype=np.float32)
    dataset_id = "IDEALLab/mto_3d_v0"
    container_id = DEFAULT_CONTAINER_IMAGE

    @dataclass
    class Config(Conditions):
        """Solver configuration not treated as a physical dataset condition.

        Image selection, working directory, timeout, and artifact retention belong to
        :class:`~engibench.problems.mto3d.model.runner.MTO3DRunner`.
        """

        max_iter: Annotated[int, greater_than(0).category(IMPL)] = 3000
        """Iteration cap for :meth:`MTO3D.optimize`; the legacy continuation stops by itself
        (at most 1,920 iterations)."""
        mode: str = "cold"
        """``cold``: legacy 20-level continuation; ``warm``: final physics from the first iteration."""
        mpi_cores: Annotated[int, greater_than(0).category(IMPL)] = DEFAULT_MPI_CORES
        """MPI ranks; 16 (the decomposition of every published case) is the parity-tested value."""
        movement_limit: Annotated[float, bounded(lower=0.0, upper=1.0).category(IMPL)] = 0.4
        frozen_ns_max_iter: Annotated[int, greater_than(0).category(IMPL)] = DEFAULT_FROZEN_NS_MAX_ITER
        """SIMPLE-pass cap for :meth:`MTO3D.simulate`, which must converge the flow to 1e-6. A design
        whose flow does not converge raises after roughly this many passes."""
        inlet_significant_figures: int | None = 3
        """Round ``inlet_velocity`` to this many significant figures (as the dataset campaigns did);
        ``None`` uses the nominal value."""

        @constraint(categories=IMPL)
        @staticmethod
        def valid_mode(mode: str) -> None:
            """Require a supported continuation mode."""
            assert mode in {"cold", "warm"}, "Config.mode must be 'cold' or 'warm'"

        @constraint(categories=IMPL)
        @staticmethod
        def valid_mpi_cores(mpi_cores: int) -> None:
            """Require a rank count with a defined 3D decomposition."""
            assert mpi_cores in DECOMPOSITIONS, f"Config.mpi_cores must be one of {sorted(DECOMPOSITIONS)}"

        @constraint(categories=IMPL)
        @staticmethod
        def valid_inlet_significant_figures(inlet_significant_figures: int | None) -> None:
            """Require a positive number of significant figures or None."""
            assert inlet_significant_figures is None or inlet_significant_figures > 0, (
                "Config.inlet_significant_figures must be positive or None"
            )

    config: Config

    def __init__(
        self,
        seed: int = 0,
        *,
        dataset: Any | None = None,
        runner: MTO3DRunner | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the MTO3D problem.

        Args:
            seed: The random seed for the problem.
            dataset: Optional in-memory dataset, used instead of the Hub dataset.
            runner: Optional solver runner. Defaults to running the pinned :attr:`container_id`
                image through `engibench.utils.container`.
            kwargs: Arguments are passed to :class:`MTO3D.Config`.
        """
        super().__init__(seed=seed)
        self.config = self.Config(**kwargs)
        self._raise_config_errors(self.config)
        self.conditions = upcast(self.config)
        self._runner = runner if runner is not None else MTO3DRunner(image=self.container_id)
        self.last_solver_run: SolverRun | None = None
        if dataset is not None:
            self._dataset = dataset

    def reset(self, seed: int | None = None) -> None:
        """Reset the EngiBench random-number generator."""
        super().reset(seed)

    def simulate_verbose(self, design: npt.NDArray, config: dict[str, Any] | None = None) -> MTO3DSimulationResult:
        """Evaluate a fixed design with final physics; the design is never updated.

        The flow is solved from rest until both initial residuals reach 1e-6. The solver then runs
        100 iterations with the design fixed, re-assembling and re-solving the energy equation each
        time until its linearUpwind correction has converged, and reports the final objectives. A
        non-converged flow or energy solve raises an error.
        """
        density = self._coerce_design(design)
        resolved = self._resolve_config(config)
        run = self._runner.run(density, self._runner_settings(resolved), kind="simulate")
        self.last_solver_run = run
        status = run.frozen_status or {}
        mean_temperature = float(status.get("mean_temperature", run.mean_temperature[-1]))
        power_dissipation = float(status.get("power_dissipation", run.power_dissipation[-1]))
        return MTO3DSimulationResult(
            objective_values=np.array([mean_temperature, power_dissipation], dtype=np.float64),
            volume_constraint_residual=float(status.get("volume_residual", run.volume_residual[-1])),
            power_constraint_residual=power_dissipation / resolved.max_power_dissipation - 1.0,
            elapsed_time=float(status.get("elapsed_clock_time", run.elapsed_time[-1])),
            ns_passes=int(status.get("ns_passes_initial", status.get("ns_passes", 0))),
            artifacts_path=run.artifacts_path,
        )

    def optimize(
        self, starting_point: npt.NDArray, config: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float32], list[MTO3DOptiStep]]:
        """Optimize a topology with adjoint sensitivities and MMA.

        Solver history row ``k`` describes the design evaluated in iteration ``k``. The returned
        final design is the subsequent MMA update, so call :meth:`simulate` when exact objectives
        for the returned design are required. History objectives come from the optimizer's single
        energy solve per iteration, with the temperature carried over from earlier iterations, and
        can differ slightly from :meth:`simulate`. The legacy continuation may finish before
        ``max_iter`` (adaptive stopping); the history then has fewer entries. ``mode="warm"``
        optimizes with final physics from the supplied design. It is not an exact restart of a
        finished run, because a stored design carries no optimizer state.
        """
        density = self._coerce_design(starting_point)
        resolved = self._resolve_config(config)
        run = self._runner.run(density, self._runner_settings(resolved), kind="optimize")
        self.last_solver_run = run
        history = [
            MTO3DOptiStep(
                obj_values=np.array([mean_temperature, power_dissipation], dtype=np.float64),
                step=step,
                volume_constraint_residual=float(volume_residual),
                power_constraint_residual=float(power_dissipation / resolved.max_power_dissipation - 1.0),
                elapsed_time=float(elapsed_time),
            )
            for step, (mean_temperature, power_dissipation, volume_residual, elapsed_time) in enumerate(
                zip(run.mean_temperature, run.power_dissipation, run.volume_residual, run.elapsed_time, strict=True),
                start=1,
            )
        ]
        return np.asarray(run.final_design, dtype=np.float32), history

    def random_design(
        self, dataset_split: str = "train", design_key: str = "optimal_design"
    ) -> tuple[npt.NDArray[np.float32], int]:
        """Sample a native design from the dataset."""
        split = self.dataset[dataset_split]
        index = int(self.np_random.integers(0, len(split)))
        return self.design_from_dataset_value(split[index][design_key]), index

    @staticmethod
    def design_from_dataset_value(value: Any) -> npt.NDArray[np.float32]:
        """Convert a native or flattened (C-order) dataset value to the design space."""
        design = np.asarray(value, dtype=np.float32)
        if design.shape != HALF_DESIGN_SHAPE and design.size == DESIGN_CELL_COUNT:
            design = design.reshape(HALF_DESIGN_SHAPE)
        return MTO3D._coerce_design(design)

    def render(self, design: npt.NDArray, *, open_window: bool = False) -> tuple[Figure, Any]:
        """Render the mirrored ``(20, 100, 200)`` design in 3D, its fluid cells colored by depth into the page.

        The heated wall faces the viewer: depth 0 is the design layer next to it and depth 20 the
        insulated face. Only the exposed, camera-facing faces of fluid cells (``xh > 0.5``) are drawn.
        """
        filled = _render_volume(self._coerce_design(design))
        elevation, azimuth = RENDER_VIEW
        el, az = np.radians(elevation), np.radians(azimuth)
        camera = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        quads, depth, axis = _camera_facing_voxel_faces(filled, camera)
        norm = Normalize(0.0, float(filled.shape[1]))
        colors = plt.get_cmap("viridis")(norm(depth))
        colors[:, :3] *= _FACE_SHADE[axis][:, None]
        fig = plt.figure(figsize=(8.0, 4.8))
        ax: Any = fig.add_axes((0.0, 0.17, 0.93, 0.80), projection="3d", computed_zorder=False)
        ax.add_collection3d(Poly3DCollection(quads, facecolors=colors, edgecolors=colors, linewidths=0.3))
        nx, ny, nz = filled.shape
        ax.plot([nx / 2, nx / 2], [0, 0], [0, nz], color="black", linewidth=0.7, linestyle="--")
        ax.set(xlim=(0, nx), ylim=(0, ny), zlim=(0, nz))
        ax.set_box_aspect((nx, ny, nz), zoom=1.12)
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_xticks([0, 50, 100, 150, 200])
        ax.set_yticks([])
        ax.set_zticks([0, 50, 100])
        ax.set_xlabel("x cell", labelpad=8)
        ax.set_zlabel("z cell (inlet at top)", labelpad=6)
        ax.set_title("MTO3D fluid cells, colored by depth", y=1.0)
        fig.colorbar(
            ScalarMappable(norm=norm, cmap="viridis"),
            cax=fig.add_axes((0.27, 0.115, 0.42, 0.03)),
            orientation="horizontal",
            ticks=[0, 5, 10, 15, 20],
            label="depth into page (y cells, 0 = heated face)",
        )
        if open_window:
            plt.show()
        return fig, ax

    @staticmethod
    def design_volume_residual(design: npt.NDArray, volfrac: float) -> float:
        """Raw all-cell volume residual before filtering/projection (all cells have equal volume)."""
        density = np.asarray(design, dtype=np.float64)
        if density.shape != HALF_DESIGN_SHAPE:
            raise ValueError(f"design must have shape {HALF_DESIGN_SHAPE}")
        return float((density.sum() + FLUID_CELL_COUNT) / XH_CELL_COUNT - volfrac)

    @staticmethod
    def uniform_starting_design(volfrac: float) -> npt.NDArray[np.float32]:
        """Create the historical cold start with every design cell at ``volfrac``.

        With ``mode="cold"`` this reproduces the published initialization exactly: the runner then
        supplies no ``0/xh`` and the solver starts from uniform ``voluse``.
        """
        if not 0.0 <= volfrac <= 1.0:
            raise ValueError("volfrac must lie in [0, 1]")
        return np.full(HALF_DESIGN_SHAPE, volfrac, dtype=np.float32)

    @staticmethod
    def _raise_config_errors(config: Config) -> None:
        errors = check_field_constraints(config).by_criticality(Criticality.Error)
        if errors:
            raise ValueError(str(errors))

    def _resolve_config(self, overrides: dict[str, Any] | None) -> Config:
        values = dataclasses.asdict(self.config)
        values.update(overrides or {})
        resolved = self.Config(**values)
        self._raise_config_errors(resolved)
        return resolved

    @staticmethod
    def _coerce_design(design: npt.NDArray) -> npt.NDArray[np.float32]:
        density = np.asarray(design, dtype=np.float32)
        if density.shape != HALF_DESIGN_SHAPE:
            raise ValueError(f"MTO3D design must have shape {HALF_DESIGN_SHAPE}; got {density.shape}")
        if not np.all(np.isfinite(density)):
            raise ValueError("MTO3D design must contain only finite values")
        if np.any((density < 0.0) | (density > 1.0)):
            raise ValueError("MTO3D design values must lie in [0, 1]")
        return np.ascontiguousarray(density)

    @staticmethod
    def _runner_settings(config: Config) -> RunnerSettings:
        return RunnerSettings(**dataclasses.asdict(config))


def main(problem_type: type[MTO3D] = MTO3D, *, open_window: bool = False) -> None:
    """Sample a dataset design, render it, and re-evaluate it with that row's own conditions."""
    problem = problem_type(seed=0)
    design, index = problem.random_design()
    row = problem.dataset["train"][index]
    conditions = {key: float(row[key]) for key in problem.conditions_keys}
    stored = {key: float(row[key]) for key in problem.objectives_keys}
    print(f"Sampled train row {index}.")
    print("Conditions: " + ", ".join(f"{k}={v:.8g}" for k, v in conditions.items()))
    print("Stored objectives: " + ", ".join(f"{k}={v:.8g}" for k, v in stored.items()))
    problem.render(design, open_window=open_window)
    result = problem.simulate_verbose(design, config=conditions)
    print(
        "Simulated objectives: "
        + ", ".join(f"{k}={float(v):.8g}" for k, v in zip(problem.objectives_keys, result.objective_values, strict=True))
    )
    print(
        "Simulation diagnostics: "
        f"volume_residual={result.volume_constraint_residual:.8g}, "
        f"power_residual={result.power_constraint_residual:.8g}, "
        f"elapsed_time={result.elapsed_time:.8g}s, ns_passes={result.ns_passes}"
    )


if __name__ == "__main__":
    main(open_window=True)
