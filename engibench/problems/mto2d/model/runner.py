"""Isolated OpenFOAM runner for the MTO2D problem."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
from typing import Literal

import numpy as np
import numpy.typing as npt

from engibench.problems.mto2d.model.design_io import DESIGN_CELL_COUNT
from engibench.problems.mto2d.model.design_io import FIXED_CELL_COUNT
from engibench.problems.mto2d.model.design_io import GAMMA_CELL_COUNT
from engibench.problems.mto2d.model.design_io import parse_internal_field
from engibench.problems.mto2d.model.design_io import read_half_design
from engibench.problems.mto2d.model.design_io import write_half_design
from engibench.utils import container

RunKind = Literal["simulate", "optimize"]
OptimizationMode = Literal["cold", "warm"]
OptimizationSchedule = Literal["legacy", "strict"]
Backend = Literal["container", "command"]
CONTINUATION_PROFILES = frozenset({"constant", "linear", "geometric"})
"""Profiles implemented by the retained warm-ready solver."""

POWER_BOUND_DECREMENT_PER_ITERATION = 0.2
"""Hard-coded legacy decrease from ``D0`` toward ``D1`` each iteration."""

LEGACY_OPTIMIZATION_ITERATIONS = 200
"""Iteration count used by the published cold-start optimization."""

LEGACY_QU_START = 0.005
LEGACY_QU_FINAL = 0.01
LEGACY_ALPHA_MAX_START = 2500.0
LEGACY_ALPHA_MAX_FINAL = 5_025_226.639126618
LEGACY_HEAVISIDE_START = 0.1
LEGACY_HEAVISIDE_FINAL = 59.8
"""Exact endpoints produced by the legacy 200-step source schedule."""

SCHEDULE_RUNTIME_VERSION = "2"
SCHEDULE_RUNTIME_MARKER = ".engibench-mto2d-runtime-version"
"""Prepared-runtime capability marker required for optimization schedules."""

FROZEN_GAMMA_ABSOLUTE_TOLERANCE = 1e-7
"""Absolute tolerance used to verify that frozen evaluation preserves gamma."""

RUN_LOG_TAIL_BYTES = 16 * 1024
RUN_LOG_TAIL_LINES = 20
"""Limits for solver output included in failure messages."""

_EXPECTED_BLOCK_MESH_LAYOUT = (
    (None, (160, 400, 1)),
    ("zone_test", (40, 400, 1)),
    ("zone_fluid", (40, 80, 1)),
    ("zone_fluid", (40, 80, 1)),
)
"""Ordered mesh blocks that place design cells before the fixed-fluid tail."""

_EXPECTED_TRANSPORT_SCALARS = {
    "solid_area": 0.0,
    "fluid_area": 1.0,
    "test_PD": 0.0,
    "D_normalization": 1.57572e-7,
}
"""Template switches and normalization required by the MTO2D API semantics."""

HISTORY_FILES = {
    "mean_temperature": "meanT.txt",
    "power_dissipation": "Disspower.txt",
    "volume_residual": "Voluse.txt",
    "elapsed_time": "Time.txt",
}

DIAGNOSTIC_HISTORY_FILES = ("aMax.txt", "qu.txt", "HEAV.txt")
"""Continuation diagnostics written by ``costfunction.H`` but not parsed here."""


@dataclass(frozen=True)
class RunnerSettings:
    """All values needed to prepare and execute one isolated solver case."""

    case_template: str | None
    inlet_velocity: float
    max_power_dissipation: float
    volume_fraction: float
    max_iter: int = 200
    mode: OptimizationMode = "cold"
    optimization_schedule: OptimizationSchedule = "legacy"
    mpi_cores: int = 1
    backend: Backend = "container"
    container_image: str | None = None
    driver_command: tuple[str, ...] = ()
    solver_executable: str = "../src_TF/EXEC"
    timeout: float | None = None
    work_dir: str | None = None
    retain_artifacts: bool = False
    retain_on_failure: bool = True
    continuation_steps: int | None = None
    power_bound_start: float | None = None
    qu_start: float | None = None
    qu_final: float = 0.01
    alpha_max_start: float | None = None
    alpha_max_final: float = 5.0252e6
    heaviside_start: float | None = None
    heaviside_final: float = 59.8
    continuation_profile: str = "geometric"
    movement_limit: float = 0.4


@dataclass(frozen=True)
class SolverRun:
    """Parsed output from one solver invocation."""

    final_design: npt.NDArray[np.float32]
    mean_temperature: npt.NDArray[np.float64]
    power_dissipation: npt.NDArray[np.float64]
    volume_residual: npt.NDArray[np.float64]
    elapsed_time: npt.NDArray[np.float64]
    artifacts_path: str | None


@dataclass(frozen=True)
class ContinuationSettings:
    """Continuation dictionary values for an optimization or simulation."""

    optimization_schedule: OptimizationSchedule
    steps: int
    profile: str
    qu: tuple[float, float]
    alpha_max: tuple[float, float]
    heaviside: tuple[float, float]


@dataclass(frozen=True)
class PreparedContinuation:
    """Resolved solver controls for one run."""

    iteration_count: int
    movement_limit: float
    continuation: ContinuationSettings


class SolverRunError(RuntimeError):
    """MTO2D solver failure with an optional retained run directory."""

    def __init__(self, message: str, artifacts_path: Path | None = None) -> None:
        suffix = f"\nSolver artifacts retained at: {artifacts_path}" if artifacts_path is not None else ""
        super().__init__(message + suffix)
        self.message = message
        self.artifacts_path = artifacts_path


def _run_log_diagnostics(path: Path) -> str:
    """Return the location and a bounded tail of a solver log."""
    if not path.is_file():
        return ""
    label = f"\nSolver log: {path}"
    try:
        with path.open("rb") as log:
            log.seek(0, os.SEEK_END)
            size = log.tell()
            log.seek(max(0, size - RUN_LOG_TAIL_BYTES))
            text = log.read().decode("utf-8", errors="replace")
    except OSError as error:
        return f"{label}\nUnable to read solver log: {error}"

    lines = text.splitlines()
    tail = lines[-RUN_LOG_TAIL_LINES:]
    if not tail:
        return f"{label}\nSolver log is empty."
    return f"{label}\nLast {len(tail)} solver log lines:\n" + "\n".join(tail)


def _mask_foam_comments(text: str) -> str:
    """Replace OpenFOAM comments with whitespace while preserving indices."""

    def mask(match: re.Match[str]) -> str:
        return re.sub(r"[^\r\n]", " ", match.group(0))

    return re.sub(r"/\*.*?\*/|//[^\r\n]*", mask, text, flags=re.DOTALL)


class MTO2DRunner:
    """Prepare and run a copied MTO2D OpenFOAM case."""

    def run(
        self,
        design: npt.NDArray[np.float32],
        settings: RunnerSettings,
        *,
        kind: RunKind,
    ) -> SolverRun:
        """Execute a frozen simulation or an MMA optimization."""
        self._validate_settings(settings, kind)
        configured_template = settings.case_template or os.environ.get("ENGIBENCH_MTO2D_CASE_TEMPLATE")
        case_template = self._resolve_case_template(configured_template) if configured_template else None
        if case_template is None and settings.backend != "container":
            self._resolve_case_template(None)
        run_root = Path(
            tempfile.mkdtemp(
                prefix="engibench-mto2d-",
                dir=settings.work_dir,
            )
        ).resolve()
        case_dir = run_root / "case"
        succeeded = False
        keep = settings.retain_artifacts

        try:
            if case_template is None:
                self._export_case_template(run_root, case_dir, settings)
                self._validate_case_template(case_dir)
            else:
                shutil.copytree(case_template, case_dir)
            if kind == "simulate":
                self._validate_runtime_marker(case_dir, "frozen simulation")
            if kind == "optimize":
                self._validate_schedule_runtime(case_dir, settings.optimization_schedule)
            self._prepare_case(case_dir, design, settings, kind)
            self._execute(case_dir, settings, kind)
            histories = self._read_histories(
                case_dir / "app", expected_steps=1 if kind == "simulate" else settings.max_iter
            )
            if kind == "simulate":
                self._validate_frozen_output(case_dir / "app", design)
            final_design = (
                np.asarray(design, dtype=np.float32).copy()
                if kind == "simulate"
                else read_half_design(self._latest_gamma(case_dir / "app"))
            )
            succeeded = True
            return SolverRun(
                final_design=final_design,
                artifacts_path=str(run_root) if keep else None,
                **histories,
            )
        except Exception as exc:
            keep = settings.retain_on_failure
            retained = run_root if keep else None
            message = exc.message if isinstance(exc, SolverRunError) else str(exc)
            artifacts_path = exc.artifacts_path if isinstance(exc, SolverRunError) else None
            raise SolverRunError(
                message + _run_log_diagnostics(case_dir / "run.log"),
                artifacts_path or retained,
            ) from exc
        finally:
            if (succeeded and not settings.retain_artifacts) or (not succeeded and not keep):
                shutil.rmtree(run_root, ignore_errors=True)

    @staticmethod
    def _validate_settings(settings: RunnerSettings, kind: RunKind) -> None:
        MTO2DRunner._validate_common_settings(settings)
        MTO2DRunner._validate_backend_settings(settings)
        if kind == "optimize":
            MTO2DRunner._validate_optimization_settings(settings)

    @staticmethod
    def _validate_common_settings(settings: RunnerSettings) -> None:
        if settings.max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if settings.mpi_cores < 1:
            raise ValueError("mpi_cores must be at least 1")
        if settings.mode not in {"cold", "warm"}:
            raise ValueError("mode must be 'cold' or 'warm'")
        if settings.optimization_schedule not in {"legacy", "strict"}:
            raise ValueError("optimization_schedule must be 'legacy' or 'strict'")
        if settings.continuation_profile not in CONTINUATION_PROFILES:
            allowed = ", ".join(sorted(CONTINUATION_PROFILES))
            raise ValueError(f"continuation_profile must be one of: {allowed}")
        if settings.timeout is not None and settings.timeout <= 0:
            raise ValueError("timeout must be positive")
        if settings.power_bound_start is not None and settings.power_bound_start <= 0:
            raise ValueError("power_bound_start must be positive")

    @staticmethod
    def _validate_backend_settings(settings: RunnerSettings) -> None:
        if settings.backend not in {"container", "command"}:
            raise ValueError("backend must be 'container' or 'command'")
        if settings.backend == "container" and not settings.container_image:
            raise ValueError("container_image is required for the container backend")
        if settings.backend == "command" and not settings.driver_command:
            raise ValueError("driver_command is required for the command backend")
        if settings.backend == "container" and settings.timeout is not None:
            raise ValueError(
                "The EngiBench container abstraction cannot enforce timeouts. "
                "Use backend='command' with a Docker, Podman, or Apptainer command when a timeout is required."
            )

    @staticmethod
    def _validate_optimization_settings(settings: RunnerSettings) -> None:
        for name, value in (
            ("qu_start", settings.qu_start),
            ("alpha_max_start", settings.alpha_max_start),
            ("heaviside_start", settings.heaviside_start),
        ):
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if settings.optimization_schedule == "legacy":
            if settings.mode != "cold":
                raise ValueError(
                    "optimization_schedule='legacy' is the cold source-reproduction schedule; "
                    "warm repair must pass optimization_schedule='strict'"
                )
            if settings.max_iter > LEGACY_OPTIMIZATION_ITERATIONS:
                raise ValueError(
                    "optimization_schedule='legacy' supports the published 200-step run or a shorter exact prefix"
                )
            if settings.continuation_steps is not None:
                raise ValueError("continuation_steps is not configurable when optimization_schedule='legacy'")
            return
        n_steps = settings.continuation_steps or settings.max_iter
        if not 1 <= n_steps <= settings.max_iter:
            raise ValueError("continuation_steps must be between 1 and max_iter")
        if settings.max_iter % n_steps:
            raise ValueError("max_iter must be divisible by continuation_steps")

    @staticmethod
    def _resolve_case_template(case_template: str | None) -> Path:
        configured = case_template or os.environ.get("ENGIBENCH_MTO2D_CASE_TEMPLATE")
        if not configured:
            raise FileNotFoundError(
                "No MTO2D case template configured. Pass config={'case_template': '/path/to/case'} "
                "or set ENGIBENCH_MTO2D_CASE_TEMPLATE."
            )
        path = Path(configured).expanduser().resolve()
        if not (path / "app").is_dir() or not (path / "src_TF").is_dir():
            raise FileNotFoundError(f"MTO2D case template must contain app/ and src_TF/: {path}")
        MTO2DRunner._validate_case_template(path)
        return path

    @staticmethod
    def _export_case_template(
        run_root: Path,
        case_dir: Path,
        settings: RunnerSettings,
    ) -> None:
        """Materialize the pristine case bundled in the published OCI image."""
        if settings.backend != "container" or settings.container_image is None:
            raise FileNotFoundError(
                "An image-provided MTO2D case template requires the container backend and a configured container image."
            )
        container_home = run_root / "container-home"
        container_tmp = run_root / "container-tmp"
        container_home.mkdir(exist_ok=True)
        container_tmp.mkdir(exist_ok=True)
        container.run(
            ["mto2d-export-case", "/work/case"],
            settings.container_image,
            mounts=((str(run_root), "/work"),),
            env={"HOME": "/work/container-home", "TMPDIR": "/work/container-tmp"},
            sync_uid=True,
        )
        if not (case_dir / "app").is_dir() or not (case_dir / "src_TF").is_dir():
            raise FileNotFoundError(
                "The configured MTO2D image did not export an app/src_TF case template. "
                "Rebuild it from docker/mto2d/Dockerfile.source in the EngiBench repository."
            )

    @staticmethod
    def _validate_schedule_runtime(case_template: Path, schedule: OptimizationSchedule) -> None:
        """Require a rebuilt executable that understands named schedules."""
        MTO2DRunner._validate_runtime_marker(
            case_template,
            f"optimization_schedule={schedule!r}",
        )

    @staticmethod
    def _validate_runtime_marker(case_template: Path, required_by: str) -> None:
        """Require the prepared-runtime version used by frozen and scheduled runs."""
        marker = case_template / SCHEDULE_RUNTIME_MARKER
        try:
            version = marker.read_text(encoding="ascii").strip()
        except OSError as error:
            raise FileNotFoundError(
                f"{required_by} requires a case rebuilt with the EngiBench runtime "
                "patches. Use the published MTO2D image or rebuild Dockerfile.source; "
                f"missing capability marker: {marker}"
            ) from error
        if version != SCHEDULE_RUNTIME_VERSION:
            raise ValueError(
                f"Unsupported MTO2D prepared-runtime version for {required_by}: "
                f"expected {SCHEDULE_RUNTIME_VERSION!r}, "
                f"got {version!r} in {marker}"
            )

    @staticmethod
    def _validate_case_template(case_template: Path) -> None:
        """Require the mesh, area switches, and gamma tail assumed by the API mapping."""
        continuation_source = case_template / "src_TF" / "continuation.H"
        if not continuation_source.is_file():
            raise FileNotFoundError(
                "MTO2D requires the continuation-aware warm-ready solver source; "
                f"missing {continuation_source}. The older solver silently ignores "
                "continuationProperties and cannot run the strict schedule."
            )

        app = case_template / "app"
        gamma_path = MTO2DRunner._gamma_template(app)
        gamma = parse_internal_field(gamma_path.read_text(encoding="ascii"), expected_count=GAMMA_CELL_COUNT)
        if np.any((gamma < 0.0) | (gamma > 1.0)):
            raise ValueError(f"MTO2D gamma template values must lie in [0, 1]: {gamma_path}")
        fixed_tail = gamma[DESIGN_CELL_COUNT:]
        if fixed_tail.size != FIXED_CELL_COUNT or not np.all(fixed_tail == 1.0):
            tail_min = float(np.min(fixed_tail)) if fixed_tail.size else math.nan
            tail_max = float(np.max(fixed_tail)) if fixed_tail.size else math.nan
            raise ValueError(
                "MTO2D requires the final 6,400 gamma cells to be the fixed-fluid region "
                f"(all gamma=1); got count={fixed_tail.size}, min={tail_min:.8g}, max={tail_max:.8g} "
                f"in {gamma_path}"
            )

        transport = app / "constant" / "transportProperties"
        transport_text = _mask_foam_comments(transport.read_text(encoding="utf-8"))
        for key, expected in _EXPECTED_TRANSPORT_SCALARS.items():
            matches = re.findall(rf"(?m)^\s*{re.escape(key)}\s+([^;]+?)\s*;", transport_text)
            if len(matches) != 1:
                raise ValueError(f"MTO2D requires exactly one {key!r} entry in {transport}")
            try:
                actual = float(matches[0])
            except ValueError as error:
                raise ValueError(f"MTO2D requires a numeric {key!r} entry in {transport}") from error
            if actual != expected:
                raise ValueError(f"MTO2D requires {key}={expected:g} in {transport}; got {actual:g}")

        block_mesh = app / "system" / "blockMeshDict"
        block_text = _mask_foam_comments(block_mesh.read_text(encoding="utf-8"))
        pattern = re.compile(
            r"\bhex\s*\([^()]*\)\s*"
            r"(?:(?P<zone>[A-Za-z_][A-Za-z0-9_]*)\s+)?"
            r"\(\s*(?P<nx>[0-9]+)\s+(?P<ny>[0-9]+)\s+(?P<nz>[0-9]+)\s*\)"
        )
        actual_layout = tuple(
            (
                match.group("zone"),
                (int(match.group("nx")), int(match.group("ny")), int(match.group("nz"))),
            )
            for match in pattern.finditer(block_text)
        )
        total_cells = sum(math.prod(dimensions) for _zone, dimensions in actual_layout)
        if total_cells != GAMMA_CELL_COUNT:
            raise ValueError(
                f"MTO2D blockMesh must define exactly {GAMMA_CELL_COUNT:,} cells; parsed {total_cells:,} from {block_mesh}"
            )
        if actual_layout != _EXPECTED_BLOCK_MESH_LAYOUT:
            raise ValueError(
                "MTO2D blockMesh must contain the ordered 80,000-cell design region followed by "
                f"the 6,400-cell zone_fluid region; got {actual_layout!r} in {block_mesh}"
            )

    def _prepare_case(
        self,
        case_dir: Path,
        design: npt.NDArray[np.float32],
        settings: RunnerSettings,
        kind: RunKind,
    ) -> None:
        app = case_dir / "app"
        # The solver's SIMP_initialize.H removes all seven history files at startup;
        # deleting stale template copies here is defense in depth for runs that fail
        # before solver initialization.
        for filename in (*HISTORY_FILES.values(), *DIAGNOSTIC_HISTORY_FILES):
            (app / filename).unlink(missing_ok=True)

        gamma_template = self._gamma_template(app)
        zero_dir = app / "0"
        zero_dir.mkdir(parents=True, exist_ok=True)
        write_half_design(design, gamma_template, zero_dir / "gamma", location="0")
        self._clear_stale_case_outputs(app)

        transport = app / "constant" / "transportProperties"
        control = app / "system" / "controlDict"
        decompose = app / "system" / "decomposeParDict"
        self._replace_dictionary_value(transport, "voluse", settings.volume_fraction)
        power_bound_start = settings.power_bound_start
        if power_bound_start is None:
            power_bound_start = settings.max_power_dissipation if kind == "simulate" or settings.mode == "warm" else 90.0
        self._replace_dictionary_value(transport, "D0", power_bound_start)
        self._replace_dictionary_value(transport, "D1", settings.max_power_dissipation)

        prepared = self._resolve_continuation(settings, kind)
        continuation = prepared.continuation

        self._replace_dictionary_value(transport, "movlim", prepared.movement_limit)
        self._upsert_plain_dictionary_value(transport, "updateDesign", kind == "optimize")
        self._replace_dictionary_value(transport, "qu", continuation.qu[0])
        self._replace_dictionary_value(transport, "alphaMax", continuation.alpha_max[0])
        self._replace_dictionary_value(transport, "alphamax", continuation.alpha_max[0])
        self._replace_dictionary_value(control, "endTime", prepared.iteration_count)
        self._replace_dictionary_value(control, "writeInterval", prepared.iteration_count)
        self._replace_dictionary_value(control, "writePrecision", 12)
        self._write_continuation(
            app / "constant" / "continuationProperties",
            continuation,
        )
        self._write_inlet_velocity(zero_dir / "U", settings.inlet_velocity)
        self._write_decomposition(decompose, settings.mpi_cores)

    @staticmethod
    def _resolve_continuation(settings: RunnerSettings, kind: RunKind) -> PreparedContinuation:
        """Resolve named schedules without approximating the legacy source timing."""
        if kind == "simulate":
            continuation = ContinuationSettings(
                optimization_schedule="strict",
                steps=1,
                profile=settings.continuation_profile,
                qu=(settings.qu_final, settings.qu_final),
                alpha_max=(settings.alpha_max_final, settings.alpha_max_final),
                heaviside=(settings.heaviside_final, settings.heaviside_final),
            )
            return PreparedContinuation(1, 0.0, continuation)
        if settings.optimization_schedule == "legacy":
            continuation = ContinuationSettings(
                optimization_schedule="legacy",
                steps=settings.max_iter,
                # The named C++ branch owns every post-update value. Keeping
                # the generic lists constant guarantees the source initial
                # values even when max_iter == 1 (the retained one-element
                # geometric helper otherwise selects its ``to`` endpoint).
                profile="constant",
                qu=(LEGACY_QU_START, LEGACY_QU_FINAL),
                alpha_max=(LEGACY_ALPHA_MAX_START, LEGACY_ALPHA_MAX_FINAL),
                heaviside=(LEGACY_HEAVISIDE_START, LEGACY_HEAVISIDE_FINAL),
            )
            return PreparedContinuation(settings.max_iter, settings.movement_limit, continuation)

        qu_final = settings.qu_final
        alpha_final = settings.alpha_max_final
        heaviside_final = settings.heaviside_final
        if settings.mode == "warm":
            qu_start = settings.qu_start if settings.qu_start is not None else qu_final
            alpha_start = settings.alpha_max_start if settings.alpha_max_start is not None else alpha_final
            heaviside_start = settings.heaviside_start if settings.heaviside_start is not None else heaviside_final
        else:
            qu_start = settings.qu_start if settings.qu_start is not None else 0.005
            alpha_start = settings.alpha_max_start if settings.alpha_max_start is not None else 2500.0
            heaviside_start = settings.heaviside_start if settings.heaviside_start is not None else 1.0
        continuation = ContinuationSettings(
            optimization_schedule="strict",
            steps=settings.continuation_steps or settings.max_iter,
            profile=settings.continuation_profile,
            qu=(qu_start, qu_final),
            alpha_max=(alpha_start, alpha_final),
            heaviside=(heaviside_start, heaviside_final),
        )
        return PreparedContinuation(settings.max_iter, settings.movement_limit, continuation)

    @staticmethod
    def _clear_stale_case_outputs(app: Path) -> None:
        """Remove copied time/decomposition outputs after preserving gamma in time zero."""
        for path in app.iterdir():
            is_old_time = path.is_dir() and path.name != "0" and path.name.isdigit()
            is_processor = path.is_dir() and re.fullmatch(r"processor[0-9]+", path.name) is not None
            if is_old_time or is_processor:
                shutil.rmtree(path)

    @staticmethod
    def _gamma_template(app: Path) -> Path:
        zero_gamma = app / "0" / "gamma"
        if zero_gamma.is_file():
            return zero_gamma
        candidates = [
            path / "gamma" for path in app.iterdir() if path.is_dir() and path.name.isdigit() and (path / "gamma").is_file()
        ]
        if not candidates:
            root_gamma = app / "gamma"
            if root_gamma.is_file():
                return root_gamma
            raise FileNotFoundError("Case template contains no app/0/gamma, numeric-time gamma, or app/gamma template")
        return max(candidates, key=lambda path: int(path.parent.name))

    @staticmethod
    def _replace_dictionary_value(path: Path, key: str, value: object) -> None:
        text = path.read_text(encoding="utf-8")
        pattern = re.compile(rf"(?m)^(\s*{re.escape(key)}\s+)([^;]*)(;.*)$")

        def replace(match: re.Match[str]) -> str:
            right_hand_side = match.group(2).rstrip()
            # OpenFOAM dimensioned entries such as
            # ``alphaMax alphaMax [0 0 -1 0 0 0 0] 2.5e3;`` must retain the
            # symbolic name and dimensions. Plain dictionary values are
            # replaced in full.
            if "]" in right_hand_side:
                prefix, separator, _old_value = right_hand_side.rpartition(" ")
                if not separator:
                    raise ValueError(f"Malformed dimensioned {key!r} entry in {path}")
                right_hand_side = f"{prefix} {value}"
            else:
                right_hand_side = str(value)
            return f"{match.group(1)}{right_hand_side}{match.group(3)}"

        updated, count = pattern.subn(replace, text, count=1)
        if count != 1:
            raise ValueError(f"Could not find exactly one {key!r} entry in {path}")
        path.write_text(updated, encoding="utf-8")

    @staticmethod
    def _upsert_plain_dictionary_value(path: Path, key: str, value: object) -> None:
        """Set or append one non-dimensioned OpenFOAM dictionary value."""
        text = path.read_text(encoding="utf-8")
        formatted = str(value).lower() if isinstance(value, bool) else str(value)
        pattern = re.compile(rf"(?m)^(\s*{re.escape(key)}\s+)([^;]*)(;.*)$")
        updated, count = pattern.subn(rf"\g<1>{formatted}\g<3>", text)
        if count > 1:
            raise ValueError(f"Found more than one {key!r} entry in {path}")
        if count == 0:
            updated = f"{text.rstrip()}\n{key} {formatted};\n"
        path.write_text(updated, encoding="utf-8")

    @staticmethod
    def _write_inlet_velocity(path: Path, inlet_velocity: float) -> None:
        text = path.read_text(encoding="utf-8")
        masked_text = _mask_foam_comments(text)
        inlet_matches = list(re.finditer(r"(?m)^\s*inlet\s*\{", masked_text))
        if len(inlet_matches) != 1:
            raise ValueError(f"MTO2D requires exactly one inlet boundary in {path}; found {len(inlet_matches)}")
        inlet_match = inlet_matches[0]
        start = inlet_match.end()
        depth = 1
        end = start
        while end < len(masked_text) and depth:
            depth += (masked_text[end] == "{") - (masked_text[end] == "}")
            end += 1
        if depth:
            raise ValueError(f"Unbalanced inlet boundary block in {path}")
        block = text[start : end - 1]
        masked_block = masked_text[start : end - 1]
        type_entries = [
            entry.strip()
            for entry in re.findall(
                r"(?m)^\s*type\s+([^;]+?)\s*;",
                masked_block,
            )
        ]
        if type_entries != ["fixedValue"]:
            raise ValueError(
                f"MTO2D requires exactly one 'type fixedValue;' entry in the inlet boundary of {path}; "
                f"found {type_entries!r}"
            )
        pattern = re.compile(r"(?m)^(\s*value\s+uniform\s+)\([^)]*\)(\s*;)")
        value_entries = list(pattern.finditer(masked_block))
        if len(value_entries) != 1:
            raise ValueError(f"MTO2D requires exactly one uniform inlet value in {path}; found {len(value_entries)}")
        value_entry = value_entries[0]
        replacement = f"{value_entry.group(1)}(0 {inlet_velocity:.12g} 0){value_entry.group(2)}"
        block = block[: value_entry.start()] + replacement + block[value_entry.end() :]
        path.write_text(text[:start] + block + text[end - 1 :], encoding="utf-8")

    @staticmethod
    def _write_decomposition(path: Path, mpi_cores: int) -> None:
        MTO2DRunner._replace_dictionary_value(path, "numberOfSubdomains", mpi_cores)
        MTO2DRunner._upsert_plain_dictionary_value(path, "method", "simple")
        text = path.read_text(encoding="utf-8")
        simple = re.search(r"\bsimpleCoeffs\s*\{(?P<body>.*?)\}", text, re.DOTALL)
        if simple is None:
            raise ValueError(f"Could not find simpleCoeffs in {path}")
        a = math.isqrt(mpi_cores)
        while mpi_cores % a:
            a -= 1
        b = mpi_cores // a
        body, count = re.subn(
            r"(?m)^(\s*n\s+)\([^)]*\)(\s*;)",
            rf"\g<1>({a} {b} 1)\g<2>",
            simple.group("body"),
            count=1,
        )
        if count != 1:
            raise ValueError(f"Could not find simpleCoeffs.n in {path}")
        path.write_text(text[: simple.start("body")] + body + text[simple.end("body") :], encoding="utf-8")

    @staticmethod
    def _write_continuation(
        path: Path,
        settings: ContinuationSettings,
    ) -> None:
        if settings.profile not in CONTINUATION_PROFILES:
            allowed = ", ".join(sorted(CONTINUATION_PROFILES))
            raise ValueError(f"continuation_profile must be one of: {allowed}")
        text = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      continuationProperties;
}}

n_steps         {settings.steps};
optimizationSchedule {settings.optimization_schedule};

qu
{{
    overallType    {settings.profile};
    from           {settings.qu[0]:.12g};
    to             {settings.qu[1]:.12g};
}}
alphaMax
{{
    overallType    {settings.profile};
    from           {settings.alpha_max[0]:.12g};
    to             {settings.alpha_max[1]:.12g};
}}
Heaviside
{{
    overallType    {settings.profile};
    from           {settings.heaviside[0]:.12g};
    to             {settings.heaviside[1]:.12g};
}}
"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _execute(self, case_dir: Path, settings: RunnerSettings, kind: RunKind) -> None:
        if settings.backend == "command":
            self._run_driver(case_dir, settings, kind)
            return
        self._run_container(case_dir, settings, kind)

    @staticmethod
    def _run_driver(case_dir: Path, settings: RunnerSettings, kind: RunKind) -> None:
        env = {
            **os.environ,
            "MTO2D_CASE_DIR": str(case_dir),
            "MTO2D_MPI_CORES": str(settings.mpi_cores),
            "MTO2D_RUN_KIND": kind,
            "MTO2D_SOLVER_EXECUTABLE": settings.solver_executable,
        }
        with (case_dir / "run.log").open("wb") as log:
            subprocess.run(
                list(settings.driver_command),
                cwd=case_dir,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=settings.timeout,
            )

    @staticmethod
    def _run_container(case_dir: Path, settings: RunnerSettings, _kind: RunKind) -> None:
        assert settings.container_image is not None
        container_home = case_dir.parent / "container-home"
        container_tmp = case_dir.parent / "container-tmp"
        container_home.mkdir(exist_ok=True)
        container_tmp.mkdir(exist_ok=True)
        executable = shlex.quote(settings.solver_executable)
        if settings.mpi_cores == 1:
            solve = executable
        else:
            solve = f"decomposePar; mpirun -np {settings.mpi_cores} {executable} -parallel"
        mesh = "if command -v mto2d-prepare-mesh >/dev/null 2>&1; then mto2d-prepare-mesh .; else blockMesh; fi"
        command = f"set -eu; exec > /work/case/run.log 2>&1; cd /work/case/app; {mesh}; {solve}"
        if settings.mpi_cores > 1:
            command += "; reconstructPar -latestTime"
        container.run(
            ["bash", "-lc", command],
            settings.container_image,
            mounts=((str(case_dir.parent), "/work"),),
            env={"HOME": "/work/container-home", "TMPDIR": "/work/container-tmp"},
            sync_uid=True,
        )

    @staticmethod
    def _read_histories(app: Path, *, expected_steps: int) -> dict[str, npt.NDArray[np.float64]]:
        histories = {name: _read_scalar_history(app / filename) for name, filename in HISTORY_FILES.items()}
        lengths = {name: len(values) for name, values in histories.items()}
        if set(lengths.values()) != {expected_steps}:
            raise ValueError(f"Expected {expected_steps} values in every solver history, got {lengths}")
        return histories

    @staticmethod
    def _latest_gamma(app: Path) -> Path:
        candidates = [
            path / "gamma"
            for path in app.iterdir()
            if path.is_dir() and path.name.isdigit() and int(path.name) > 0 and (path / "gamma").is_file()
        ]
        if not candidates:
            raise FileNotFoundError(f"No reconstructed numeric-time gamma found under {app}")
        return max(candidates, key=lambda path: int(path.parent.name))

    @staticmethod
    def _validate_frozen_output(app: Path, design: npt.NDArray[np.float32]) -> None:
        """Require a frozen solver step to preserve a finite input design."""
        try:
            written_design = read_half_design(MTO2DRunner._latest_gamma(app))
        except (OSError, ValueError) as error:
            raise ValueError(
                "Frozen simulation wrote a non-finite or invalid gamma. The solver executable "
                "likely lacks updateDesign support; use the published MTO2D image."
            ) from error
        if not np.allclose(
            written_design,
            design,
            rtol=0.0,
            atol=FROZEN_GAMMA_ABSOLUTE_TOLERANCE,
        ):
            max_difference = float(np.max(np.abs(written_design.astype(np.float64) - design.astype(np.float64))))
            raise ValueError(
                "Frozen simulation changed the design after objective evaluation; "
                f"maximum absolute difference is {max_difference:.8g}"
            )


def _read_scalar_history(path: Path) -> npt.NDArray[np.float64]:
    """Read a finite, nonempty one-value-per-line solver history."""
    try:
        values = np.loadtxt(path, dtype=np.float64, ndmin=1)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not parse scalar history {path}: {exc}") from exc
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"Scalar history must be nonempty and finite: {path}")
    return values
