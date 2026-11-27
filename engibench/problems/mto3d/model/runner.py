"""Isolated OpenFOAM runner for the MTO3D problem."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Literal, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from engibench.problems.mto3d.model.design_io import check_fixed_cells
from engibench.problems.mto3d.model.design_io import read_xh
from engibench.problems.mto3d.model.design_io import write_half_design
from engibench.problems.mto3d.model.design_io import XH_CELL_COUNT
from engibench.problems.mto3d.model.design_io import xh_to_half_design
from engibench.utils import container

if TYPE_CHECKING:
    from collections.abc import Callable

RunKind = Literal["simulate", "optimize"]
OptimizationMode = Literal["cold", "warm"]

SOLVER_EXECUTABLE = "../src/EXEC"
"""Path of the solver binary relative to the case ``app/`` directory."""

XH_TEMPLATE = "xh_template.gz"
"""Final-time ``xh`` field (exact fixed cells) used as the writer template, at the case root."""

RUNTIME_VERSION = "1"
RUNTIME_MARKER = ".engibench-mto3d-runtime-version"
"""Capability marker: the image's solver carries the EngiBench frozen-evaluation patch."""

FROZEN_STATUS_FILE = "frozen_status.txt"
"""Full-precision record written by a frozen evaluation (``updateDesign no``)."""

LEGACY_NS_MAX_ITER = 150
"""SIMPLE passes per optimization iteration in the published campaigns (``NSmaxIter``)."""

DEFAULT_FROZEN_NS_MAX_ITER = 20_000
"""Upper bound on SIMPLE passes for a frozen evaluation, which must converge to tol 1e-6."""

FROZEN_ITERATIONS = 100
"""Solver iterations of a frozen evaluation; the design stays fixed throughout.

The energy equation's convection term (``Gauss linearUpwind``) is applied by OpenFOAM as an explicit
correction computed from the current temperature, so a single solve from the uniform initial field
returns the first-order upwind temperature (mean temperature up to 16% low for the published designs)
and depends on that initial field. Each iteration re-assembles and re-solves the energy equation (and
runs at least one SIMPLE pass). The correction converges geometrically, by a factor of about 0.6 per
iteration in tests, so 100 iterations make the result independent of the initial temperature. The
image's solver stops as soon as an iteration ends with the flow unconverged (all ``NSmaxIter`` passes
used), so a flow that does not converge costs one capped solve, not 100."""

FROZEN_CONVERGENCE_WINDOW = 21
"""Final iterations whose 6-significant-figure mean temperature must agree within one unit."""

FROZEN_T_TOLERANCE = 1e-10
FROZEN_T_MAX_ITER = 20_000
"""Energy-equation solver settings of a frozen evaluation. The case's tolerance (1e-5) suits the
optimizer's warm starts but would stop updating the correction once its change fell below 1e-5."""

WRITE_PRECISION = 12
"""``writePrecision`` for every run. The legacy value 3 made ``decomposePar`` round the supplied
design (and the inlet velocity) to 3 significant figures."""

DEFAULT_MPI_CORES = 16
DECOMPOSITIONS = {
    1: (1, 1, 1),
    2: (1, 1, 2),
    4: (2, 1, 2),
    8: (2, 2, 2),
    16: (2, 2, 4),
    32: (2, 2, 8),
    64: (4, 2, 8),
}
"""``simple`` decompositions by rank count. Only 16 ranks, ``(2 2 4)`` -- the decomposition of
every published case -- is parity-tested; parallel reductions make other counts differ slightly."""

FROZEN_DESIGN_ABSOLUTE_TOLERANCE = 1e-7
"""Absolute tolerance used to verify that a frozen evaluation preserves the design."""

OPTIMIZED_XH_TOLERANCE = 1e-6
"""Excursion outside ``[0, 1]`` tolerated (and clipped) in an optimizer-written ``xh``. The PDE filter is
solved iteratively and can overshoot by a few 1e-9, which the Heaviside projection keeps: 13,461 fixed fluid
cells reach 1.0000000087 after one cold iteration of case 0. Larger excursions raise."""

RUN_LOG_TAIL_BYTES = 16 * 1024
RUN_LOG_TAIL_LINES = 20

HISTORY_FILES = {
    "mean_temperature": "MeanT.txt",
    "power_dissipation": "PowerDiss.txt",
    "volume_residual": "Voluse.txt",
    "elapsed_time": "Time.txt",
}
DIAGNOSTIC_HISTORY_FILES = ("BI.txt", "viol.txt", "normL2.txt", "qu.txt", "aMax.txt", "Heaviside.txt")


@dataclass(frozen=True)
class RunnerSettings:
    """Solver-facing values for one case; they mirror ``MTO3D.Config``.

    Image selection, working directory, timeout, and artifact retention belong to
    :class:`MTO3DRunner`.
    """

    inlet_velocity: float
    max_power_dissipation: float
    volfrac: float
    max_iter: int = 3000
    mode: OptimizationMode = "cold"
    mpi_cores: int = DEFAULT_MPI_CORES
    movement_limit: float = 0.4
    frozen_ns_max_iter: int = DEFAULT_FROZEN_NS_MAX_ITER
    inlet_significant_figures: int | None = 3


@dataclass(frozen=True)
class SolverRun:
    """Parsed output from one solver invocation."""

    final_design: npt.NDArray[np.float32]
    mean_temperature: npt.NDArray[np.float64]
    power_dissipation: npt.NDArray[np.float64]
    volume_residual: npt.NDArray[np.float64]
    elapsed_time: npt.NDArray[np.float64]
    frozen_status: dict[str, float] | None
    artifacts_path: str | None


class SolverRunError(RuntimeError):
    """MTO3D solver failure with an optional retained run directory."""

    def __init__(self, message: str, artifacts_path: Path | None = None) -> None:
        suffix = f"\nSolver artifacts retained at: {artifacts_path}" if artifacts_path is not None else ""
        super().__init__(message + suffix)
        self.message = message
        self.artifacts_path = artifacts_path


def round_significant(value: float, digits: int) -> float:
    """Round to ``digits`` significant figures, as OpenFOAM's ``writePrecision`` does on output."""
    return float(f"{value:.{digits}g}")


def _run_log_diagnostics(path: Path) -> str:
    if not path.is_file():
        return ""
    label = f"\nSolver log: {path}"
    try:
        with path.open("rb") as log:
            log.seek(0, os.SEEK_END)
            log.seek(max(0, log.tell() - RUN_LOG_TAIL_BYTES))
            text = log.read().decode("utf-8", errors="replace")
    except OSError as error:
        return f"{label}\nUnable to read solver log: {error}"
    tail = text.splitlines()[-RUN_LOG_TAIL_LINES:]
    if not tail:
        return f"{label}\nSolver log is empty."
    return f"{label}\nLast {len(tail)} solver log lines:\n" + "\n".join(tail)


def _mask_foam_comments(text: str) -> str:
    """Replace OpenFOAM comments with whitespace while preserving indices."""

    def mask(match: re.Match[str]) -> str:
        return re.sub(r"[^\r\n]", " ", match.group(0))

    return re.sub(r"/\*.*?\*/|//[^\r\n]*", mask, text, flags=re.DOTALL)


def _block_span(text: str, keyword: str, path: Path) -> tuple[int, int]:
    """Return the span of the body of the unique top-level-or-nested dictionary ``keyword { ... }``."""
    masked = _mask_foam_comments(text)
    matches = list(re.finditer(rf"(?m)^\s*{re.escape(keyword)}\s*\{{", masked))
    if len(matches) != 1:
        raise ValueError(f"MTO3D requires exactly one {keyword!r} dictionary in {path}; found {len(matches)}")
    start = matches[0].end()
    depth, end = 1, start
    while end < len(masked) and depth:
        depth += (masked[end] == "{") - (masked[end] == "}")
        end += 1
    if depth:
        raise ValueError(f"Unbalanced {keyword!r} dictionary in {path}")
    return start, end - 1


class MTO3DRunner:
    """Prepare and run a copied MTO3D OpenFOAM case.

    Host concerns live here rather than in ``MTO3D.Config``: which image to run, where to put the
    isolated run directory, how long to allow, and what to keep afterwards. Execution goes through
    :mod:`engibench.utils.container`, the same seam used by the other containerized problems.
    """

    def __init__(
        self,
        image: str | None = None,
        *,
        case_template: str | None = None,
        work_dir: str | None = None,
        timeout: float | None = None,
        retain_artifacts: bool = False,
        retain_on_failure: bool = True,
        execute: Callable[[Path, RunnerSettings, RunKind], None] | None = None,
    ) -> None:
        """Configure how and where solver cases are executed.

        Args:
            image: Container image to run.
            case_template: Optional local case (``app/``, ``src/``, ``xh_template.gz``), used
                instead of the one bundled in the image. The image still supplies the runtime unless
                a test execution callback is injected.
            work_dir: Parent directory for the isolated run directory.
            timeout: Optional wall-clock limit in seconds for each container invocation.
            retain_artifacts: Keep the run directory after a successful run.
            retain_on_failure: Keep the run directory after a failed run.
            execute: Optional replacement for container execution, for tests that exercise case
                preparation and parsing without a container runtime.
        """
        if timeout is not None and timeout <= 0.0:
            raise ValueError("timeout must be positive")
        self.image = image
        self.case_template = case_template
        self.work_dir = work_dir
        self.timeout = timeout
        self.retain_artifacts = retain_artifacts
        self.retain_on_failure = retain_on_failure
        self._execute_fn = execute or self._run_container

    def run(self, design: npt.NDArray[np.float32], settings: RunnerSettings, *, kind: RunKind) -> SolverRun:
        """Execute a frozen simulation or an MMA optimization."""
        self.validate_settings(settings)
        run_root = Path(tempfile.mkdtemp(prefix="engibench-mto3d-", dir=self.work_dir)).resolve()
        case_dir = run_root / "case"
        succeeded = False
        keep = self.retain_artifacts
        try:
            self._materialize_case(run_root, case_dir)
            self._validate_case_template(case_dir)
            self._prepare_case(case_dir, design, settings, kind)
            self._execute_fn(case_dir, settings, kind)
            app = case_dir / "app"
            frozen_status: dict[str, float] | None = None
            if kind == "simulate":
                # The solver stops early when the flow does not converge; report that before the row count.
                frozen_status = self._read_frozen_status(app, settings)
            histories = self._read_histories(app, frozen=kind == "simulate")
            if frozen_status is not None:
                self._check_frozen_convergence(histories["mean_temperature"])
                initial_passes = self._initial_simple_passes(case_dir / "run.log")
                if initial_passes is not None:
                    frozen_status["ns_passes_initial"] = float(initial_passes)
                self._validate_frozen_output(app, design)
                final_design = np.asarray(design, dtype=np.float32).copy()
            else:
                final_design = self._read_final_design(app)
            succeeded = True
            return SolverRun(
                final_design=final_design,
                frozen_status=frozen_status,
                artifacts_path=str(run_root) if keep else None,
                **histories,
            )
        except Exception as exc:
            keep = self.retain_on_failure
            message = exc.message if isinstance(exc, SolverRunError) else str(exc)
            raise SolverRunError(message + _run_log_diagnostics(case_dir / "run.log"), run_root if keep else None) from exc
        finally:
            if (succeeded and not self.retain_artifacts) or (not succeeded and not keep):
                shutil.rmtree(run_root, ignore_errors=True)

    @staticmethod
    def validate_settings(settings: RunnerSettings) -> None:
        """Validate rules that span several settings."""
        if settings.mpi_cores not in DECOMPOSITIONS:
            allowed = ", ".join(str(n) for n in DECOMPOSITIONS)
            raise ValueError(f"mpi_cores must be one of {allowed}; got {settings.mpi_cores}")
        if settings.inlet_significant_figures is not None and settings.inlet_significant_figures < 1:
            raise ValueError("inlet_significant_figures must be positive or None")

    def _materialize_case(self, run_root: Path, case_dir: Path) -> None:
        if self.case_template is not None:
            shutil.copytree(Path(self.case_template).expanduser().resolve(), case_dir)
            return
        if self.image is None:
            raise FileNotFoundError("No MTO3D container image configured. Pass MTO3DRunner(image=...).")
        for name in ("container-home", "container-tmp"):
            (run_root / name).mkdir(exist_ok=True)
        container.run(
            ["mto3d-export-case", "/work/case"],
            self.image,
            mounts=((str(run_root), "/work"),),
            env={"HOME": "/work/container-home", "TMPDIR": "/work/container-tmp"},
            sync_uid=True,
            timeout=self.timeout,
        )

    @staticmethod
    def _validate_case_template(case_dir: Path) -> None:
        """Require the solver binary, the runtime marker, the writer template and the 3D mesh."""
        if not (case_dir / "src" / "EXEC").is_file():
            raise FileNotFoundError(f"MTO3D requires its compiled solver executable: {case_dir / 'src' / 'EXEC'}")
        marker = case_dir / RUNTIME_MARKER
        try:
            version = marker.read_text(encoding="ascii").strip()
        except OSError as error:
            raise FileNotFoundError(
                f"MTO3D requires a case built with the EngiBench solver patches; missing {marker}"
            ) from error
        if version != RUNTIME_VERSION:
            raise ValueError(f"Unsupported MTO3D runtime version: expected {RUNTIME_VERSION!r}, got {version!r}")
        check_fixed_cells(read_xh(case_dir / XH_TEMPLATE).values, name=XH_TEMPLATE)
        block_mesh = case_dir / "app" / "system" / "blockMeshDict"
        text = _mask_foam_comments(block_mesh.read_text(encoding="utf-8"))
        counts = [
            math.prod(int(n) for n in m.group(1).split())
            for m in re.finditer(r"\bhex\s*\([^()]*\)\s*(?:[A-Za-z_]\w*\s+)?\(\s*(\d+\s+\d+\s+\d+)\s*\)", text)
        ]
        if sum(counts) != XH_CELL_COUNT:
            raise ValueError(f"MTO3D blockMesh must define {XH_CELL_COUNT:,} cells; parsed {sum(counts):,}")

    def _prepare_case(
        self, case_dir: Path, design: npt.NDArray[np.float32], settings: RunnerSettings, kind: RunKind
    ) -> None:
        app = case_dir / "app"
        self._clear_stale_case_outputs(app)
        zero = app / "0"
        for name in ("xh", "xh.gz"):
            (zero / name).unlink(missing_ok=True)
        # A uniform volfrac start with cold continuation is the historical initialization: writing no
        # 0/xh keeps the solver bitwise on the legacy path (the filter's initial guess differs otherwise).
        legacy_cold_start = (
            kind == "optimize" and settings.mode == "cold" and bool(np.all(design == np.float32(settings.volfrac)))
        )
        if not legacy_cold_start:
            write_half_design(design, case_dir / XH_TEMPLATE, zero / "xh.gz", location="0")

        opt = app / "constant" / "optProperties"
        self._replace_dictionary_value(opt, "voluse", settings.volfrac)
        self._replace_dictionary_value(opt, "PowerDissMax", settings.max_power_dissipation)
        self._upsert_plain_dictionary_value(opt, "updateDesign", "yes" if kind == "optimize" else "no")
        if kind == "optimize":
            self._replace_dictionary_value(opt, "movlim", settings.movement_limit)
        final_physics = kind == "simulate" or settings.mode == "warm"
        self._set_continuation_flag(opt, "NoContinuation", 1 if final_physics else 0)

        transport = app / "constant" / "transportProperties"
        ns_max = settings.frozen_ns_max_iter if kind == "simulate" else LEGACY_NS_MAX_ITER
        self._replace_dictionary_value(transport, "NSmaxIter", ns_max)

        control = app / "system" / "controlDict"
        self._replace_dictionary_value(control, "endTime", FROZEN_ITERATIONS if kind == "simulate" else settings.max_iter)
        self._replace_dictionary_value(control, "writePrecision", WRITE_PRECISION)
        if kind == "simulate":
            self._replace_dictionary_value(control, "writeInterval", FROZEN_ITERATIONS)
            self._set_energy_solver(app / "system" / "fvSolution", FROZEN_T_TOLERANCE, FROZEN_T_MAX_ITER)

        inlet = settings.inlet_velocity
        if settings.inlet_significant_figures is not None:
            inlet = round_significant(inlet, settings.inlet_significant_figures)
        self._write_inlet_velocity(zero / "U", inlet)
        self._write_decomposition(app / "system" / "decomposeParDict", settings.mpi_cores)

    @staticmethod
    def _clear_stale_case_outputs(app: Path) -> None:
        for name in (*HISTORY_FILES.values(), *DIAGNOSTIC_HISTORY_FILES, FROZEN_STATUS_FILE):
            (app / name).unlink(missing_ok=True)
        for path in app.iterdir():
            stale_time = path.is_dir() and path.name != "0" and path.name.isdigit()
            stale_processor = path.is_dir() and re.fullmatch(r"processor\d+", path.name) is not None
            if stale_time or stale_processor:
                shutil.rmtree(path)

    @staticmethod
    def _replace_dictionary_value(path: Path, key: str, value: object) -> None:
        text = path.read_text(encoding="utf-8")
        pattern = re.compile(rf"(?m)^(\s*{re.escape(key)}\s+)([^;]*)(;.*)$")

        def replace(match: re.Match[str]) -> str:
            right_hand_side = match.group(2).rstrip()
            if "]" in right_hand_side:  # dimensioned entry: keep name and dimensions
                prefix, _separator, _old = right_hand_side.rpartition(" ")
                right_hand_side = f"{prefix} {value}"
            else:
                right_hand_side = str(value)
            return f"{match.group(1)}{right_hand_side}{match.group(3)}"

        updated, count = pattern.subn(replace, text)
        if count != 1:
            raise ValueError(f"Could not find exactly one {key!r} entry in {path}")
        path.write_text(updated, encoding="utf-8")

    @staticmethod
    def _upsert_plain_dictionary_value(path: Path, key: str, value: str) -> None:
        """Set or insert (before the continuation dictionary) one top-level plain value."""
        text = path.read_text(encoding="utf-8")
        pattern = re.compile(rf"(?m)^({re.escape(key)}\s+)([^;]*)(;.*)$")
        updated, count = pattern.subn(rf"\g<1>{value}\g<3>", text)
        if count > 1:
            raise ValueError(f"Found more than one {key!r} entry in {path}")
        if count == 0:
            updated, inserted = re.subn(r"(?m)^(continuation\s*\n\s*\{)", f"{key} {value};\n\\1", text)
            if inserted != 1:
                updated = f"{text.rstrip()}\n{key} {value};\n"
        path.write_text(updated, encoding="utf-8")

    @staticmethod
    def _set_continuation_flag(path: Path, key: str, value: int) -> None:
        text = path.read_text(encoding="utf-8")
        start, end = _block_span(text, "continuation", path)
        body = text[start:end]
        pattern = re.compile(rf"(?m)^(\s*{re.escape(key)}\s+)([^;]*)(;.*)$")
        new_body, count = pattern.subn(rf"\g<1>{value}\g<3>", body)
        if count > 1:
            raise ValueError(f"Found more than one {key!r} entry in the continuation dictionary of {path}")
        if count == 0:
            new_body = f"\n    {key}      {value};" + body
        path.write_text(text[:start] + new_body + text[end:], encoding="utf-8")

    @staticmethod
    def _set_energy_solver(path: Path, tolerance: float, max_iter: int) -> None:
        """Set the ``"(T)"`` linear solver's tolerance and iteration cap; ``relTol`` must already be 0."""
        text = path.read_text(encoding="utf-8")
        start, end = _block_span(text, '"(T)"', path)
        body = text[start:end]
        masked = _mask_foam_comments(body)
        if [v.strip() for v in re.findall(r"(?m)^\s*relTol\s+([^;]+);", masked)] != ["0"]:
            raise ValueError(f"MTO3D requires exactly one 'relTol 0;' in the (T) solver of {path}")
        body, count = re.subn(r"(?m)^(\s*tolerance\s+)[^;]*(;)", rf"\g<1>{tolerance:g}\g<2>", body)
        if count != 1:
            raise ValueError(f"Could not find exactly one tolerance entry in the (T) solver of {path}")
        body, count = re.subn(r"(?m)^(\s*maxIter\s+)[^;]*(;)", rf"\g<1>{max_iter}\g<2>", body)
        if count == 0:
            body = f"{body.rstrip()}\n        maxIter         {max_iter};\n    "
        path.write_text(text[:start] + body + text[end:], encoding="utf-8")

    @staticmethod
    def _write_inlet_velocity(path: Path, inlet_velocity: float) -> None:
        text = path.read_text(encoding="utf-8")
        start, end = _block_span(text, "inlet", path)
        block = text[start:end]
        masked = _mask_foam_comments(block)
        types = [t.strip() for t in re.findall(r"(?m)^\s*type\s+([^;]+?)\s*;", masked)]
        if types != ["fixedValue"]:
            raise ValueError(f"MTO3D requires exactly one 'type fixedValue;' in the inlet boundary of {path}; got {types}")
        value_re = re.compile(r"(?m)^(\s*value\s+uniform\s+)\(([^)]*)\)(\s*;)")
        matches = list(value_re.finditer(masked))
        if len(matches) != 1:
            raise ValueError(f"MTO3D requires exactly one uniform inlet value in {path}; found {len(matches)}")
        match = matches[0]
        replacement = f"{match.group(1)}(0 0 {inlet_velocity:.12g}){match.group(3)}"
        block = block[: match.start()] + replacement + block[match.end() :]
        path.write_text(text[:start] + block + text[end:], encoding="utf-8")

    @staticmethod
    def _write_decomposition(path: Path, mpi_cores: int) -> None:
        MTO3DRunner._replace_dictionary_value(path, "numberOfSubdomains", mpi_cores)
        text = path.read_text(encoding="utf-8")
        start, end = _block_span(text, "simpleCoeffs", path)
        nx, ny, nz = DECOMPOSITIONS[mpi_cores]
        body, count = re.subn(r"(?m)^(\s*n\s+)\([^)]*\)(\s*;)", rf"\g<1>({nx} {ny} {nz})\g<2>", text[start:end], count=1)
        if count != 1:
            raise ValueError(f"Could not find simpleCoeffs.n in {path}")
        path.write_text(text[:start] + body + text[end:], encoding="utf-8")

    def _run_container(self, case_dir: Path, settings: RunnerSettings, _kind: RunKind) -> None:
        if self.image is None:
            raise FileNotFoundError("No MTO3D container image configured. Pass MTO3DRunner(image=...).")
        for name in ("container-home", "container-tmp"):
            (case_dir.parent / name).mkdir(exist_ok=True)
        if settings.mpi_cores == 1:
            solve = SOLVER_EXECUTABLE
        else:
            # --oversubscribe lets the parity-tested 16 ranks run on machines with fewer cores (for
            # example an 8-vCPU Docker VM, where Open MPI otherwise refuses to start); results depend
            # only on the rank count, and nothing changes where enough cores are available.
            solve = (
                f"decomposePar; mpirun --oversubscribe -np {settings.mpi_cores} {SOLVER_EXECUTABLE} -parallel; "
                "reconstructPar -latestTime"
            )
        command = f"set -eu; exec > /work/case/run.log 2>&1; cd /work/case/app; mto3d-prepare-mesh .; {solve}"
        container.run(
            ["bash", "-lc", command],
            self.image,
            mounts=((str(case_dir.parent), "/work"),),
            env={"HOME": "/work/container-home", "TMPDIR": "/work/container-tmp"},
            sync_uid=True,
            timeout=self.timeout,
        )

    @staticmethod
    def _read_histories(app: Path, *, frozen: bool) -> dict[str, npt.NDArray[np.float64]]:
        histories: dict[str, npt.NDArray[np.float64]] = {}
        iterations: dict[str, npt.NDArray[np.float64]] = {}
        for name, filename in HISTORY_FILES.items():
            iterations[name], histories[name] = _read_iteration_history(app / filename)
        lengths = {name: len(values) for name, values in histories.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Solver histories have different lengths: {lengths}")
        steps = next(iter(iterations.values()))
        if any(not np.array_equal(steps, other) for other in iterations.values()):
            raise ValueError("Solver histories do not share the same iteration identifiers")
        if not np.array_equal(steps, np.arange(1, len(steps) + 1)):
            raise ValueError("Solver history iteration identifiers must be 1, 2, ..., n")
        if frozen and len(steps) != FROZEN_ITERATIONS:
            raise ValueError(f"A frozen evaluation must write {FROZEN_ITERATIONS} history rows; got {len(steps)}")
        return histories

    @staticmethod
    def _check_frozen_convergence(mean_temperature: npt.NDArray[np.float64]) -> None:
        """Require the energy equation's deferred correction to have converged.

        The history is written at 6 significant figures; over the final iterations it may vary by at
        most one unit in the last digit.
        """
        tail = mean_temperature[-FROZEN_CONVERGENCE_WINDOW:]
        unit = 10.0 ** (math.floor(math.log10(max(abs(float(tail[-1])), 1e-300))) - 5)
        spread = float(tail.max() - tail.min())
        if spread > unit * (1.0 + 1e-6):
            raise ValueError(
                f"Frozen evaluation did not converge the energy equation within {FROZEN_ITERATIONS} iterations: "
                f"the mean temperature varied by {spread:.3g} over the last {FROZEN_CONVERGENCE_WINDOW} iterations"
            )

    @staticmethod
    def _initial_simple_passes(log_path: Path) -> int | None:
        """SIMPLE passes of the first iteration (flow from rest): pressure solves before the first energy solve."""
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        first_energy_solve = text.find("Solving for T,")
        if first_energy_solve < 0:
            return None
        return text.count("Solving for p,", 0, first_energy_solve)

    @staticmethod
    def _read_frozen_status(app: Path, settings: RunnerSettings) -> dict[str, float]:
        path = app / FROZEN_STATUS_FILE
        try:
            lines = path.read_text(encoding="ascii").splitlines()
        except OSError as error:
            raise FileNotFoundError(
                f"Frozen evaluation wrote no {FROZEN_STATUS_FILE}; the solver likely lacks the EngiBench "
                "frozen-evaluation patch."
            ) from error
        status = {key: float(value) for key, value in (line.split(None, 1) for line in lines if line.strip())}
        if status.get("ns_converged") != 1.0:
            raise ValueError(
                f"Frozen evaluation did not converge the flow within {settings.frozen_ns_max_iter} SIMPLE passes "
                f"(initial residuals U={status.get('initial_residual_U')}, p={status.get('initial_residual_p')})"
            )
        return status

    @staticmethod
    def _latest_xh(app: Path) -> Path:
        time_dirs = [d for d in app.iterdir() if d.is_dir() and d.name.isdigit() and int(d.name) > 0]
        candidates = [f for d in time_dirs for f in (d / "xh.gz", d / "xh") if f.is_file()]
        if not candidates:
            raise FileNotFoundError(f"No reconstructed numeric-time xh found under {app}")
        return max(candidates, key=lambda f: int(f.parent.name))

    @staticmethod
    def _read_final_design(app: Path) -> npt.NDArray[np.float32]:
        """Return the design cells of the last written ``xh``.

        The solver writes ``xh`` after the MMA update, PDE filter and Heaviside projection, which act
        on every cell, and re-imposes the fixed solid/fluid cells only at the start of the next
        iteration (``update.H``). Before the continuation has converged, the written fixed cells
        therefore hold projected values (0.01-0.06 in the solid shells after three cold iterations).
        They are not part of the design: the design cells are exactly what the next iteration
        evaluates, and the writer restores the fixed cells from the template. Values within
        :data:`OPTIMIZED_XH_TOLERANCE` outside ``[0, 1]`` are clipped.
        """
        bounds = (-OPTIMIZED_XH_TOLERANCE, 1.0 + OPTIMIZED_XH_TOLERANCE)
        field = read_xh(MTO3DRunner._latest_xh(app), bounds=bounds)
        return xh_to_half_design(np.clip(field.values, 0.0, 1.0))

    @staticmethod
    def _validate_frozen_output(app: Path, design: npt.NDArray[np.float32]) -> None:
        path = MTO3DRunner._latest_xh(app)
        written = xh_to_half_design(read_xh(path).values)
        if not np.allclose(written, design, rtol=0.0, atol=FROZEN_DESIGN_ABSOLUTE_TOLERANCE):
            difference = float(np.max(np.abs(written.astype(np.float64) - design.astype(np.float64))))
            raise ValueError(f"Frozen evaluation changed the design; maximum absolute difference {difference:.8g}")


def _read_iteration_history(path: Path) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Read a finite, nonempty two-column ``iteration value`` solver history."""
    try:
        table = np.loadtxt(path, dtype=np.float64, ndmin=2)
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not parse solver history {path}: {error}") from error
    if table.shape[0] == 0 or table.shape[1] != 2 or not np.all(np.isfinite(table)):  # noqa: PLR2004
        raise ValueError(f"Solver history must be a nonempty, finite two-column table: {path}")
    return table[:, 0], table[:, 1]
