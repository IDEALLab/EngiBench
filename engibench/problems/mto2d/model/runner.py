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
import time
from typing import Literal, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from engibench.problems.mto2d.model.design_io import read_half_design
from engibench.problems.mto2d.model.design_io import write_half_design
from engibench.utils import container

if TYPE_CHECKING:
    from collections.abc import Sequence

RunKind = Literal["simulate", "optimize"]
OptimizationMode = Literal["cold", "warm"]
Backend = Literal["local", "container", "command"]

HISTORY_FILES = {
    "mean_temperature": "meanT.txt",
    "power_dissipation": "Disspower.txt",
    "volume_residual": "Voluse.txt",
    "elapsed_time": "Time.txt",
}


@dataclass(frozen=True)
class RunnerSettings:
    """All values needed to prepare and execute one isolated solver case."""

    case_template: str | None
    inlet_velocity: float
    max_power_dissipation: float
    volume_fraction: float
    max_iter: int = 200
    mode: OptimizationMode = "cold"
    mpi_cores: int = 1
    backend: Backend = "local"
    container_image: str | None = None
    driver_command: tuple[str, ...] = ()
    solver_executable: str = "../src_TF/EXEC"
    build_solver: bool = False
    timeout: float | None = None
    work_dir: str | None = None
    retain_artifacts: bool = False
    retain_on_failure: bool = True
    continuation_steps: int | None = None
    qu_start: float | None = None
    qu_final: float = 0.019
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

    steps: int
    profile: str
    qu: tuple[float, float]
    alpha_max: tuple[float, float]
    heaviside: tuple[float, float]


class SolverRunError(RuntimeError):
    """MTO2D solver failure with an optional retained run directory."""

    def __init__(self, message: str, artifacts_path: Path | None = None) -> None:
        suffix = f"\nSolver artifacts retained at: {artifacts_path}" if artifacts_path is not None else ""
        super().__init__(message + suffix)
        self.artifacts_path = artifacts_path


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
        case_template = self._resolve_case_template(settings.case_template)
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
            shutil.copytree(case_template, case_dir)
            self._prepare_case(case_dir, design, settings, kind)
            self._execute(case_dir, settings, kind)
            histories = self._read_histories(
                case_dir / "app", expected_steps=1 if kind == "simulate" else settings.max_iter
            )
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
            if isinstance(exc, SolverRunError):
                if exc.artifacts_path is None and retained is not None:
                    raise SolverRunError(str(exc), retained) from exc
                raise
            raise SolverRunError(str(exc), retained) from exc
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
        if settings.timeout is not None and settings.timeout <= 0:
            raise ValueError("timeout must be positive")

    @staticmethod
    def _validate_backend_settings(settings: RunnerSettings) -> None:
        if settings.backend not in {"local", "container", "command"}:
            raise ValueError("backend must be 'local', 'container', or 'command'")
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
        return path

    def _prepare_case(
        self,
        case_dir: Path,
        design: npt.NDArray[np.float32],
        settings: RunnerSettings,
        kind: RunKind,
    ) -> None:
        app = case_dir / "app"
        for filename in HISTORY_FILES.values():
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
        self._replace_dictionary_value(transport, "D1", settings.max_power_dissipation)

        if kind == "simulate":
            iteration_count = 1
            movement_limit = 0.0
            qu_start = qu_final = settings.qu_final
            alpha_start = alpha_final = settings.alpha_max_final
            heaviside_start = heaviside_final = settings.heaviside_final
            continuation_steps = 1
        else:
            iteration_count = settings.max_iter
            movement_limit = settings.movement_limit
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
            continuation_steps = settings.continuation_steps or settings.max_iter

        self._replace_dictionary_value(transport, "movlim", movement_limit)
        self._replace_dictionary_value(transport, "qu", qu_start)
        self._replace_dictionary_value(transport, "alphaMax", alpha_start)
        self._replace_dictionary_value(transport, "alphamax", alpha_start)
        self._replace_dictionary_value(control, "endTime", iteration_count)
        self._replace_dictionary_value(control, "writeInterval", iteration_count)
        self._write_continuation(
            app / "constant" / "continuationProperties",
            ContinuationSettings(
                steps=continuation_steps,
                profile=settings.continuation_profile,
                qu=(qu_start, qu_final),
                alpha_max=(alpha_start, alpha_final),
                heaviside=(heaviside_start, heaviside_final),
            ),
        )
        self._write_inlet_velocity(zero_dir / "U", settings.inlet_velocity)
        self._write_decomposition(decompose, settings.mpi_cores)

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
    def _write_inlet_velocity(path: Path, inlet_velocity: float) -> None:
        text = path.read_text(encoding="utf-8")
        inlet_match = re.search(r"\binlet\s*\{", text)
        if inlet_match is None:
            raise ValueError(f"Could not find inlet boundary in {path}")
        start = inlet_match.end()
        depth = 1
        end = start
        while end < len(text) and depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        if depth:
            raise ValueError(f"Unbalanced inlet boundary block in {path}")
        block = text[start : end - 1]
        pattern = re.compile(r"(?m)^(\s*value\s+uniform\s+)\([^)]*\)(\s*;)")
        block, count = pattern.subn(rf"\g<1>(0 {inlet_velocity:.12g} 0)\g<2>", block, count=1)
        if count != 1:
            raise ValueError(f"Could not find inlet uniform value in {path}")
        path.write_text(text[:start] + block + text[end - 1 :], encoding="utf-8")

    @staticmethod
    def _write_decomposition(path: Path, mpi_cores: int) -> None:
        MTO2DRunner._replace_dictionary_value(path, "numberOfSubdomains", mpi_cores)
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
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", settings.profile):
            raise ValueError("continuation_profile must be an OpenFOAM word")
        text = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      continuationProperties;
}}

n_steps         {settings.steps};

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
        if settings.backend == "container":
            self._run_container(case_dir, settings, kind)
            return
        self._run_local(case_dir, settings, kind)

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
    def _run_local(case_dir: Path, settings: RunnerSettings, kind: RunKind) -> None:
        started = time.monotonic()
        commands: list[tuple[Sequence[str], Path]] = []
        if settings.build_solver:
            commands.append((("wmake",), case_dir / "src_TF"))
        app = case_dir / "app"
        commands.extend(
            [
                (("blockMesh",), app),
                (("decomposePar",), app),
                (
                    (
                        "mpirun",
                        "-np",
                        str(settings.mpi_cores),
                        settings.solver_executable,
                        "-parallel",
                    ),
                    app,
                ),
            ]
        )
        if kind == "optimize":
            commands.append((("reconstructPar", "-latestTime"), app))
        with (case_dir / "run.log").open("wb") as log:
            for command, cwd in commands:
                remaining = None
                if settings.timeout is not None:
                    remaining = settings.timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, settings.timeout)
                subprocess.run(
                    list(command),
                    cwd=cwd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=remaining,
                )

    @staticmethod
    def _run_container(case_dir: Path, settings: RunnerSettings, kind: RunKind) -> None:
        assert settings.container_image is not None
        container_home = case_dir.parent / "container-home"
        container_tmp = case_dir.parent / "container-tmp"
        container_home.mkdir(exist_ok=True)
        container_tmp.mkdir(exist_ok=True)
        build = "cd /work/case/src_TF && wmake && " if settings.build_solver else ""
        command = (
            "set -eu; "
            "exec > /work/case/run.log 2>&1; "
            f"{build}"
            "cd /work/case/app; "
            "blockMesh; "
            "decomposePar; "
            f"mpirun -np {settings.mpi_cores} {shlex.quote(settings.solver_executable)} -parallel"
        )
        if kind == "optimize":
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
