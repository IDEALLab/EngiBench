"""Slurm executor for parameter space discovery."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
import importlib
import itertools
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Generic, TYPE_CHECKING, TypeVar

from numpy import typing as npt

from engibench.core import OptiStep
from engibench.core import Problem

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    import numpy.typing as npt


@dataclass
class Args:
    """Collection of arguments passed to `Problem()`, `Problem.simulate()` and `DesignType()`."""

    problem_args: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments to be passed to :class:`engibench.core.Problem()`."""
    simulate_args: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments to be passed to :meth:`engibench.core.Problem.simulate()`."""
    optimize_args: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments to be passed to :meth:`engibench.core.Problem.optimize()`."""
    design_args: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments to be passed to `DesignType()` or
    the `design_factory` argument of :func:`submit`."""


def merge_args(a: Args, b: Args) -> Args:
    """Merge arguments from `a` with `b`."""
    return Args(
        problem_args={**a.problem_args, **b.problem_args},
        simulate_args={**a.simulate_args, **b.simulate_args},
        design_args={**a.design_args, **b.design_args},
    )


DesignType = TypeVar("DesignType")


@dataclass
class Job(Generic[DesignType]):
    """Representation of a single slurm job."""

    job_type: str
    problem: Callable[..., Problem[DesignType]]
    design_factory: Callable[..., DesignType] | None
    args: Args

    def serialize(self) -> dict[str, Any]:
        """Serialize a job object for an other python process."""
        return {
            "job_type": self.job_type,
            "problem": serialize_callable(self.problem),
            "args": asdict(self.args),
            "design_factory": serialize_callable(self.design_factory) if self.design_factory is not None else None,
        }

    @classmethod
    def deserialize(cls, serialized_job: dict[str, Any]) -> Job:
        """Deserialize a job object from an other python process."""
        design_factory = serialized_job["design_factory"]
        return cls(
            job_type=serialized_job["job_type"],
            problem=deserialize_callable(serialized_job["problem"]),
            args=Args(**serialized_job["args"]),
            design_factory=deserialize_callable(design_factory) if design_factory is not None else None,
        )

    def run(self) -> tuple[DesignType, list[OptiStep]] | npt.NDArray[Any] | Any:
        """Run the optimization defined by the job."""
        problem = self.problem(config=self.args.problem_args)
        design = self.args.design_args.get("design", None)
        if self.job_type == "simulate":
            return problem.simulate(design=design, config=self.args.simulate_args)
        if self.job_type == "optimize":
            return problem.optimize(starting_point=design, config=self.args.optimize_args)
        if self.job_type == "render":
            return problem.render(design=design, config=self.args.simulate_args)  # type: ignore  # noqa: PGH003
        msg = f"Unknown job type: {self.job_type}"
        raise ValueError(msg)


def design_type(t: type[Problem] | Callable[..., Problem]) -> type[Any]:
    """Deduce the design type corresponding to the given `Problem` type."""
    if not isinstance(t, type):
        msg = f"Could not deduce the design type corresponding to `{t.__name__}`: The object is not a type"
        raise TypeError(msg) from None
    if not issubclass(t, Problem):
        msg = f"Could not deduce the design type corresponding to `{t.__name__}`: The object is not a Problem type"
        raise TypeError(msg) from None
    try:
        (design_type,) = t.__orig_bases__[0].__args__  # type: ignore[attr-defined]
    except AttributeError:
        msg = f"Could not deduce the design type corresponding to `{t.__name__}`: The Problem class does not specify its type for its design"
        raise ValueError(msg) from None
    return design_type


SerializedType = tuple[str, str, str]


def serialize_callable(t: Callable[..., Any] | type[Any]) -> SerializedType:
    """Serialize a callable (problem type supported) so it can be imported by a different python process."""
    top_level_module, _ = t.__module__.split(".", 1)
    path = sys.modules[top_level_module].__file__
    if path is None:
        msg = "Got a module without path"
        raise RuntimeError(msg)
    if os.path.basename(path) == "__init__.py":
        path = os.path.dirname(path)
    path = os.path.dirname(path)
    return (path, t.__module__, t.__name__)


def deserialize_callable(serialized_type: SerializedType) -> Callable[..., Any] | type[Any]:
    """Deserialize information on how to load a callable serialized by a different python process."""
    path, module_name, problem_name = serialized_type
    sys.path.append(path)
    module = importlib.import_module(module_name)
    return getattr(module, problem_name)


@dataclass
class SlurmConfig:
    """Collection of slurm parameters passed to sbatch."""

    sbatch_executable: str = "sbatch"
    """Path to the sbatch executable if not in PATH"""
    log_dir: str | None = None
    """Path of the log directory"""
    name: str | None = None
    """Optional name for the jobs"""
    account: str | None = None
    """Slurm account to use"""
    runtime: str | None = None
    """Optional runtime in the format ``hh:mm:ss``. """
    constraint: str | None = None
    """Optional constraint"""
    mem_per_cpu: str | None = None
    """E.g. "4G"."""
    mem: str | None = None
    """E.g. "4G"."""
    nodes: int | None = None
    ntasks: int | None = None
    cpus_per_task: int | None = None
    extra_args: Sequence[str] = ()
    """Extra arguments passed to sbatch."""


def submit(
    job_type: str,
    problem: type[Problem],
    parameter_space: list[Args],
    design_factory: Callable[..., DesignType] | None = None,
    config: SlurmConfig | None = None,
) -> None:
    """Submit a job array for a parameter discovery to slurm.

    - :attr:`job_type` - The type of the job to be submitted: 'simulate', 'optimize', or 'render'.
    - :attr:`problem` - The problem type for which the simulation should be run.
    - :attr:`parameter_space` - One :class:`Args` instance per simulation run to be submitted.
    - :attr:`design_factory` - If not None, pass `Args.design_args` to `design_factory` instead of `DesignType()`.
    -  :attr:`design_factory` - Custom arguments passed to `sbatch`.
    """
    if config is None:
        config = SlurmConfig()

    log_file = os.path.join(config.log_dir, "%j.log") if config.log_dir is not None else None
    if config.log_dir is not None:
        os.makedirs(config.log_dir, exist_ok=True)

    # Dump parameter space:
    param_dir = tempfile.mkdtemp(dir=os.environ.get("SCRATCH"))
    for job_no, args in enumerate(parameter_space, start=1):
        job = Job(job_type, problem=problem, design_factory=design_factory, args=args)
        dump_job(job, param_dir, job_no)

    optional_args = (
        ("--output", log_file),
        ("--comment", config.name),
        ("--time", config.runtime),
        ("--constraint", config.constraint),
        ("--mem-per-cpu", config.mem_per_cpu),
        ("--mem", config.mem),
        ("--nodes", config.nodes),
        ("--ntasks", config.ntasks),
        ("--cpus-per-task", config.cpus_per_task),
    )
    cmd = [
        config.sbatch_executable,
        "--parsable",
        "--export=ALL",
        f"--array=1-{len(parameter_space)}%1000",
        *(f"{arg}={value}" for arg, value in optional_args if value is not None),
        *config.extra_args,
        "--wrap",
        f"{sys.executable} {__file__} run {param_dir}",
    ]

    job_id = run_sbatch(cmd)
    cleanup_cmd = [
        config.sbatch_executable,
        "--parsable",
        f"--dependency=afterany:{job_id}",
        "--export=ALL",
        "--wait",
        "--wrap",
        f"{sys.executable} {__file__} cleanup {param_dir}",
    ]
    run_sbatch(cleanup_cmd)


def dump_job(job: Job, folder: str, index: int) -> None:
    """Dump a job object corresponding to the item of a slurm job array with specified index to disk."""
    parameter_file = os.path.join(folder, f"parameter_space_{index}.pkl")
    with open(parameter_file, "wb") as stream:
        pickle.dump(job.serialize(), stream)


def load_job(folder: str, index: int) -> Job:
    """Load a job object corresponding to the item of a slurm job array with specified index from disk."""
    parameter_file = os.path.join(folder, f"parameter_space_{index}.pkl")
    with open(parameter_file, "rb") as stream:
        return Job.deserialize(pickle.load(stream))


def load_job_args(folder: str) -> Iterable[tuple[int, dict[str, Any]]]:
    """Load the enumerated argument parts of all jobs of a slurm job array from disk."""
    for index in itertools.count(1):
        parameter_file = os.path.join(folder, f"parameter_space_{index}.pkl")
        try:
            with open(parameter_file, "rb") as stream:
                yield index, pickle.load(stream)["args"]
        except FileNotFoundError:
            break


def run_sbatch(cmd: list[str]) -> str:
    """Execute sbatch with the given arguments, returning the job id of the submitted job."""
    try:
        proc = subprocess.run(cmd, shell=False, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        msg = f"sbatch job submission failed: {e.stderr.decode()}"
        raise RuntimeError(msg) from e
    return proc.stdout.decode().strip()


def slurm_job_entrypoint() -> None:
    """Entrypoint of a single slurm job.

    The "run" mode is for the job array items which run the simulation:
    ```sh
    python slurm.py run <work_dir>
    ```
    this mode will read from the environment variable `SLURM_ARRAY_TASK_ID` and will load the corresponding simulation parameters.
    The "cleanup" mode combines the results of all simulations to one file.
    ```sh
    python slurm.py cleanup <work_dir>
    ```
    """

    def run(work_dir: str) -> None:
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        job = load_job(work_dir, index)
        results = job.run()
        result_file = os.path.join(work_dir, f"{index}.pkl")
        with open(result_file, "wb") as stream:
            pickle.dump(results, stream)

    def cleanup(work_dir: str) -> None:
        results = []
        for index, result_args in load_job_args(work_dir):
            result_file = os.path.join(work_dir, f"{index}.pkl")
            if not os.path.exists(result_file):
                print(f"Warning: Result file {result_file} does not exist. Skipping.")
                continue
            try:
                with open(result_file, "rb") as stream:
                    result = pickle.load(stream)
                results.append({"results": result, **result_args})
            except Exception as e:  # noqa: BLE001
                print(f"Error loading {result_file}: {e}. Skipping.")
                continue

        print(os.getcwd())
        with open("results.pkl", "wb") as stream:
            pickle.dump(results, stream)
        shutil.rmtree(work_dir)

    modes = {f.__name__: f for f in (run, cleanup)}
    parser = ArgumentParser()
    parser.add_argument("mode", choices=list(modes.keys()), help="either run or cleanup")
    parser.add_argument("work_dir", help="Path to the work directory")
    args = parser.parse_args()
    mode = modes[args.mode]
    mode(work_dir=args.work_dir)


if __name__ == "__main__":
    slurm_job_entrypoint()
