"""Slurm executor for parameter space discovery."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
import itertools
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Generic, TypeVar


class JobError(Exception):
    """User error happening during execution of a slurm job."""

    def __init__(self, origin: Exception, context: str, job_args: dict[str, Any]) -> None:
        self.origin = origin
        self.context = context
        self.job_args = job_args

    def __str__(self) -> str:
        args = f"\nargs = {self.job_args}" if self.job_args else ""
        return f"""💥 JobError({self.context}):
{self.origin}{args}
"""


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


R = TypeVar("R")


def sbatch_map(  # noqa: PLR0913
    f: Callable[..., R],
    args: Iterable[dict[str, Any]],
    slurm_args: SlurmConfig | None = None,
    group_size: int = 1,
    reduce_job: Callable[[list[R]], None] | None = None,
    out: str | None = None,
) -> None:
    """Submit a job array for a parameter discovery to slurm.

    - :attr:`f` - The callable which will be applied to each item in `args`.
    - :attr:`args` - Array of keyword arguments which will be passed to `f`.
    - :attr:`slurm_args` - Arguments passed to `sbatch`.
    - :attr:`group_size` - Sequentially process a number of `group_size` jobs in one slurm job.
    - :attr:`reduce_job` - When all jobs are done, run `reduce_job` as a separate slurm job on the list of all results.
    - :attr:`out` - Output path for a pickle result archive containing a list of the collected return values of `f` or :class:`JobError` instances.

    Details: The individual jobs of the jobarray will be processed in
    individual python instances running the `engibench.utils.slurm.run_job`
    standalone script.
    """
    # Dump jobs:
    work_dir = tempfile.mkdtemp(dir=os.environ.get("SCRATCH"))
    os.makedirs(os.path.join(work_dir, "jobs"))
    os.makedirs(os.path.join(work_dir, "results"))
    n_jobs = 0
    with open(os.path.join(work_dir, "jobs", "f.pkl"), "wb") as stream:
        pickle.dump(MemorizeModule(f), stream)
    for job_no, arg in enumerate(args):
        with open(os.path.join(work_dir, "jobs", f"{job_no}.pkl"), "wb") as stream:
            pickle.dump(MemorizeModule(arg), stream)
        n_jobs += 1

    worker = os.path.join(os.path.dirname(__file__), "run_job.py")

    map_cmd = f"{sys.executable} {worker} run {work_dir} {n_jobs}"
    job_id = run_sbatch(
        cmd=map_cmd,
        slurm_args=slurm_args or SlurmConfig(),
        array_len=n_jobs // group_size + (1 if n_jobs % group_size else 0),
    )
    if reduce_job is not None:
        with open(os.path.join(work_dir, "jobs", "reduce.pkl"), "wb") as stream:
            pickle.dump(MemorizeModule(reduce_job), stream)
    if reduce_job is not None or out is not None:
        out_args = ("-o", out) if out is not None else ()
        reduce_cmd = " ".join((sys.executable, worker, "reduce", *out_args, work_dir, str(n_jobs)))
        run_sbatch(cmd=reduce_cmd, slurm_args=slurm_args or SlurmConfig(), job_dependency=job_id, wait=True)


def run_sbatch(
    cmd: str,
    slurm_args: SlurmConfig | None = None,
    array_len: int | None = None,
    job_dependency: str | None = None,
    *,
    wait: bool = False,
) -> str:
    """Execute sbatch with the given arguments, returning the job id of the submitted job."""
    if slurm_args is None:
        slurm_args = SlurmConfig()
    log_file = os.path.join(slurm_args.log_dir, "%j.log") if slurm_args.log_dir is not None else None
    if slurm_args.log_dir is not None:
        os.makedirs(slurm_args.log_dir, exist_ok=True)

    optional_args = (
        ("--output", log_file),
        ("--comment", slurm_args.name),
        ("--time", slurm_args.runtime),
        ("--constraint", slurm_args.constraint),
        ("--mem-per-cpu", slurm_args.mem_per_cpu),
        ("--mem", slurm_args.mem),
        ("--nodes", slurm_args.nodes),
        ("--ntasks", slurm_args.ntasks),
        ("--cpus-per-task", slurm_args.cpus_per_task),
        ("--array", f"1-{array_len}%1000" if array_len is not None else None),
        ("--dependency", f"afterany:{job_dependency}" if job_dependency is not None else None),
    )
    sbatch_cmd = [
        slurm_args.sbatch_executable,
        "--parsable",
        "--export=ALL",
        *(f"{arg}={value}" for arg, value in optional_args if value is not None),
        *slurm_args.extra_args,
        *(("--wait",) if wait else ()),
        "--wrap",
        cmd,
    ]
    try:
        proc = subprocess.run(sbatch_cmd, shell=False, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        msg = f"sbatch job submission failed: {e.stderr.decode()}"
        raise RuntimeError(msg) from e
    return proc.stdout.decode().strip()


def load_results() -> list[Any]:
    """Load the pickled results produced by :func:`sbatch_map`."""
    with open("results.pkl", "rb") as stream:
        return pickle.load(stream)


def collect_jobs(work_dir: str, n_jobs: int) -> list[Any]:
    """Collect all results of a slurm job array into a list."""

    def load_result(path: str) -> Any:
        try:
            with open(path, "rb") as stream:
                return pickle.load(stream)
        except Exception as e:  # noqa: BLE001
            return JobError(e, "Collect job", {})

    results = [load_result(os.path.join(work_dir, "results", f"{index}.pkl")) for index in range(n_jobs)]
    if not any(isinstance(r, JobError) for r in results):
        shutil.rmtree(work_dir)
    return results


class MemorizeModule:
    """Wrapper which allows unpickling the wrapped object even when its module is not in PYTHONPATH.

    Use it like `pickle.dumps(MemorizeModule(obj))`.
    The resulting pickle archive will directly unpickle to obj.
    """

    def __init__(self, obj: Any) -> None:
        self.obj = obj

    @staticmethod
    def _reconstruct(reduced_module: str | None, pickled_obj: bytes) -> Any:
        if reduced_module is not None and reduced_module not in sys.path:
            sys.path.append(reduced_module)
        return pickle.loads(pickled_obj)

    def __reduce__(self) -> tuple[Callable[..., Any], tuple[str | None, bytes]]:
        pickled_obj = pickle.dumps(self.obj)
        return (self._reconstruct, (module_path(self.obj), pickled_obj))


def module_path(obj: Any) -> str | None:
    """Return the path of the toplevel module of the module containing `obj`."""
    if not hasattr(obj, "__module__"):
        return None
    top_level_module, _ = obj.__module__.split(".", 1)
    path = sys.modules[top_level_module].__file__
    if path is None:
        msg = "Got a module without path"
        raise RuntimeError(msg)
    if os.path.basename(path) == "__init__.py":
        path = os.path.dirname(path)
    return os.path.dirname(path)
