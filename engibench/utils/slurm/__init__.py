"""Slurm executor for parameter space discovery."""

from collections.abc import Callable, Iterable, Sequence
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
from traceback import StackSummary
from traceback import TracebackException
from typing import Any, Generic, TypeVar

import __main__


class JobError(Exception):
    """User error happening during execution of a slurm job.

    - :attr:`origin` - Original exception instance.
    - :attr:`context` - Info (string) about which step failed (e.g. map, reduce or save).
    - :attr:`job_args` - dict containing the arguments passed to the job callback if the exception occurred during a job.
    - :attr:`traceback` - `TracebackException <https://docs.python.org/3/library/traceback.html#traceback.TracebackException>`_ object.
    """

    def __init__(self, origin: Exception, context: str, job_args: dict[str, Any]) -> None:
        self.origin = origin
        self.context = context
        self.job_args = job_args
        self.traceback = TracebackException.from_exception(origin)

    def __str__(self) -> str:
        args = f"\nargs = {self.job_args}" if self.job_args else ""
        return f"""💥 JobError({self.context}):
{self.origin}{args}

{"".join(self.traceback.format())}
"""


def dump_with_job_error(obj: Any, path: str) -> None:
    """Pickle objects to a file which might contain a :py:class:`JobError` instance."""
    with open(path, "wb") as stream:
        pickler = TracebackPickler(stream)
        pickler.dump(obj)


class TracebackPickler(pickle.Pickler):
    """Custom pickler to avoid pickling code objects when pickling tracebacks."""

    def reducer_override(self, obj):
        """Custom reducer for StackSummary."""
        if isinstance(obj, StackSummary):
            return StackSummary.from_list, ([(s.filename, s.lineno, s.name, s.line) for s in obj],)
        # For any other object, fallback to usual reduction
        return NotImplemented


if sys.version_info < (3, 11):

    class ExceptionGroup(Exception):  # noqa: N818
        """Backport of ExceptionGroup added in python 3.11."""

        def __init__(self, msg: str, excs: list[Exception]) -> None:
            self.message = msg
            self.exceptions = excs


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
S = TypeVar("S")

WORKER = os.path.join(os.path.dirname(__file__), "run_job.py")


class SubmittedJobArray(Generic[R]):
    """Representation for a submitted slurm job array."""

    def __init__(self, job_id: str, work_dir: str, n_jobs: int) -> None:
        self.job_id = job_id
        self.work_dir = work_dir
        self.n_jobs = n_jobs

    def reduce(
        self, f_reduce: Callable[[list[R]], S], slurm_args: SlurmConfig | None = None, size_limit: int | None = 10000000
    ) -> S:
        """Reduce the results of a slurm job array.

        The return values of the callable `f` passed to :py:func:`sbatch_map` will be collected into a list and passed as
        the single argument to `f_reduce`.
        After running `f_reduce` as a slurm job, its return value will be passed back and will be the return value of this method.

        To prevent larger workloads running on a login node, this function will raise an exception if the resulting list in pickled
        form takes more than `size_limit` bytes (recommendation: 10MB).
        Only increase / set to 0 if you want to annoy the HPC team 😈.
        - :attr:`f_reduce` - The callable which performs the post processing on the list of return values for each job.
        - :attr:`slurm_args` - Arguments passed to `sbatch`.
        - :attr:`size_limit` - Upper limit for the allowed size of the post processed data in pickled form.
        """
        with open(os.path.join(self.work_dir, "jobs", "reduce.pkl"), "wb") as stream:
            pickle.dump(MemorizeModule(f_reduce), stream)

        # Submit reduce job:
        cmd = " ".join((sys.executable, WORKER, "reduce", self.work_dir, str(self.n_jobs)))
        run_sbatch(cmd, slurm_args=slurm_args or SlurmConfig(), job_dependency=self.job_id, wait=True)

        # Try to load and return the reduced result if it is not too large:
        reduced_path = os.path.join(self.work_dir, "reduced.pkl")
        if size_limit is not None and os.path.getsize(reduced_path) > size_limit:
            raise RuntimeError(f"""Pickled data is too large to be processed by a login node.
Please submit a separate slurm job for postprocessing.
The pickled data is still accessible here: {reduced_path}
""")
        with open(reduced_path, "rb") as stream:
            result = pickle.load(stream)
        shutil.rmtree(self.work_dir)
        if isinstance(result, JobError):
            raise result
        return result

    def save(self, out: str, slurm_args: SlurmConfig | None = None) -> None:
        """Save the collected results of a slurm job array.

        The return values of the callable `f` passed to :py:func:`sbatch_map` will be collected into a list and saved to disk.

        - :attr:`out` - Path to store the pickle archive.
        """
        cmd = " ".join((sys.executable, WORKER, "save", "-o", out, self.work_dir, str(self.n_jobs)))
        run_sbatch(cmd, slurm_args=slurm_args, job_dependency=self.job_id, wait=True)


def sbatch_map(  # noqa: PLR0913
    f: Callable[..., R],
    args: Iterable[dict[str, Any]],
    slurm_args: SlurmConfig | None = None,
    group_size: int = 1,
    work_dir: str | None = None,
    *,
    wait: bool = False,
) -> SubmittedJobArray:
    """Submit a job array for a parameter discovery to slurm.

    The returned :py:class:`SubmittedJobArray` object can be used to
    start a post processing job which will run after all jobs of the array are done.

    - :attr:`f` - The callable which will be applied to each item in `args`.
    - :attr:`args` - Array of keyword arguments which will be passed to `f`.
    - :attr:`slurm_args` - Arguments passed to `sbatch`.
    - :attr:`group_size` - Sequentially process a number of `group_size` jobs in one slurm job.
    - :attr:`wait` - Wait until all job array elements have completed.
      This is useful if neither :py:meth:`SubmittedJobArray.save()` not :py:meth:`SubmittedJobArray.reduce()` is
      required after submitting the job array.

    Details: The individual jobs of the jobarray will be processed in
    individual python instances running the `engibench.utils.slurm.run_job`
    standalone script.
    """
    if work_dir is None:
        work_dir = tempfile.mkdtemp(dir=os.environ.get("SCRATCH"))
    os.makedirs(os.path.join(work_dir, "jobs"), exist_ok=True)
    os.makedirs(os.path.join(work_dir, "results"), exist_ok=True)
    n_jobs = 0
    with open(os.path.join(work_dir, "jobs", "map_callback.pkl"), "wb") as stream:
        pickle.dump(MemorizeModule(f), stream)
    # Dump jobs:
    for job_no, arg in enumerate(args):
        with open(os.path.join(work_dir, "jobs", f"{job_no}.pkl"), "wb") as stream:
            pickle.dump(MemorizeModule(arg), stream)
        n_jobs += 1

    map_cmd = f"{sys.executable} {WORKER} run {work_dir} {n_jobs}"
    job_id = run_sbatch(
        cmd=map_cmd,
        slurm_args=slurm_args or SlurmConfig(),
        array_len=n_jobs // group_size + (1 if n_jobs % group_size else 0),
        wait=wait,
    )
    return SubmittedJobArray(job_id, work_dir, n_jobs)


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
        ("--account", slurm_args.account),
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


def load_results(path: str) -> list[Any]:
    """Load the pickled results produced by :func:`sbatch_map`."""
    with open(path, "rb") as stream:
        return pickle.load(stream)


class MemorizeModule:
    """Wrapper which allows unpickling the wrapped object even when its module is not in PYTHONPATH.

    Use it like `pickle.dumps(MemorizeModule(obj))`.
    The resulting pickle archive will directly unpickle to obj.
    """

    def __init__(self, obj: Any) -> None:
        self.obj = obj

    @staticmethod
    def _reconstruct(reduced_module: tuple[str, ...] | None, pickled_obj: bytes) -> Any:
        mod_path, *modules = reduced_module or (None,)
        if mod_path is not None and mod_path not in sys.path:
            sys.path.append(mod_path)
        if modules:  # obj was pickled from __main__
            (mod_name, obj_name) = modules
            check_main_guard(mod_path, mod_name)
            mod = importlib.import_module(mod_name)
            obj = getattr(mod, obj_name)
            setattr(__main__, obj_name, obj)
        return pickle.loads(pickled_obj)

    def __reduce__(self) -> tuple[Callable[..., Any], tuple[tuple[str, ...] | None, bytes]]:
        pickled_obj = pickle.dumps(self.obj)
        return (self._reconstruct, (module_path(self.obj), pickled_obj))


def module_path(obj: Any) -> tuple[str, ...] | None:
    """Return the path of the toplevel module of the module containing `obj`."""
    if not hasattr(obj, "__module__"):
        return None
    top_level_module = obj.__module__.split(".", 1)[0]
    if top_level_module == "builtins":
        return None
    path = sys.modules[top_level_module].__file__
    if path is None:
        msg = "Got a module without path"
        raise RuntimeError(msg)
    if top_level_module == "__main__":
        return os.path.dirname(path), os.path.basename(path).removesuffix(".py"), obj.__name__
    if os.path.basename(path) == "__init__.py":
        path = os.path.dirname(path)
    return (os.path.dirname(path),)


def check_main_guard(path, mod_name):
    """Check that a python main script as a __main__ guard.

    If this is not the case and pickle tries to load from that file,
    this will run the whole script during unpickling.
    """
    with open(os.path.join(path, mod_name) + ".py") as stream:
        content = stream.read()
        if 'if __name__ == "__main__"' not in content and "if __name__ == '__main__'" not in content:
            raise RuntimeError("Main script does not have a __main__ guard. This will lead to infinite recurson")
