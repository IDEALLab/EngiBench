"""Slurm job worker."""

from argparse import ArgumentParser
import os
import pickle
import shutil
import sys
from typing import Any

from engibench.utils.slurm import dump_with_job_error
from engibench.utils.slurm import JobError
from engibench.utils.slurm import MemorizeModule

if sys.version_info < (3, 11):
    from engibench.utils.slurm import ExceptionGroup


def map_job_group(work_dir: str, n_jobs: int) -> None:
    """Process a job or group of job of a slurm job array.

    This is the "map" step of "map - reduce".
    """
    start = int(os.environ["SLURM_ARRAY_TASK_MIN"])
    stop = int(os.environ["SLURM_ARRAY_TASK_MAX"])
    current = int(os.environ["SLURM_ARRAY_TASK_ID"]) - start
    array_size = stop - start + 1
    group_size = n_jobs // array_size + (1 if n_jobs % array_size else 0)
    try:
        with open(os.path.join(work_dir, "jobs", "map_callback.pkl"), "rb") as in_stream:
            map_callback = pickle.load(in_stream)
    except Exception as e:  # noqa: BLE001
        exception = e

        def map_callback(**_kwargs) -> None:
            raise exception

    # Run `group_size` jobs as sub jobs of current job (usecase: many small jobs):
    for index in range(current * group_size, min((current + 1) * group_size, n_jobs)):
        result_path = os.path.join(work_dir, "results", f"{index}.pkl")
        try:
            with open(os.path.join(work_dir, "jobs", f"{index}.pkl"), "rb") as stream:
                args = pickle.load(stream)
        except Exception as e:  # noqa: BLE001
            dump_with_job_error(JobError(e, "Unpickle job array item", {}), result_path)
            continue
        try:
            result = map_callback(**args)
            dump_with_job_error(MemorizeModule(result), result_path)
        except Exception as e:  # noqa: BLE001
            dump_with_job_error(JobError(e, "Run job array item", args), result_path)


def reduce_job_results(work_dir: str, n_jobs: int) -> None:
    """Collect all results or errors from job array jobs, passing to a reduce callback."""
    results = []  # prepare empty list for error, occurring before `results` is assigned a value
    reduced_pkl = os.path.join(work_dir, "reduced.pkl")
    try:
        with open(os.path.join(work_dir, "jobs", "reduce.pkl"), "rb") as in_stream:
            reduce_callback = pickle.load(in_stream)

        results = collect_jobs(work_dir, n_jobs)
        reduced = reduce_callback(results)
        dump_with_job_error(MemorizeModule(reduced), reduced_pkl)
    except Exception as e:  # noqa: BLE001
        errors = [e] + [err for err in results if isinstance(err, Exception)]
        dump_with_job_error(
            JobError(ExceptionGroup("", errors), "reduce", {}) if errors else JobError(e, "reduce", {}), reduced_pkl
        )


def save(work_dir: str, n_jobs: int, out: str) -> None:
    """Collect all results or errors from job array jobs and save as a pickled list to disk."""
    results = collect_jobs(work_dir, n_jobs)

    if not any(isinstance(r, JobError) for r in results):
        shutil.rmtree(work_dir)
    dump_with_job_error(results, out)


def collect_jobs(work_dir: str, n_jobs: int) -> list[Any]:
    """Collect all results of a slurm job array into a list."""

    def load_result(path: str) -> Any:
        try:
            with open(path, "rb") as stream:
                return pickle.load(stream)
        except Exception as e:  # noqa: BLE001
            return JobError(e, "Collect job", {})

    return [load_result(os.path.join(work_dir, "results", f"{index}.pkl")) for index in range(n_jobs)]


def cli() -> None:
    """Entrypoint of a single slurm job.

    The "run" mode is for the job array items which run the simulation:
    ```sh
    python slurm.py run <work_dir>
    ```
    this mode will read from the environment variable `SLURM_ARRAY_TASK_ID` and will load the corresponding simulation parameters.
    The "cleanup" mode combines the results of all simulations to one file.
    ```sh
    python slurm.py reduce <work_dir>
    ```
    """
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="subcmd",
        title="List of sub-commands",
        description="For an overview of action specific parameters, use %(prog)s <SUB-COMMAND> --help",
        help="Sub-command help",
        metavar="<SUB-COMMAND>",
    )
    subparser = subparsers.add_parser("run", help=map_job_group.__doc__)
    subparser.set_defaults(subcmd=map_job_group)
    subparser.add_argument("work_dir", help="Path to the work directory")
    subparser.add_argument("n_jobs", type=int, help="Total number of jobs")
    subparser = subparsers.add_parser("reduce", help=reduce_job_results.__doc__)
    subparser.set_defaults(subcmd=reduce_job_results)
    subparser.add_argument("work_dir", help="Path to the work directory")
    subparser.add_argument("n_jobs", type=int, help="Total number of jobs")
    subparser = subparsers.add_parser("save", help=save.__doc__)
    subparser.set_defaults(subcmd=save)
    subparser.add_argument("work_dir", help="Path to the work directory")
    subparser.add_argument("n_jobs", type=int, help="Total number of jobs")
    subparser.add_argument("-o", dest="out", default=None, help="Output path for the pickle archive containing the results")
    args = vars(parser.parse_args())
    subcmd = args.pop("subcmd")
    subcmd(**args)


if __name__ == "__main__":
    cli()
