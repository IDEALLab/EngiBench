"""Slurm job worker."""

from argparse import ArgumentParser
import os
import pickle
import shutil

from engibench.utils.slurm import collect_jobs
from engibench.utils.slurm import JobError
from engibench.utils.slurm import MemorizeModule


def run(work_dir: str, n_jobs: int) -> None:
    """Process a job or group of job of a slurm job array."""
    start = int(os.environ["SLURM_ARRAY_TASK_MIN"])
    stop = int(os.environ["SLURM_ARRAY_TASK_MAX"])
    current = int(os.environ["SLURM_ARRAY_TASK_ID"]) - start
    array_size = stop - start
    group_size = n_jobs // array_size + (1 if n_jobs % array_size else 0)
    try:
        with open(os.path.join(work_dir, "jobs", "f.pkl"), "rb") as in_stream:
            f = pickle.load(in_stream)
    except Exception as e:  # noqa: BLE001
        exception = e

        def f(**_kwargs) -> None:
            raise exception

    for index in range(current * group_size, max((current + 1) * group_size, n_jobs)):
        result_path = os.path.join(work_dir, "results", f"{index}.pkl")
        try:
            with open(os.path.join(work_dir, "jobs", f"{index}.pkl"), "rb") as stream:
                args = pickle.load(stream)
        except Exception as e:  # noqa: BLE001
            with open(result_path, "wb") as out_stream:
                pickle.dump(JobError(e, "Unpickle job array item", {}), out_stream)
            continue
        try:
            result = f(**args)
            with open(result_path, "wb") as out_stream:
                pickle.dump(MemorizeModule(result), out_stream)
        except Exception as e:  # noqa: BLE001
            with open(result_path, "wb") as out_stream:
                pickle.dump(JobError(e, "Run job array item", args), out_stream)


def reduce(work_dir: str, n_jobs: int) -> None:
    """Collect all results or errors from job array jobs, passing to a reduce callback."""
    with open(os.path.join(work_dir, "jobs", "reduce.pkl"), "rb") as in_stream:
        reduce_job = pickle.load(in_stream)

    results = collect_jobs(work_dir, n_jobs)
    reduced = reduce_job(results)
    with open(os.path.join(work_dir, "reduced.pkl"), "wb") as out_stream:
        pickle.dump(MemorizeModule(reduced), out_stream)


def save(work_dir: str, n_jobs: int, out: str) -> None:
    """Collect all results or errors from job array jobs and save as a pickled list to disk."""
    results = collect_jobs(work_dir, n_jobs)

    if not any(isinstance(r, JobError) for r in results):
        shutil.rmtree(work_dir)
    with open(out, "wb") as out_stream:
        pickle.dump(results, out_stream)


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
    subparser = subparsers.add_parser("run", help=run.__doc__)
    subparser.set_defaults(subcmd=run)
    subparser.add_argument("work_dir", help="Path to the work directory")
    subparser.add_argument("n_jobs", type=int, help="Total number of jobs")
    subparser = subparsers.add_parser("reduce", help=reduce.__doc__)
    subparser.set_defaults(subcmd=reduce)
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
