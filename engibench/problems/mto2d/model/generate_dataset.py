"""Generate and assemble a gridded MTO2D dataset.

Examples:
    Preview the exact 10,000-case default grid without launching a solver::

        python -m engibench.problems.mto2d.model.generate_dataset generate \
            --output-dir ./mto2d_shards --dry-run

    Resume local generation using an external solver configuration::

        python -m engibench.problems.mto2d.model.generate_dataset generate \
            --output-dir ./mto2d_shards --solver-config solver.json

    Assemble all 10,000 shards into train/val/test Arrow datasets::

        python -m engibench.problems.mto2d.model.generate_dataset assemble \
            --shard-dir ./mto2d_shards --output-dir ./mto2d_dataset
"""

import argparse
from pathlib import Path
from typing import Any

from engibench.problems.mto2d.model.dataset import assemble_shards
from engibench.problems.mto2d.model.dataset import condition_grid
from engibench.problems.mto2d.model.dataset import DEFAULT_GRID_SHAPE
from engibench.problems.mto2d.model.dataset import DEFAULT_WRITER_BATCH_SIZE
from engibench.problems.mto2d.model.dataset import generate_local
from engibench.problems.mto2d.model.dataset import generation_jobs
from engibench.problems.mto2d.model.dataset import load_solver_config
from engibench.problems.mto2d.model.dataset import submit_slurm


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="optimize grid points into resumable NPZ shards")
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--solver-config", type=Path, help="JSON object passed to MTO2D.Config")
    generate.add_argument("--start-index", type=int, default=0)
    generate.add_argument("--stop-index", type=int)
    generate.add_argument(
        "--evaluate-final",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run frozen simulate() on each optimized design for exact final objectives",
    )
    generate.add_argument("--force", action="store_true", help="replace existing valid shards")
    generate.add_argument("--dry-run", action="store_true", help="print selected grid bounds without running jobs")
    generate.add_argument("--slurm", action="store_true", help="submit through EngiBench sbatch_map")
    generate.add_argument("--group-size", type=int, default=1)
    generate.add_argument("--max-array-size", type=int, default=1_000)
    generate.add_argument("--slurm-work-dir", type=Path)
    generate.add_argument("--wait", action="store_true", help="wait for the submitted SLURM array")
    generate.add_argument("--account")
    generate.add_argument("--runtime", default="12:00:00")
    generate.add_argument("--log-dir", type=Path)
    generate.add_argument("--mem")
    generate.add_argument("--nodes", type=int, default=1)
    generate.add_argument("--ntasks", type=int)
    generate.add_argument("--cpus-per-task", type=int)
    generate.add_argument(
        "--sbatch-extra-arg",
        action="append",
        default=[],
        help="extra sbatch argument; repeat this option as needed",
    )

    assemble = subparsers.add_parser("assemble", help="stream completed shards into a DatasetDict")
    assemble.add_argument("--shard-dir", type=Path, required=True)
    assemble.add_argument("--output-dir", type=Path, required=True)
    assemble.add_argument(
        "--expected-count",
        type=int,
        default=10_000,
        help="required contiguous case count",
    )
    assemble.add_argument(
        "--allow-partial", action="store_true", help="assemble every discovered shard without requiring IDs"
    )
    assemble.add_argument("--seed", type=int, default=1)
    assemble.add_argument("--cache-dir", type=Path)
    assemble.add_argument("--writer-batch-size", type=int, default=DEFAULT_WRITER_BATCH_SIZE)
    return parser


def _run_generate(args: argparse.Namespace) -> None:
    grid = condition_grid()
    solver_config = load_solver_config(args.solver_config)
    jobs = generation_jobs(
        args.output_dir,
        solver_config=solver_config,
        grid=grid,
        start_index=args.start_index,
        stop_index=args.stop_index,
        evaluate_final=args.evaluate_final,
        force=args.force,
    )
    if not jobs:
        print("No grid points selected.")
        return

    first = jobs[0]
    last = jobs[-1]
    print(
        f"Selected {len(jobs):,} of {len(grid):,} grid points "
        f"(shape={DEFAULT_GRID_SHAPE}, IDs {first['case_id']}..{last['case_id']})."
    )
    print(
        "First conditions: "
        f"u={first['inlet_velocity']:.6g}, D1={first['max_power_dissipation']:.6g}, "
        f"volume={first['volume_fraction']:.6g}"
    )
    print(
        "Last conditions: "
        f"u={last['inlet_velocity']:.6g}, D1={last['max_power_dissipation']:.6g}, "
        f"volume={last['volume_fraction']:.6g}"
    )
    if args.dry_run:
        return

    if args.slurm:
        from engibench.utils.slurm import SlurmConfig  # noqa: PLC0415

        slurm_config = SlurmConfig(
            name="mto2d-grid",
            account=args.account,
            runtime=args.runtime,
            log_dir=None if args.log_dir is None else str(args.log_dir.expanduser().resolve()),
            mem=args.mem,
            nodes=args.nodes,
            ntasks=args.ntasks or int(solver_config.get("mpi_cores", 1)),
            cpus_per_task=args.cpus_per_task or 1,
            extra_args=tuple(args.sbatch_extra_arg),
        )
        submitted_batches = submit_slurm(
            jobs,
            slurm_config=slurm_config,
            group_size=args.group_size,
            max_array_size=args.max_array_size,
            work_dir=args.slurm_work_dir,
            wait=args.wait,
        )
        job_ids = ", ".join(batch.job_id for batch in submitted_batches)
        print(
            f"Submitted {len(submitted_batches)} SLURM array batch(es): {job_ids}. "
            f"Shards are written under {args.output_dir.resolve()}."
        )
        return

    paths = generate_local(jobs)
    print(f"Finished {len(paths):,} cases; shards are under {args.output_dir.resolve()}.")


def _run_assemble(args: argparse.Namespace) -> None:
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    dataset = assemble_shards(
        args.shard_dir,
        expected_count=None if args.allow_partial else args.expected_count,
        seed=args.seed,
        cache_dir=args.cache_dir,
        writer_batch_size=args.writer_batch_size,
    )
    dataset.save_to_disk(str(output))
    sizes = ", ".join(f"{name}={len(split):,}" for name, split in dataset.items())
    print(f"Saved {sizes} to {output}.")


def main(argv: list[str] | None = None) -> None:
    """Run the gridded generation command line interface."""
    args = _parser().parse_args(argv)
    handlers: dict[str, Any] = {"generate": _run_generate, "assemble": _run_assemble}
    handlers[args.command](args)


if __name__ == "__main__":
    main()
