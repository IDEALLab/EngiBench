"""Dataset generation for the Photonics2D v1 problem via the SLURM API.

This script collects the optimization results and assembles a HuggingFace dataset, so it can
(re)generate ``IDEALLab/photonics_2d_120_120_v1``. Because v1 guarantees
``simulate(optimal_design) == optimization_history[-1]``, the stored objective column is reproducible
by ``problem.simulate`` by construction.

Two-phase workflow (run on an HPC login node with SLURM and a HuggingFace token):

    # 1. Submit the optimization job array and block until the results pickle is written.
    python -m engibench.problems.photonics2d.dataset_slurm_v1 generate \
        --account "$SLURM_JOB_ACCOUNT" --out photonics_v1_results.pkl

    # 2. Assemble the pickle into a DatasetDict and push to the Hub (needs HF auth).
    python -m engibench.problems.photonics2d.dataset_slurm_v1 assemble \
        --results photonics_v1_results.pkl --push

By default the condition grid is taken from the existing v0 dataset (``--from-v0``) so the v1 dataset
covers the same boundary conditions and is directly comparable; pass ``--sample`` to draw a freshly
sampled condition grid instead.
"""

from __future__ import annotations

from argparse import ArgumentParser
from itertools import product
from typing import Any

import numpy as np

from engibench.problems.photonics2d import Photonics2D
from engibench.utils import slurm

DATASET_ID_V0 = "IDEALLab/photonics_2d_120_120_v0"
DATASET_ID_V1 = "IDEALLab/photonics_2d_120_120_v1"
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- SLURM job (map)


def generate_row(  # noqa: PLR0913
    lambda1: float,
    lambda2: float,
    blur_radius: int,
    split: str,
    num_optimization_steps: int = 200,
    noise: float = 0.001,
) -> dict[str, Any]:
    """Optimize one Photonics2D v1 design and return a dataset row.

    This is the per-job callable submitted to SLURM. It must return only picklable, HF-friendly data.
    """
    # Boundary conditions are passed via `config=` (they are Conditions fields), not as ctor kwargs.
    problem = Photonics2D(config={"lambda1": lambda1, "lambda2": lambda2, "blur_radius": blur_radius})
    start_design, _ = problem.random_design(noise=noise)
    optimal_design, history = problem.optimize(
        start_design, config={"num_optimization_steps": num_optimization_steps}
    )
    objective_history = [float(step.obj_values[0]) for step in history]
    return {
        "lambda1": float(lambda1),
        "lambda2": float(lambda2),
        "blur_radius": int(blur_radius),
        "optimal_design": np.asarray(optimal_design, dtype=np.float32),
        "optimization_history": objective_history,
        # v1 contract: this equals problem.simulate(optimal_design)[0].
        "total_overlap": objective_history[-1],
        "split": split,
    }


# --------------------------------------------------------------------------- condition grids


def configs_from_v0() -> list[dict[str, Any]]:
    """Reuse the exact (lambda1, lambda2, blur_radius) conditions + splits from the v0 dataset."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID_V0)
    return [
        {
            "lambda1": float(row["lambda1"]),
            "lambda2": float(row["lambda2"]),
            "blur_radius": int(row["blur_radius"]),
            "split": "val" if split in ("val", "validation") else split,
        }
        for split in ds
        for row in ds[split]
    ]


def configs_sampled(seed: int = 42) -> list[dict[str, Any]]:
    """Draw a fresh condition grid matching the v0 generation ranges (20 x 20 x 5)."""
    rng = np.random.default_rng(seed)
    lambda1 = rng.uniform(low=0.5, high=1.25, size=20)
    lambda2 = rng.uniform(low=0.75, high=1.5, size=20)
    blur_radius = range(5)
    combos = list(product(lambda1, lambda2, blur_radius))
    rng.shuffle(combos)
    n = len(combos)
    # 80 / 10 / 10 split.
    bounds = {"train": (0, int(0.8 * n)), "val": (int(0.8 * n), int(0.9 * n)), "test": (int(0.9 * n), n)}
    configs: list[dict[str, Any]] = []
    for split, (lo, hi) in bounds.items():
        for l1, l2, br in combos[lo:hi]:
            configs.append({"lambda1": float(l1), "lambda2": float(l2), "blur_radius": int(br), "split": split})
    return configs


# --------------------------------------------------------------------------- phases


def submit(args: Any) -> None:
    """Submit the optimization job array and save the collected results to a pickle."""
    configs = configs_from_v0() if args.from_v0 else configs_sampled(seed=args.seed)
    parameter_space = [{**cfg, "num_optimization_steps": args.steps} for cfg in configs]
    print(f"Submitting {len(parameter_space)} Photonics2D v1 optimizations to SLURM...")

    slurm_config = slurm.SlurmConfig(
        name="photonics2d_v1_dataset",
        account=args.account,
        runtime=args.runtime,
        mem_per_cpu=args.mem_per_cpu,
        ntasks=1,
        cpus_per_task=1,
        log_dir="./opt_logs_v1/",
    )
    job_array = slurm.sbatch_map(
        generate_row,
        args=parameter_space,
        slurm_args=slurm_config,
        group_size=args.group_size,
    )
    # Designs are large (120x120 each), so .save() to disk rather than .reduce() on the login node.
    job_array.save(args.out, slurm_args=slurm_config)
    print(f"Results saved to {args.out}. Next: `assemble --results {args.out} --push`.")


def assemble(args: Any) -> None:
    """Load the results pickle, drop failed jobs, build a DatasetDict, and optionally push to the Hub."""
    from datasets import Dataset
    from datasets import DatasetDict

    results = slurm.load_results(args.results)
    errors = [r for r in results if isinstance(r, slurm.JobError)]
    rows = [r for r in results if not isinstance(r, slurm.JobError)]
    print(f"Loaded {len(results)} results ({len(errors)} failed, {len(rows)} usable).")
    if errors:
        print("First failure context:", errors[0].context, errors[0].job_args)

    columns = ["lambda1", "lambda2", "blur_radius", "optimal_design", "optimization_history", "total_overlap"]
    split_dict = {}
    for split in SPLITS:
        split_rows = [r for r in rows if r["split"] == split]
        if not split_rows:
            continue
        split_dict[split] = Dataset.from_dict({col: [r[col] for r in split_rows] for col in columns})
        print(f"  {split}: {len(split_rows)} rows")

    dataset = DatasetDict(split_dict)
    if args.push:
        print(f"Pushing to {DATASET_ID_V1} (requires HF auth)...")
        dataset.push_to_hub(DATASET_ID_V1)
        print("Done.")
    else:
        dataset.save_to_disk(args.save_to)
        print(f"Saved locally to {args.save_to} (pass --push to upload to {DATASET_ID_V1}).")


# --------------------------------------------------------------------------- CLI


def main() -> None:
    """Parse CLI arguments and dispatch to the ``generate`` or ``assemble`` phase."""
    parser = ArgumentParser(description="Generate the Photonics2D v1 dataset via SLURM.")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Submit the SLURM optimization job array and save results.")
    grid = gen.add_mutually_exclusive_group()
    grid.add_argument("--from-v0", dest="from_v0", action="store_true", default=True, help="Reuse v0 conditions (default).")
    grid.add_argument("--sample", dest="from_v0", action="store_false", help="Draw a fresh condition grid.")
    gen.add_argument("--account", default=None, help="SLURM account (omit for fair-share clusters like Euler).")
    gen.add_argument("--runtime", default="00:20:00", help="Per-array-task runtime HH:MM:SS.")
    gen.add_argument("--mem-per-cpu", dest="mem_per_cpu", default="4G", help="Memory per CPU (e.g. 4G).")
    gen.add_argument(
        "--group-size",
        dest="group_size",
        type=int,
        default=2,
        help="Optimizations per SLURM array task (keep ceil(n_jobs/group_size) <= cluster MaxArraySize).",
    )
    gen.add_argument("--steps", type=int, default=200, help="num_optimization_steps.")
    gen.add_argument("--seed", type=int, default=42, help="Seed for --sample grid.")
    gen.add_argument("--out", default="photonics_v1_results.pkl", help="Output pickle path.")
    gen.set_defaults(func=submit)

    asm = sub.add_parser("assemble", help="Build the dataset from results and push to the Hub.")
    asm.add_argument("--results", default="photonics_v1_results.pkl", help="Results pickle from `generate`.")
    asm.add_argument("--push", action="store_true", help="Push to the HuggingFace Hub.")
    asm.add_argument("--save-to", dest="save_to", default="photonics_2d_120_120_v1", help="Local save dir if not pushing.")
    asm.set_defaults(func=assemble)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
