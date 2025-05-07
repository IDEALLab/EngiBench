#!/bin/env python3

import argparse
import shlex
import subprocess


def parse_array_range(s: str) -> tuple[slice, int | None]:
    """Parse a string like 1-3 or 1-3%1000."""
    if "%" in s:
        s, max_jobs_raw = s.split("%", 1)
        max_jobs = int(max_jobs_raw)
    else:
        max_jobs = None
    start, stop = s.split("-")
    return slice(int(start), int(stop)), max_jobs


def parse_cmd(s: str) -> list[str]:
    return shlex.split(s)


def main() -> None:
    """Serial, local version of sbatch only considering --wrap and --array."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrap", type=parse_cmd, required=True)
    parser.add_argument("--array", type=parse_array_range, default=None)
    args, _ = parser.parse_known_args()
    if args.array is not None:
        arr, _max_jobs = args.array
        for index in range(arr.start, arr.stop + 1):
            subprocess.run(
                args.wrap,
                check=True,
                env={
                    "SLURM_ARRAY_TASK_ID": str(index),
                    "SLURM_ARRAY_TASK_MIN": str(arr.start),
                    "SLURM_ARRAY_TASK_MAX": str(arr.stop),
                },
            )
    else:
        subprocess.run(args.wrap, check=True)
    # Print a fake slurm job id:
    print(0)


if __name__ == "__main__":
    main()
