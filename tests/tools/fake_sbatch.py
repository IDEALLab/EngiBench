#!/bin/env python3

import argparse
import shlex
import subprocess


def parse_array_range(s: str) -> slice:
    """Parse a string like 1-3."""
    start, stop = s.split("-")
    return slice(int(start), int(stop))


def parse_cmd(s: str) -> list[str]:
    return shlex.split(s)


def main() -> None:
    """Serial, local version of sbatch only considering --wrap and --array."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrap", type=parse_cmd, required=True)
    parser.add_argument("--array", type=parse_array_range, default=None)
    args, _ = parser.parse_known_args()
    if args.array is not None:
        for index in range(args.array.start, args.array.stop + 1):
            subprocess.run(args.wrap, check=True, env={"SLURM_ARRAY_TASK_ID": str(index)})
    else:
        subprocess.run(args.wrap, check=True)
    # Print a fake slurm job id:
    print(0)


if __name__ == "__main__":
    main()
