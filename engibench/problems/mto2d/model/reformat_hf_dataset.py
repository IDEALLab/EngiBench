"""Reformat the pinned raw IDEALLab/MTO-2D NumPy files for EngiBench.

The command memory-maps the 1.49 GB design array and streams one reconstructed
400x200 design at a time. It does not upload or push anything.

Examples:
    Convert already-downloaded files::

        python -m engibench.problems.mto2d.model.reformat_hf_dataset \
            --raw-dir /path/to/raw --output-dir ./mto2d_dataset

    Download the pinned revision into the Hugging Face cache, then convert::

        python -m engibench.problems.mto2d.model.reformat_hf_dataset \
            --output-dir ./mto2d_dataset
"""

import argparse
from pathlib import Path

from engibench.problems.mto2d.model.dataset import convert_raw_arrays
from engibench.problems.mto2d.model.dataset import download_raw_files
from engibench.problems.mto2d.model.dataset import raw_file_paths
from engibench.problems.mto2d.model.dataset import RAW_REPOSITORY
from engibench.problems.mto2d.model.dataset import RAW_REVISION


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--raw-dir", type=Path, help="directory containing the five raw .npy files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default=RAW_REPOSITORY)
    parser.add_argument("--revision", default=RAW_REVISION)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the raw dataset conversion command line interface."""
    args = _parser().parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")

    if args.raw_dir is None:
        print(
            f"Downloading raw files from {args.repo_id}@{args.revision}. "
            "The published design array alone is approximately 1.49 GB."
        )
        paths = download_raw_files(repo_id=args.repo_id, revision=args.revision, cache_dir=args.cache_dir)
    else:
        paths = raw_file_paths(args.raw_dir)

    dataset = convert_raw_arrays(
        paths,
        seed=args.seed,
        cache_dir=args.cache_dir,
        source_dataset=args.repo_id,
        source_revision=args.revision,
    )
    dataset.save_to_disk(str(output))
    sizes = ", ".join(f"{name}={len(split):,}" for name, split in dataset.items())
    print(f"Saved {sizes} to {output}. No data was uploaded.")


if __name__ == "__main__":
    main()
