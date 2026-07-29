"""Reformat the pinned raw IDEALLab/MTO-2D NumPy files for EngiBench.

The command memory-maps the 1.49 GB design array and streams one reconstructed
400x200 design at a time. Uploading is opt-in with ``--push-to-hub``.

Examples:
    Convert already-downloaded files::

        python -m engibench.problems.mto2d.model.reformat_hf_dataset \
            --raw-dir /path/to/raw --output-dir ./mto2d_dataset

    Download the pinned revision into the Hugging Face cache, then convert::

        python -m engibench.problems.mto2d.model.reformat_hf_dataset \
            --output-dir ./mto2d_dataset
"""

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from engibench.problems.mto2d.model.dataset import convert_raw_arrays
from engibench.problems.mto2d.model.dataset import DEFAULT_SPLIT_SEED
from engibench.problems.mto2d.model.dataset import DEFAULT_WRITER_BATCH_SIZE
from engibench.problems.mto2d.model.dataset import download_raw_files
from engibench.problems.mto2d.model.dataset import LEGACY_SPLIT_ALGORITHM
from engibench.problems.mto2d.model.dataset import LEGACY_SPLIT_FRACTIONS
from engibench.problems.mto2d.model.dataset import legacy_split_indices
from engibench.problems.mto2d.model.dataset import LEGACY_SPLIT_POLICY
from engibench.problems.mto2d.model.dataset import raw_file_paths
from engibench.problems.mto2d.model.dataset import RAW_REPOSITORY
from engibench.problems.mto2d.model.dataset import RAW_REVISION
from engibench.problems.mto2d.model.dataset import RAW_SHA256
from engibench.problems.mto2d.model.dataset import validate_legacy_dataset
from engibench.problems.mto2d.model.design_io import HALF_DESIGN_SHAPE


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--raw-dir", type=Path, help="directory containing the five raw .npy files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default=RAW_REPOSITORY)
    parser.add_argument("--revision", default=RAW_REVISION)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="paper-compatible PyTorch random_split seed (default: 1)",
    )
    parser.add_argument("--writer-batch-size", type=int, default=DEFAULT_WRITER_BATCH_SIZE)
    parser.add_argument(
        "--verify-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="verify local files against the pinned Hugging Face LFS SHA-256 values",
    )
    parser.add_argument("--push-to-hub", metavar="REPO_ID", help="optionally publish the validated DatasetDict")
    parser.add_argument("--private", action="store_true", help="create a private Hub dataset when publishing")
    parser.add_argument("--max-shard-size", default="500MB")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the raw dataset conversion command line interface."""
    args = _parser().parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        if not args.push_to_hub:
            raise FileExistsError(f"output directory already exists: {output}")
        if args.raw_dir is not None:
            raise ValueError("--raw-dir cannot be combined with publishing an existing output directory")
        _publish_existing(output, args)
        return

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
        writer_batch_size=args.writer_batch_size,
        verify_hashes=args.verify_hashes,
    )
    row_count = sum(len(split) for split in dataset.values())
    sizes = validate_legacy_dataset(
        dataset,
        row_count=row_count,
        seed=args.seed,
        source_dataset=args.repo_id,
        source_revision=args.revision,
    )
    dataset.save_to_disk(str(output))

    from datasets import load_from_disk  # noqa: PLC0415

    reloaded = load_from_disk(str(output))
    validate_legacy_dataset(
        reloaded,
        row_count=row_count,
        seed=args.seed,
        source_dataset=args.repo_id,
        source_revision=args.revision,
    )
    manifest = {
        "schema": "engibench-mto2d-v0-beams3d-compatible-flat-design",
        "source_dataset": args.repo_id,
        "source_revision": args.revision,
        "source_sha256": RAW_SHA256 if args.repo_id == RAW_REPOSITORY and args.revision == RAW_REVISION else None,
        "source_hashes_verified": bool(
            args.verify_hashes and args.repo_id == RAW_REPOSITORY and args.revision == RAW_REVISION
        ),
        "row_count": row_count,
        "native_design_shape": list(HALF_DESIGN_SHAPE),
        "stored_design_shape": [int(HALF_DESIGN_SHAPE[0] * HALF_DESIGN_SHAPE[1])],
        "stored_design_dtype": "float32",
        "stored_design_order": "C",
        "condition_dtype": "float64",
        "objective_dtype": "float32",
        "objective_semantics": (
            "published legacy cold-start source labels; not evaluated on the "
            "reconstructed design; default simulation uses source-matched q=0.01 physics"
        ),
        "split_policy": LEGACY_SPLIT_POLICY,
        "split_fractions": list(LEGACY_SPLIT_FRACTIONS),
        "split_seed": args.seed,
        "split_algorithm": LEGACY_SPLIT_ALGORITHM,
        "split_sizes": sizes,
        "transform": (
            "legacy 256x256 whole left half -> mirror to 256x512 -> "
            "PyTorch-compatible bicubic 400x400 -> left 400x200 -> flatten"
        ),
        "design_is_exact": False,
        "objectives_evaluated_on_design": False,
    }
    (output / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_dataset_card(output, manifest)
    size_text = ", ".join(f"{name}={size:,}" for name, size in sizes.items())
    print(f"Saved and revalidated {size_text} at {output}.")

    if args.push_to_hub:
        _publish_dataset(reloaded, output, args)
    else:
        print(
            "No data was uploaded. Push later with:\n"
            "  python -m engibench.problems.mto2d.model.reformat_hf_dataset \\\n"
            f"    --output-dir {output} --push-to-hub IDEALLab/mto_2d_v0"
        )


def _publish_existing(output: Path, args: argparse.Namespace) -> None:
    from datasets import load_from_disk  # noqa: PLC0415

    manifest = _load_manifest(output)
    row_count, seed, source_dataset, source_revision = _manifest_validation_settings(manifest)
    dataset = load_from_disk(str(output))
    sizes = validate_legacy_dataset(
        dataset,
        row_count=row_count,
        seed=seed,
        source_dataset=source_dataset,
        source_revision=source_revision,
    )
    if sizes != manifest["split_sizes"]:
        raise ValueError("conversion manifest split sizes do not match the saved dataset")
    _write_dataset_card(output, manifest)
    size_text = ", ".join(f"{name}={size:,}" for name, size in sizes.items())
    print(
        f"Reloaded and revalidated {size_text} at {output} "
        f"using its manifest ({source_dataset}@{source_revision}, seed={seed})."
    )
    _publish_dataset(dataset, output, args)


def _publish_dataset(dataset: Any, output: Path, args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi  # noqa: PLC0415

    api = HfApi()
    api.create_repo(
        repo_id=args.push_to_hub,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    api.upload_file(
        path_or_fileobj=output / "README.md",
        path_in_repo="README.md",
        repo_id=args.push_to_hub,
        repo_type="dataset",
        commit_message="Add MTO2D dataset card",
    )
    dataset.push_to_hub(
        args.push_to_hub,
        private=args.private,
        max_shard_size=args.max_shard_size,
        commit_message="Add EngiBench-compatible MTO2D v0 dataset",
    )

    api.upload_file(
        path_or_fileobj=output / "conversion_manifest.json",
        path_in_repo="conversion_manifest.json",
        repo_id=args.push_to_hub,
        repo_type="dataset",
        commit_message="Add MTO2D conversion manifest",
    )
    print(f"Published data, dataset card, and manifest to https://huggingface.co/datasets/{args.push_to_hub}")


def _load_manifest(output: Path) -> dict[str, Any]:
    manifest_path = output / "conversion_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"converted dataset manifest is missing: {manifest_path}")
    parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError(f"conversion manifest must contain a JSON object: {manifest_path}")
    return parsed


def _manifest_validation_settings(manifest: Mapping[str, Any]) -> tuple[int, int, str, str]:
    if manifest.get("schema") != "engibench-mto2d-v0-beams3d-compatible-flat-design":
        raise ValueError("conversion manifest has an unsupported schema")
    if manifest.get("native_design_shape") != list(HALF_DESIGN_SHAPE):
        raise ValueError("conversion manifest has an unexpected native design shape")
    if manifest.get("stored_design_shape") != [int(HALF_DESIGN_SHAPE[0] * HALF_DESIGN_SHAPE[1])]:
        raise ValueError("conversion manifest has an unexpected stored design shape")
    if manifest.get("split_policy") != LEGACY_SPLIT_POLICY:
        raise ValueError(
            "conversion manifest does not use the paper-compatible legacy split policy; regenerate the converted dataset"
        )
    if manifest.get("split_fractions") != list(LEGACY_SPLIT_FRACTIONS):
        raise ValueError("conversion manifest has unexpected legacy split fractions")

    split_sizes = manifest.get("split_sizes")
    if not isinstance(split_sizes, dict) or set(split_sizes) != {"train", "val", "test"}:
        raise ValueError("conversion manifest must contain train/val/test split sizes")
    try:
        row_count = int(manifest.get("row_count", sum(int(value) for value in split_sizes.values())))
        seed = int(manifest["split_seed"])
        source_dataset = str(manifest["source_dataset"])
        source_revision = str(manifest["source_revision"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("conversion manifest contains invalid validation settings") from error
    if row_count != sum(int(value) for value in split_sizes.values()):
        raise ValueError("conversion manifest row count does not equal its split sizes")
    expected_sizes = {name: len(indices) for name, indices in legacy_split_indices(row_count, seed=seed).items()}
    if {name: int(value) for name, value in split_sizes.items()} != expected_sizes:
        raise ValueError("conversion manifest split sizes do not match the paper-compatible legacy split policy")
    return row_count, seed, source_dataset, source_revision


def _write_dataset_card(output: Path, manifest: Mapping[str, Any]) -> None:
    sizes = manifest["split_sizes"]
    split_seed = manifest["split_seed"]
    source_dataset = manifest["source_dataset"]
    source_revision = manifest["source_revision"]
    tolerance_note = ""
    if source_dataset == RAW_REPOSITORY and source_revision == RAW_REVISION:
        tolerance_note = """
At this pinned revision, 3,149 of 5,666 published power labels are slightly
above their listed limit, but the largest excess is only `0.4997` normalized
units (`0.976%`). Preserve the exact residual and explicitly state the
tolerance used for any derived feasibility flag.
"""
    card = f"""---
license: mit
pretty_name: MTO2D v0
tags:
- topology-optimization
- heat-transfer
- computational-fluid-dynamics
---

# MTO2D v0

EngiBench-compatible conversion of
[{source_dataset}](https://huggingface.co/datasets/{source_dataset}) at revision
`{source_revision}`.

- Splits: train {sizes["train"]:,}, validation {sizes["val"]:,}, test {sizes["test"]:,}.
- Split policy: paper/VQGAN-TO 75/5/20 with PyTorch CPU
  `random_split`-compatible membership and order (seed `{split_seed}`).
- `optimal_design` is a flat C-order `list<float32>` of length 80,000 and reshapes to `(400, 200)`.
- Conditions are `float64`; mean temperature and power dissipation are `float32`.

## Important reconstruction warning

The source stores lossy `256 x 256` images. This conversion mirrors each source
half, applies PyTorch-compatible non-antialiased bicubic resizing, takes the
native `(400, 200)` left half, and flattens it. Stored objective values belong
to the original solver-native topology, **not** the reconstructed design.
They also describe the source solver's pre-update field, whereas the stored
topology is post-update. EngiBench `simulate()` defaults to the source-matched
final `q=0.01` material interpolation, but re-simulating the lossy
reconstruction is still expected to produce different values.

The fields `design_is_exact = false` and
`objectives_evaluated_on_design = false` make this distinction
machine-readable. Treat the objective columns as preserved source labels, not
fresh evaluations of `optimal_design`.

{tolerance_note}
See `conversion_manifest.json` for hashes, split policy, exact provenance, and
representation details.

## Citation

*To Quantize or Not to Quantize: Effects on Generative Models for Topology
Optimization Problems*. https://doi.org/10.1115/1.4071440
"""
    (output / "README.md").write_text(card, encoding="utf-8")


if __name__ == "__main__":
    main()
