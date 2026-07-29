"""Validate and optionally update the exact-native MTO2D Hub dataset.

The default invocation is a local dry run: it fully validates the saved
DatasetDict and prints the explicit command required to update the existing
Hub repository. Uploading requires both ``--push`` and
``--confirm-redistribution-rights``.

The rich local dataset retains provenance and validation evidence. The public
projection deliberately contains only the columns used by EngiBench. Hub
updates are a single data-only commit, so the existing human-edited README is
preserved byte-for-byte and local JSON/JSONL manifests are never uploaded.

Example:
    Validate without network writes::

        python -m engibench.problems.mto2d.model.publish_native_dataset \
            --dataset-dir dataset_output/mto_2d_exact_source_v0

    Publish after redistribution rights have been confirmed::

        python -m engibench.problems.mto2d.model.publish_native_dataset \
            --dataset-dir dataset_output/mto_2d_exact_source_v0 \
            --repo-id IDEALLab/mto_2d_v0 \
            --confirm-redistribution-rights \
            --push
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shlex
import tempfile
from typing import Any

import numpy as np

from engibench.problems.mto2d.model.dataset import canonicalize_dataset_columns
from engibench.problems.mto2d.model.dataset import dataset_features
from engibench.problems.mto2d.model.dataset import legacy_split_indices
from engibench.problems.mto2d.model.dataset import LEGACY_SPLIT_POLICY
from engibench.problems.mto2d.model.dataset import RAW_REVISION
from engibench.problems.mto2d.model.dataset import RAW_ROW_COUNT
from engibench.problems.mto2d.model.dataset import RAW_SHA256
from engibench.problems.mto2d.model.design_io import DESIGN_CELL_COUNT
from engibench.problems.mto2d.model.reformat_native_gamma_dataset import _sha256_file
from engibench.problems.mto2d.model.reformat_native_gamma_dataset import CONVERSION_MANIFEST_FILENAME
from engibench.problems.mto2d.model.reformat_native_gamma_dataset import METADATA_KEYS
from engibench.problems.mto2d.model.reformat_native_gamma_dataset import NATIVE_SOURCE_PROVENANCE
from engibench.problems.mto2d.model.retrieve_native_gammas import VALIDATION_RECORDS_FILENAME

CANONICAL_REPO_ID = "IDEALLab/mto_2d_v0"
SOURCE_DATASET_ID = "IDEALLab/MTO-2D"
SUPPORTED_SCHEMA = "engibench-mto2d-exact-native-v0"
PUBLIC_COLUMNS = (
    "optimal_design",
    "inlet_velocity",
    "max_power_dissipation",
    "volfrac",
    "mean_temperature",
    "power_dissipation",
)
PUBLIC_SPLITS = ("train", "val", "test")
PUBLIC_SHARD_COUNTS = {"train": 3, "val": 1, "test": 1}
PUBLIC_PARQUET_PATTERN = re.compile(r"^data/(?:train|val|test)-\d{5}-of-\d{5}\.parquet$")
FRONT_MATTER_PART_COUNT = 3
REQUIRED_FINITE_FIELDS = (
    "inlet_velocity",
    "max_power_dissipation",
    "volfrac",
    "mean_temperature",
    "power_dissipation",
    "power_constraint_residual_absolute",
    "power_constraint_residual_relative",
    "volume_constraint_residual",
)


@dataclass(frozen=True)
class ValidatedDataset:
    """A publication candidate that passed all local checks."""

    path: Path
    dataset: Any
    manifest: dict[str, Any]
    split_sizes: dict[str, int]


@dataclass(frozen=True)
class PublicationResult:
    """Result of a data-only Hub update."""

    commit_oid: str
    commit_url: str
    readme_sha256: str
    data_files: tuple[str, ...]


def _load_manifest(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / CONVERSION_MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"exact-native conversion manifest is missing: {path}")
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError(f"conversion manifest must contain a JSON object: {path}")
    return parsed


def _require_manifest_identity(manifest: dict[str, Any]) -> tuple[int, int]:
    expected_metadata_hashes = {key: RAW_SHA256[key] for key in METADATA_KEYS}
    checks = {
        "schema": SUPPORTED_SCHEMA,
        "source_dataset": SOURCE_DATASET_ID,
        "source_revision": RAW_REVISION,
        "row_count": RAW_ROW_COUNT,
        "split_policy": LEGACY_SPLIT_POLICY,
        "native_design_shape": [400, 200],
        "stored_design_length": DESIGN_CELL_COUNT,
        "design_is_exact": True,
        "objectives_evaluated_on_design": False,
        "metadata_hashes_verified": True,
        "metadata_sha256": expected_metadata_hashes,
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise ValueError(f"manifest {key!r} is {manifest.get(key)!r}; expected {expected!r}")
    seed = manifest.get("split_seed")
    if not isinstance(seed, int):
        raise TypeError("manifest split_seed must be an integer")
    return RAW_ROW_COUNT, seed


def _logical_arrow_column(split: Any, name: str) -> Any:
    """Return a logical Arrow column, respecting Dataset row indices."""
    return split.with_format("arrow")[name]


def _validate_design_column(split_name: str, split: Any) -> None:
    import pyarrow.compute as pc  # type: ignore[import-untyped]  # noqa: PLC0415

    designs = _logical_arrow_column(split, "optimal_design")
    for chunk in designs.chunks:
        if chunk.null_count or chunk.values.null_count:
            raise ValueError(f"{split_name} contains null designs or design values")
        length_bounds = pc.min_max(pc.list_value_length(chunk)).as_py()
        if length_bounds != {"min": DESIGN_CELL_COUNT, "max": DESIGN_CELL_COUNT}:
            raise ValueError(f"{split_name} contains a design whose length is not {DESIGN_CELL_COUNT}")
        values = chunk.values
        if not pc.all(pc.is_finite(values)).as_py():
            raise ValueError(f"{split_name} contains non-finite design values")
        bounds = pc.min_max(values).as_py()
        if bounds["min"] < 0.0 or bounds["max"] > 1.0:
            raise ValueError(f"{split_name} contains design values outside [0, 1]")


def _validate_split_content(split_name: str, split: Any) -> None:  # noqa: C901
    import pyarrow.compute as pc  # noqa: PLC0415

    if split.features != dataset_features():
        raise ValueError(f"{split_name} features do not match the MTO2D v0 schema")
    _validate_design_column(split_name, split)
    for field in REQUIRED_FINITE_FIELDS:
        column = _logical_arrow_column(split, field).combine_chunks()
        if column.null_count or not pc.all(pc.is_finite(column)).as_py():
            raise ValueError(f"{split_name}.{field} must contain only finite, non-null values")
    if any(value is not True for value in split["design_is_exact"]):
        raise ValueError(f"{split_name} must mark every design exact")
    if any(value is not False for value in split["objectives_evaluated_on_design"]):
        raise ValueError(f"{split_name} must mark historical objectives unevaluated on the stored design")
    if set(split["source_dataset"]) != {SOURCE_DATASET_ID}:
        raise ValueError(f"{split_name} has unexpected source_dataset values")
    if set(split["source_revision"]) != {RAW_REVISION}:
        raise ValueError(f"{split_name} has unexpected source_revision values")
    if set(split["design_provenance"]) != {NATIVE_SOURCE_PROVENANCE}:
        raise ValueError(f"{split_name} has unexpected design provenance")

    maximum = np.asarray(split["max_power_dissipation"], dtype=np.float64)
    measured = np.asarray(split["power_dissipation"], dtype=np.float64)
    absolute = np.asarray(split["power_constraint_residual_absolute"], dtype=np.float64)
    relative = np.asarray(split["power_constraint_residual_relative"], dtype=np.float64)
    if not np.allclose(absolute, measured - maximum, rtol=1e-6, atol=1e-5):
        raise ValueError(f"{split_name}.power_constraint_residual_absolute is inconsistent")
    if not np.allclose(relative, measured / maximum - 1.0, rtol=1e-6, atol=1e-7):
        raise ValueError(f"{split_name}.power_constraint_residual_relative is inconsistent")


def _validate_splits(dataset: Any, *, row_count: int, seed: int) -> dict[str, int]:
    expected_indices = legacy_split_indices(row_count, seed=seed)
    expected_sizes = {name: len(indices) for name, indices in expected_indices.items()}
    if set(dataset) != set(expected_sizes):
        raise ValueError(f"dataset splits must be {sorted(expected_sizes)}; got {sorted(dataset)}")

    all_source_ids: list[int] = []
    for split_name, positions in expected_indices.items():
        split = dataset[split_name]
        if len(split) != len(positions):
            raise ValueError(f"{split_name} has {len(split)} rows; expected {len(positions)}")
        actual_positions = np.asarray(split["source_row_index"], dtype=np.int64)
        if not np.array_equal(actual_positions, positions):
            raise ValueError(f"{split_name}.source_row_index does not match the paper-compatible split")
        source_ids = np.asarray(split["source_case_id"], dtype=np.int64)
        if len(np.unique(source_ids)) != len(source_ids):
            raise ValueError(f"{split_name} contains duplicate source_case_id values")
        all_source_ids.extend(source_ids.tolist())
        _validate_split_content(split_name, split)

    if len(all_source_ids) != row_count or len(set(all_source_ids)) != row_count:
        raise ValueError("source_case_id values must be unique across the complete dataset")
    return expected_sizes


def validate_publication_dataset(dataset_dir: str | Path) -> ValidatedDataset:
    """Fully validate an exact-native saved DatasetDict without network writes."""
    directory = Path(dataset_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {directory}")
    manifest = _load_manifest(directory)
    row_count, seed = _require_manifest_identity(manifest)

    evidence = directory / VALIDATION_RECORDS_FILENAME
    if not evidence.is_file():
        raise FileNotFoundError(f"gamma validation evidence is missing: {evidence}")
    expected_evidence_hash = manifest.get("gamma_validation_records_sha256")
    if not isinstance(expected_evidence_hash, str) or _sha256_file(evidence) != expected_evidence_hash:
        raise ValueError("gamma validation evidence SHA-256 does not match the conversion manifest")

    from datasets import load_from_disk  # noqa: PLC0415

    dataset = canonicalize_dataset_columns(load_from_disk(str(directory)))
    split_sizes = _validate_splits(dataset, row_count=row_count, seed=seed)
    if manifest.get("split_sizes") != split_sizes:
        raise ValueError("manifest split_sizes do not match the validated dataset")
    return ValidatedDataset(directory, dataset, manifest, split_sizes)


def public_dataset_features() -> Any:
    """Return the minimal public schema shared with EngiBench."""
    from datasets import Features  # noqa: PLC0415
    from datasets import Sequence as DatasetSequence  # noqa: PLC0415
    from datasets import Value  # noqa: PLC0415

    return Features(
        {
            "optimal_design": DatasetSequence(Value("float32")),
            "inlet_velocity": Value("float64"),
            "max_power_dissipation": Value("float64"),
            "volfrac": Value("float64"),
            "mean_temperature": Value("float32"),
            "power_dissipation": Value("float32"),
        }
    )


def public_dataset(candidate: ValidatedDataset) -> Any:
    """Project the validated rich dataset to its six public columns."""
    projected = candidate.dataset.select_columns(list(PUBLIC_COLUMNS))
    expected_features = public_dataset_features()
    if tuple(projected) != PUBLIC_SPLITS:
        raise ValueError(f"public dataset splits must be {PUBLIC_SPLITS}; got {tuple(projected)}")
    for split_name in PUBLIC_SPLITS:
        split = projected[split_name]
        if tuple(split.column_names) != PUBLIC_COLUMNS:
            raise ValueError(f"{split_name} public columns do not match {PUBLIC_COLUMNS}")
        if split.features != expected_features:
            raise ValueError(f"{split_name} public features do not match the minimal EngiBench schema")
        if len(split) != candidate.split_sizes[split_name]:
            raise ValueError(f"{split_name} public row count changed during projection")
    return projected


def _stage_public_parquet(dataset: Any, data_dir: Path) -> tuple[Path, ...]:
    """Write the deterministic 3/1/1 public Parquet layout."""
    import pyarrow.parquet as pq  # type: ignore[import-untyped]  # noqa: PLC0415

    data_dir.mkdir(parents=True, exist_ok=False)
    expected_schema = public_dataset_features().arrow_schema
    staged: list[Path] = []
    for split_name in PUBLIC_SPLITS:
        split = dataset[split_name]
        shard_count = PUBLIC_SHARD_COUNTS[split_name]
        rows_written = 0
        for shard_index in range(shard_count):
            shard = split.shard(num_shards=shard_count, index=shard_index, contiguous=True)
            path = data_dir / f"{split_name}-{shard_index:05d}-of-{shard_count:05d}.parquet"
            shard.to_parquet(path)
            parquet = pq.ParquetFile(path)
            if not parquet.schema_arrow.equals(expected_schema, check_metadata=False):
                raise ValueError(f"staged Parquet schema is not the minimal public schema: {path}")
            rows_written += parquet.metadata.num_rows
            staged.append(path)
        if rows_written != len(split):
            raise ValueError(f"staged {split_name} row count is {rows_written}; expected {len(split)}")
    return tuple(staged)


def _repo_path(path: Path) -> str:
    path_in_repo = f"data/{path.name}"
    if PUBLIC_PARQUET_PATTERN.fullmatch(path_in_repo) is None:
        raise ValueError(f"unexpected public Parquet filename: {path.name}")
    return path_in_repo


def _read_remote_readme(repo_id: str, revision: str) -> bytes:
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    cached_path = hf_hub_download(
        repo_id,
        "README.md",
        repo_type="dataset",
        revision=revision,
        force_download=True,
    )
    return Path(cached_path).read_bytes()


def _require_readme_license(readme: bytes, license_id: str) -> None:
    try:
        text = readme.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("the existing Hub README must be UTF-8") from error
    front_matter = text.split("---", 2)
    if len(front_matter) < FRONT_MATTER_PART_COUNT or f"\nlicense: {license_id}\n" not in f"\n{front_matter[1]}\n":
        raise ValueError(
            f"the existing Hub README does not declare license {license_id!r}; "
            "edit it separately instead of letting the dataset publisher overwrite it"
        )


def _commit_operations(
    staged: tuple[Path, ...],
    existing_files: set[str],
) -> tuple[list[Any], tuple[str, ...]]:
    from huggingface_hub import CommitOperationAdd  # noqa: PLC0415
    from huggingface_hub import CommitOperationDelete  # noqa: PLC0415

    additions = {_repo_path(path): path for path in staged}
    stale = sorted(
        path for path in existing_files if path.startswith("data/") and path.endswith(".parquet") and path not in additions
    )
    operations: list[Any] = [CommitOperationDelete(path_in_repo=path) for path in stale]
    operations.extend(
        CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=path)
        for path_in_repo, path in sorted(additions.items())
    )
    if any(
        not operation.path_in_repo.startswith("data/") or not operation.path_in_repo.endswith(".parquet")
        for operation in operations
    ):
        raise AssertionError("publication operations must be limited to data/*.parquet")
    return operations, tuple(sorted(additions))


def publish_dataset(
    candidate: ValidatedDataset,
    *,
    repo_id: str,
    license_id: str,
    private: bool,
) -> PublicationResult:
    """Replace only the existing Hub repository's public Parquet shards."""
    from huggingface_hub import HfApi  # noqa: PLC0415

    projected = public_dataset(candidate)
    api = HfApi()
    repo_info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    if not isinstance(repo_info.sha, str):
        raise TypeError("the existing Hub repository did not report a string commit SHA")
    if bool(repo_info.private) != private:
        visibility = "private" if private else "public"
        raise ValueError(f"the existing Hub repository is not {visibility}")
    parent_commit = repo_info.sha
    existing_files = set(
        api.list_repo_files(
            repo_id=repo_id,
            repo_type="dataset",
            revision=parent_commit,
        )
    )
    if "README.md" not in existing_files:
        raise FileNotFoundError("the existing Hub repository has no README.md to preserve")
    readme_before = _read_remote_readme(repo_id, parent_commit)
    _require_readme_license(readme_before, license_id)
    readme_sha256 = hashlib.sha256(readme_before).hexdigest()

    with tempfile.TemporaryDirectory(prefix="engibench-mto2d-hf-") as temporary:
        staged = _stage_public_parquet(projected, Path(temporary) / "data")
        operations, data_files = _commit_operations(staged, existing_files)
        commit = api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            revision="main",
            parent_commit=parent_commit,
            operations=operations,
            commit_message="Use minimal EngiBench-compatible MTO2D columns",
        )

    if not isinstance(commit.oid, str):
        raise TypeError("the Hub update did not report a string commit SHA")
    readme_after = _read_remote_readme(repo_id, commit.oid)
    if readme_after != readme_before:
        raise RuntimeError("Hub README.md changed during the data-only update")
    final_files = set(
        api.list_repo_files(
            repo_id=repo_id,
            repo_type="dataset",
            revision=commit.oid,
        )
    )
    if set(data_files) != {path for path in final_files if path.startswith("data/") and path.endswith(".parquet")}:
        raise RuntimeError("Hub Parquet file layout does not match the staged public dataset")
    return PublicationResult(
        commit_oid=commit.oid,
        commit_url=str(commit.commit_url),
        readme_sha256=readme_sha256,
        data_files=data_files,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default=CANONICAL_REPO_ID)
    parser.add_argument("--license", dest="license_id", default="mit")
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--push",
        action="store_true",
        help="perform the network upload; omitted by default for a safe local dry run",
    )
    parser.add_argument(
        "--confirm-redistribution-rights",
        action="store_true",
        help="confirm that the publisher has verified rights for the solver-native gamma fields",
    )
    return parser


def _push_command(args: argparse.Namespace, dataset_path: Path) -> str:
    command = [
        "python",
        "-m",
        "engibench.problems.mto2d.model.publish_native_dataset",
        "--dataset-dir",
        str(dataset_path),
        "--repo-id",
        args.repo_id,
        "--license",
        args.license_id,
        "--confirm-redistribution-rights",
        "--push",
    ]
    if args.private:
        command.append("--private")
    return " ".join(shlex.quote(part) for part in command)


def main(argv: list[str] | None = None) -> None:
    """Validate locally by default, and publish only through two explicit gates."""
    args = _parser().parse_args(argv)
    candidate = validate_publication_dataset(args.dataset_dir)
    sizes = ", ".join(f"{name}={size:,}" for name, size in candidate.split_sizes.items())
    print(f"Validated exact-native MTO2D dataset at {candidate.path} ({sizes}).")
    print(
        "Labels are historical pre-update values; canonical frozen simulation uses q=0.01 and does not update the design."
    )

    if not args.push:
        print("Dry run complete; no Hub repository was created or modified.")
        print("After verifying redistribution rights, publish with:")
        print(f"  {_push_command(args, candidate.path)}")
        return
    if not args.confirm_redistribution_rights:
        raise ValueError("--push requires --confirm-redistribution-rights")

    result = publish_dataset(
        candidate,
        repo_id=args.repo_id,
        license_id=args.license_id,
        private=args.private,
    )
    print(f"Updated https://huggingface.co/datasets/{args.repo_id}/commit/{result.commit_oid}")
    print(f"README.md preserved byte-for-byte (SHA-256 {result.readme_sha256}).")
    print(f"Published columns: {', '.join(PUBLIC_COLUMNS)}")


if __name__ == "__main__":
    main()
