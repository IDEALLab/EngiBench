"""Build an EngiBench dataset from validated solver-native gamma fields.

This converter consumes the manifests written by
``retrieve_native_gammas validate`` and the four small metadata arrays from
the pinned ``IDEALLab/MTO-2D`` snapshot. Historical objective labels are
preserved, but explicitly marked as not evaluated on the downloaded
post-update gamma field.

Example:
    Convert a complete validated retrieval::

        python -m engibench.problems.mto2d.model.reformat_native_gamma_dataset \
            --gamma-dir /path/to/source-gammas \
            --raw-dir /path/to/pinned/MTO-2D/snapshot \
            --output-dir /path/to/mto_2d_native_v0
"""

import argparse
from collections.abc import Iterator, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import numpy.typing as npt

from engibench.problems.mto2d.model.dataset import canonicalize_dataset_columns
from engibench.problems.mto2d.model.dataset import CONDITION_COLUMN_COUNT
from engibench.problems.mto2d.model.dataset import dataset_features
from engibench.problems.mto2d.model.dataset import DEFAULT_SPLIT_SEED
from engibench.problems.mto2d.model.dataset import DEFAULT_WRITER_BATCH_SIZE
from engibench.problems.mto2d.model.dataset import LEGACY_SPLIT_ALGORITHM
from engibench.problems.mto2d.model.dataset import LEGACY_SPLIT_FRACTIONS
from engibench.problems.mto2d.model.dataset import legacy_split_indices
from engibench.problems.mto2d.model.dataset import LEGACY_SPLIT_POLICY
from engibench.problems.mto2d.model.dataset import RAW_FILENAMES
from engibench.problems.mto2d.model.dataset import RAW_REPOSITORY
from engibench.problems.mto2d.model.dataset import RAW_REVISION
from engibench.problems.mto2d.model.dataset import RAW_ROW_COUNT
from engibench.problems.mto2d.model.dataset import RAW_SHA256
from engibench.problems.mto2d.model.design_io import DESIGN_CELL_COUNT
from engibench.problems.mto2d.model.design_io import FIXED_CELL_COUNT
from engibench.problems.mto2d.model.design_io import GAMMA_CELL_COUNT
from engibench.problems.mto2d.model.design_io import gamma_to_half_design
from engibench.problems.mto2d.model.design_io import parse_internal_field
from engibench.problems.mto2d.model.retrieve_native_gammas import read_source_cases
from engibench.problems.mto2d.model.retrieve_native_gammas import SOURCE_CASES_FILENAME
from engibench.problems.mto2d.model.retrieve_native_gammas import VALIDATION_RECORDS_FILENAME
from engibench.problems.mto2d.model.retrieve_native_gammas import VALIDATION_SUMMARY_FILENAME

METADATA_KEYS = (
    "conditions",
    "mean_temperature",
    "power_dissipation",
    "source_case_id",
)
NATIVE_SOURCE_PROVENANCE = (
    "exact solver-native OpenFOAM app/200 gamma field; historical objectives "
    "were logged before the final MMA/Heaviside design update"
)
SHA256_HEX_LENGTH = 64
CONVERSION_MANIFEST_FILENAME = "native_conversion_manifest.json"
DATASET_CARD_FILENAME = "README.md"
PUBLICATION_BLOCK_REASON = (
    "Redistribution rights for the solver-native gamma fields have not been "
    "verified; do not publish this dataset until permission is confirmed."
)
CANONICAL_FROZEN_SIMULATION = {
    "ramp_q": 0.01,
    "alpha_max": 5_025_200.0,
    "heaviside": 59.8,
    "design_update": False,
}
"""Source-matched final physics used by the canonical frozen simulator."""

HISTORICAL_LABEL_SEMANTICS = {
    "evaluated_on_stored_design": False,
    "timing": "pre-final-MMA/Heaviside-update",
    "stored_design_timing": "post-final-MMA/Heaviside-update",
    "note": (
        "The historical labels were produced by the source optimization and "
        "must not be presented as frozen q=0.01 evaluations of the stored design."
    ),
}
"""Explicit distinction between source labels and canonical re-evaluation."""


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def raw_metadata_paths(raw_dir: str | Path) -> dict[str, Path]:
    """Resolve the four small pinned arrays needed for native conversion."""
    directory = Path(raw_dir).expanduser().resolve()
    paths = {key: directory / RAW_FILENAMES[key] for key in METADATA_KEYS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing raw MTO-2D metadata files: {missing}")
    return paths


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_metadata_hashes(paths: Mapping[str, str | Path]) -> None:
    """Verify metadata arrays against the pinned Hugging Face revision."""
    missing_keys = sorted(set(METADATA_KEYS) - set(paths))
    if missing_keys:
        raise ValueError(f"raw metadata path mapping is missing keys: {missing_keys}")
    for key in METADATA_KEYS:
        path = Path(paths[key]).expanduser().resolve()
        actual = _sha256_file(path)
        expected = RAW_SHA256[key]
        if actual != expected:
            raise ValueError(f"{key} SHA-256 mismatch: expected {expected}, got {actual} for {path}")


def _open_metadata_arrays(paths: Mapping[str, str | Path]) -> dict[str, npt.NDArray]:
    missing_keys = sorted(set(METADATA_KEYS) - set(paths))
    if missing_keys:
        raise ValueError(f"raw metadata path mapping is missing keys: {missing_keys}")
    return {key: np.load(Path(paths[key]).expanduser(), mmap_mode="r", allow_pickle=False) for key in METADATA_KEYS}


def validate_metadata_arrays(
    paths: Mapping[str, str | Path],
    *,
    expected_row_count: int | None = None,
) -> int:
    """Validate native-conversion metadata and return its common row count."""
    arrays = _open_metadata_arrays(paths)
    row_count = len(arrays["conditions"])
    if expected_row_count is not None and row_count != expected_row_count:
        raise ValueError(f"raw metadata contains {row_count} rows; expected {expected_row_count}")
    expected_shapes = {
        "conditions": (row_count, CONDITION_COLUMN_COUNT),
        "mean_temperature": (row_count,),
        "power_dissipation": (row_count,),
        "source_case_id": (row_count,),
    }
    for key, expected_shape in expected_shapes.items():
        if arrays[key].shape != expected_shape:
            raise ValueError(f"{key} has shape {arrays[key].shape}; expected {expected_shape}")
    if not np.issubdtype(arrays["source_case_id"].dtype, np.integer):
        raise TypeError(f"source_case_id array must be integer; got {arrays['source_case_id'].dtype}")
    if len(np.unique(arrays["source_case_id"])) != row_count:
        raise ValueError("source_case_id values must be unique")
    numeric_arrays = (
        arrays["conditions"],
        arrays["mean_temperature"],
        arrays["power_dissipation"],
    )
    if any(not np.all(np.isfinite(array)) for array in numeric_arrays):
        raise ValueError("raw conditions and objectives must contain only finite values")
    if np.any(arrays["conditions"][:, 1] <= 0.0):
        raise ValueError("max_power_dissipation values must be positive")
    return row_count


def validated_gamma_records(gamma_dir: str | Path) -> list[dict[str, Any]]:  # noqa: C901
    """Load a complete validation manifest in source-row order."""
    directory = Path(gamma_dir).expanduser().resolve()
    cases = read_source_cases(directory / SOURCE_CASES_FILENAME)
    summary_path = directory / VALIDATION_SUMMARY_FILENAME
    records_path = directory / VALIDATION_RECORDS_FILENAME
    if not summary_path.is_file() or not records_path.is_file():
        raise FileNotFoundError("native gamma validation summary and records are required")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or summary.get("complete") is not True:
        raise ValueError("native gamma validation is incomplete")
    if summary.get("expected") != len(cases) or summary.get("valid") != len(cases):
        raise ValueError("native gamma validation summary does not match the source-case manifest")

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(records_path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid gamma validation JSON on line {line_number}") from error
        if not isinstance(parsed, dict):
            raise TypeError(f"gamma validation record {line_number} must be a JSON object")
        records.append(parsed)
    if len(records) != len(cases):
        raise ValueError("gamma validation record count does not match the source-case manifest")

    for case, record in zip(cases, records, strict=True):
        expected_identity = (case.source_row_index, case.source_case_id, case.relative_path)
        actual_identity = (
            record.get("source_row_index"),
            record.get("source_case_id"),
            record.get("relative_path"),
        )
        if actual_identity != expected_identity:
            raise ValueError(f"gamma validation record identity mismatch for source row {case.source_row_index}")
        digest = record.get("sha256")
        if record.get("status") != "valid" or record.get("fixed_tail_valid") is not True:
            raise ValueError(f"source row {case.source_row_index} is not a validated native gamma field")
        if not isinstance(digest, str) or len(digest) != SHA256_HEX_LENGTH:
            raise ValueError(f"source row {case.source_row_index} has no valid SHA-256 record")
    return records


def _read_validated_gamma(path: Path, expected_sha256: str) -> npt.NDArray[np.float64]:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read validated gamma field: {path}") from error
    actual_sha256 = _sha256_bytes(content)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"validated gamma SHA-256 changed: expected {expected_sha256}, got {actual_sha256} for {path}")
    try:
        text = content.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"validated gamma is not ASCII text: {path}") from error
    values = parse_internal_field(text, expected_count=GAMMA_CELL_COUNT)
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"validated gamma values lie outside [0, 1]: {path}")
    if not np.array_equal(values[DESIGN_CELL_COUNT:], np.ones(FIXED_CELL_COUNT, dtype=values.dtype)):
        raise ValueError(f"validated gamma fixed-cell tail changed: {path}")
    return values


def native_source_row(  # noqa: PLR0913
    *,
    gamma_values: npt.NDArray,
    conditions: npt.NDArray,
    mean_temperature: float,
    power_dissipation: float,
    source_case_id: int,
    source_row_index: int,
    source_dataset: str = RAW_REPOSITORY,
    source_revision: str = RAW_REVISION,
) -> dict[str, Any]:
    """Build one exact-design row with preserved historical labels."""
    values = np.asarray(gamma_values, dtype=np.float64)
    if values.shape != (GAMMA_CELL_COUNT,):
        raise ValueError(f"gamma must contain {GAMMA_CELL_COUNT} values; got {values.shape}")
    if not np.array_equal(values[DESIGN_CELL_COUNT:], np.ones(FIXED_CELL_COUNT, dtype=values.dtype)):
        raise ValueError("gamma final 6,400 fixed/non-design cells must all equal one")
    condition_array = np.asarray(conditions, dtype=np.float64)
    if condition_array.shape != (CONDITION_COLUMN_COUNT,) or not np.all(np.isfinite(condition_array)):
        raise ValueError("conditions must contain three finite values")
    inlet_velocity, max_power_dissipation, volfrac = (float(value) for value in condition_array)
    if max_power_dissipation <= 0.0:
        raise ValueError("max_power_dissipation must be positive")
    mean_temperature = float(mean_temperature)
    power_dissipation = float(power_dissipation)
    if not np.all(np.isfinite((mean_temperature, power_dissipation))):
        raise ValueError("objectives must be finite")

    return {
        "optimal_design": gamma_to_half_design(values).reshape(-1),
        "inlet_velocity": inlet_velocity,
        "max_power_dissipation": max_power_dissipation,
        "volfrac": volfrac,
        "mean_temperature": mean_temperature,
        "power_dissipation": power_dissipation,
        "power_constraint_residual_absolute": power_dissipation - max_power_dissipation,
        "power_constraint_residual_relative": power_dissipation / max_power_dissipation - 1.0,
        "volume_constraint_residual": float(np.mean(values) - volfrac),
        "source_case_id": int(source_case_id),
        "source_row_index": int(source_row_index),
        "optimization_steps": None,
        "optimization_elapsed_time": None,
        "evaluation_elapsed_time": None,
        "source_dataset": source_dataset,
        "source_revision": source_revision,
        "design_provenance": NATIVE_SOURCE_PROVENANCE,
        "design_is_exact": True,
        "objectives_evaluated_on_design": False,
    }


def _iter_native_rows(
    metadata_paths: Mapping[str, str],
    gamma_dir: str,
    records: Sequence[Mapping[str, Any]],
    positions: Sequence[int],
    source: Mapping[str, str],
) -> Iterator[dict[str, Any]]:
    arrays = _open_metadata_arrays(metadata_paths)
    directory = Path(gamma_dir)
    for position in positions:
        record = records[position]
        values = _read_validated_gamma(directory / str(record["relative_path"]), str(record["sha256"]))
        yield native_source_row(
            gamma_values=values,
            conditions=arrays["conditions"][position],
            mean_temperature=float(arrays["mean_temperature"][position]),
            power_dissipation=float(arrays["power_dissipation"][position]),
            source_case_id=int(arrays["source_case_id"][position]),
            source_row_index=int(position),
            source_dataset=source["dataset"],
            source_revision=source["revision"],
        )


def convert_native_gamma_fields(  # noqa: PLR0913
    gamma_dir: str | Path,
    metadata_paths: Mapping[str, str | Path],
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    cache_dir: str | Path | None = None,
    source_dataset: str = RAW_REPOSITORY,
    source_revision: str = RAW_REVISION,
    writer_batch_size: int = DEFAULT_WRITER_BATCH_SIZE,
    verify_hashes: bool = True,
) -> Any:
    """Stream validated native fields and pinned metadata into a DatasetDict."""
    from datasets import Dataset  # noqa: PLC0415
    from datasets import DatasetDict  # noqa: PLC0415

    if writer_batch_size < 1:
        raise ValueError("writer_batch_size must be positive")
    normalized_metadata = {key: str(Path(path).expanduser().resolve()) for key, path in metadata_paths.items()}
    is_pinned_source = source_dataset == RAW_REPOSITORY and source_revision == RAW_REVISION
    if is_pinned_source and verify_hashes:
        verify_metadata_hashes(normalized_metadata)
    expected_row_count = RAW_ROW_COUNT if is_pinned_source else None
    row_count = validate_metadata_arrays(normalized_metadata, expected_row_count=expected_row_count)

    resolved_gamma_dir = Path(gamma_dir).expanduser().resolve()
    records = validated_gamma_records(resolved_gamma_dir)
    if len(records) != row_count:
        raise ValueError(f"validated gamma manifest contains {len(records)} rows; metadata contains {row_count}")
    arrays = _open_metadata_arrays(normalized_metadata)
    metadata_case_ids = np.asarray(arrays["source_case_id"], dtype=np.int64)
    record_case_ids = np.asarray([record["source_case_id"] for record in records], dtype=np.int64)
    if not np.array_equal(record_case_ids, metadata_case_ids):
        raise ValueError("validated gamma case IDs do not match index_5666.npy in source-row order")

    splits = legacy_split_indices(row_count, seed=seed)
    features = dataset_features()
    datasets = {}
    serialized_records = tuple(
        {
            "relative_path": str(record["relative_path"]),
            "sha256": str(record["sha256"]),
        }
        for record in records
    )
    for split_name, positions in splits.items():
        if len(positions) == 0:
            dataset = Dataset.from_dict(
                {name: [] for name in features},
                features=features,
                split=split_name,
            )
        else:
            try:
                dataset = Dataset.from_generator(
                    _iter_native_rows,
                    features=features,
                    cache_dir=None if cache_dir is None else str(Path(cache_dir).expanduser().resolve()),
                    gen_kwargs={
                        "metadata_paths": normalized_metadata,
                        "gamma_dir": str(resolved_gamma_dir),
                        "records": serialized_records,
                        "positions": positions.tolist(),
                        "source": {
                            "dataset": source_dataset,
                            "revision": source_revision,
                        },
                    },
                    split=split_name,
                    writer_batch_size=writer_batch_size,
                )
            except Exception as error:
                cause = error.__cause__
                if isinstance(cause, ValueError):
                    raise cause from error
                raise
        dataset.info.description = (
            "Exact solver-native MTO2D app/200 gamma fields paired with preserved historical pre-update objective labels."
        )
        datasets[split_name] = dataset
    return DatasetDict(datasets)


def _validate_saved_dataset(
    dataset: Mapping[str, Any],
    *,
    source_case_ids: npt.NDArray[np.int64],
    seed: int,
) -> dict[str, int]:
    dataset = canonicalize_dataset_columns(dataset)
    expected_indices = legacy_split_indices(len(source_case_ids), seed=seed)
    expected_sizes = {name: len(indices) for name, indices in expected_indices.items()}
    if set(dataset) != set(expected_sizes):
        raise ValueError(f"saved dataset splits must be {sorted(expected_sizes)}; got {sorted(dataset)}")

    for split_name, positions in expected_indices.items():
        split = dataset[split_name]
        if len(split) != len(positions):
            raise ValueError(f"saved {split_name} split has {len(split)} rows; expected {len(positions)}")
        if split.features != dataset_features():
            raise ValueError(f"saved {split_name} split does not match the MTO2D schema")
        if np.asarray(split["source_row_index"], dtype=np.int64).tolist() != positions.tolist():
            raise ValueError(f"saved {split_name} source-row order does not match the legacy split")
        expected_case_ids = source_case_ids[positions]
        if np.asarray(split["source_case_id"], dtype=np.int64).tolist() != expected_case_ids.tolist():
            raise ValueError(f"saved {split_name} source-case IDs do not match the pinned index")
        if any(value is not True for value in split["design_is_exact"]):
            raise ValueError(f"saved {split_name} split must mark every design exact")
        if any(value is not False for value in split["objectives_evaluated_on_design"]):
            raise ValueError(f"saved {split_name} split must mark historical objectives unevaluated on the design")
    return expected_sizes


def _write_handoff_metadata(  # noqa: PLR0913
    output: Path,
    *,
    source_dataset: str,
    source_revision: str,
    row_count: int,
    seed: int,
    split_sizes: Mapping[str, int],
    metadata_hashes_verified: bool,
    metadata_sha256: Mapping[str, str] | None,
    validation_records_sha256: str,
) -> None:
    manifest = {
        "schema": "engibench-mto2d-exact-native-v0",
        "source_dataset": source_dataset,
        "source_revision": source_revision,
        "row_count": row_count,
        "split_policy": LEGACY_SPLIT_POLICY,
        "split_fractions": list(LEGACY_SPLIT_FRACTIONS),
        "split_algorithm": LEGACY_SPLIT_ALGORITHM,
        "split_seed": seed,
        "split_sizes": dict(split_sizes),
        "native_design_shape": [400, 200],
        "stored_design_length": DESIGN_CELL_COUNT,
        "design_is_exact": True,
        "objectives_evaluated_on_design": False,
        "objective_semantics": "historical pre-final-MMA/Heaviside-update source labels",
        "historical_label_semantics": HISTORICAL_LABEL_SEMANTICS,
        "canonical_frozen_simulation": CANONICAL_FROZEN_SIMULATION,
        "residual_semantics": {
            "volume_constraint_residual": (
                "mean of all 86,400 cells in the exact post-update app/200/gamma field minus volfrac"
            ),
            "power_constraint_residual": (
                "computed from the historical pre-update power-dissipation label and max_power_dissipation"
            ),
        },
        "metadata_hashes_verified": metadata_hashes_verified,
        "metadata_sha256": None if metadata_sha256 is None else dict(metadata_sha256),
        "gamma_validation_records_file": VALIDATION_RECORDS_FILENAME,
        "gamma_validation_records_sha256": validation_records_sha256,
        "redistribution_rights": "unverified",
        "publication_ready": False,
        "publication_block_reason": PUBLICATION_BLOCK_REASON,
    }
    (output / CONVERSION_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / DATASET_CARD_FILENAME).write_text(
        f"""# MTO2D exact-native v0

This local DatasetDict contains exact solver-native `(400, 200)` MTO2D
topologies retrieved from the source `app/200/gamma` fields. Historical
mean-temperature and power-dissipation labels are preserved, but they were
logged before the final MMA/Heaviside update and were not evaluated on the
stored post-update design.

## Simulation physics and label semantics

EngiBench's canonical frozen simulation uses the source-matched final
RAMP parameter `q=0.01`, `alphaMax=5.0252e6`, and `Heaviside=59.8`, with
design updates disabled. The stored objective columns are historical source
labels, **not** frozen `q=0.01` evaluations of the stored fields. Keep
`objectives_evaluated_on_design=false` unless each row is explicitly
re-evaluated and relabeled.

## Residual timing

`volume_constraint_residual` is evaluated directly on the exact post-update
`app/200/gamma` field as the mean of all 86,400 cells minus the requested
volume fraction. The power residual instead uses the historical pre-update
power-dissipation label. These residuals therefore describe different solver
states and must not be interpreted as one simultaneous evaluation.

## Publication and licensing

**Publication blocked:** {PUBLICATION_BLOCK_REASON}

No license is asserted for the retrieved higher-resolution topology fields.
The original raw metadata source is `{source_dataset}` at revision
`{source_revision}`.
""",
        encoding="utf-8",
    )


def _verify_copied_evidence(path: Path, expected_sha256: str) -> None:
    if _sha256_file(path) != expected_sha256:
        raise ValueError("copied gamma validation evidence has an unexpected SHA-256")


def convert_and_save(
    gamma_dir: str | Path,
    raw_dir: str | Path,
    output_dir: str | Path,
    **kwargs: Any,
) -> Any:
    """Convert, reload-validate, and atomically save the native DatasetDict."""
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_paths = raw_metadata_paths(raw_dir)
    dataset = convert_native_gamma_fields(
        gamma_dir,
        metadata_paths,
        **kwargs,
    )
    source_dataset = str(kwargs.get("source_dataset", RAW_REPOSITORY))
    source_revision = str(kwargs.get("source_revision", RAW_REVISION))
    seed = int(kwargs.get("seed", DEFAULT_SPLIT_SEED))
    verify_hashes = bool(kwargs.get("verify_hashes", True))
    pinned_source = source_dataset == RAW_REPOSITORY and source_revision == RAW_REVISION
    metadata_hashes_verified = pinned_source and verify_hashes
    metadata_sha256 = {key: RAW_SHA256[key] for key in METADATA_KEYS} if pinned_source else None
    source_case_ids = np.asarray(_open_metadata_arrays(metadata_paths)["source_case_id"], dtype=np.int64)
    gamma_directory = Path(gamma_dir).expanduser().resolve()
    validation_records = gamma_directory / VALIDATION_RECORDS_FILENAME
    validation_records_sha256 = _sha256_file(validation_records)

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
        )
    )
    try:
        dataset.save_to_disk(str(temporary))
        shutil.copyfile(validation_records, temporary / VALIDATION_RECORDS_FILENAME)
        _verify_copied_evidence(temporary / VALIDATION_RECORDS_FILENAME, validation_records_sha256)

        from datasets import load_from_disk  # noqa: PLC0415

        reloaded = load_from_disk(str(temporary))
        split_sizes = _validate_saved_dataset(
            reloaded,
            source_case_ids=source_case_ids,
            seed=seed,
        )
        _write_handoff_metadata(
            temporary,
            source_dataset=source_dataset,
            source_revision=source_revision,
            row_count=len(source_case_ids),
            seed=seed,
            split_sizes=split_sizes,
            metadata_hashes_verified=metadata_hashes_verified,
            metadata_sha256=metadata_sha256,
            validation_records_sha256=validation_records_sha256,
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--gamma-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--writer-batch-size", type=int, default=DEFAULT_WRITER_BATCH_SIZE)
    parser.add_argument(
        "--verify-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="verify all four metadata arrays against the pinned Hugging Face revision",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the exact-native dataset conversion command."""
    args = _parser().parse_args(argv)
    dataset = convert_and_save(
        args.gamma_dir,
        args.raw_dir,
        args.output_dir,
        seed=args.seed,
        cache_dir=args.cache_dir,
        writer_batch_size=args.writer_batch_size,
        verify_hashes=args.verify_hashes,
    )
    sizes = {name: len(split) for name, split in dataset.items()}
    size_text = ", ".join(f"{name}={size:,}" for name, size in sizes.items())
    print(f"Saved exact-native MTO2D dataset ({size_text}) at {Path(args.output_dir).expanduser().resolve()}.")


if __name__ == "__main__":
    main()
