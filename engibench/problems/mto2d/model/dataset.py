"""Dataset generation and migration helpers for MTO2D.

The functions in this module deliberately keep solver outputs in independent
``.npz`` shards. A SLURM worker therefore returns only a small path string, and
an interrupted 10,000-case campaign can resume without rerunning completed
cases. Dataset assembly streams those shards into Arrow one row at a time.
"""

from collections.abc import Iterator, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import numpy.typing as npt

from engibench.problems.mto2d.model.design_io import HALF_DESIGN_SHAPE
from engibench.problems.mto2d.model.design_io import legacy_256_to_half
from engibench.problems.mto2d.model.design_io import LEGACY_DESIGN_SHAPE

DEFAULT_GRID_SHAPE = (20, 20, 25)
"""Default inlet, power, and volume grid sizes (exactly 10,000 cases)."""

DEFAULT_INLET_RANGE = (-0.095, -0.025)
DEFAULT_POWER_RANGE = (50.0, 75.0)
DEFAULT_VOLUME_RANGE = (0.25, 0.70)
DEFAULT_SPLIT_FRACTIONS = (0.80, 0.15, 0.05)
DEFAULT_SPLIT_SEED = 1
DEFAULT_WRITER_BATCH_SIZE = 16
CONDITION_COLUMN_COUNT = len(DEFAULT_GRID_SHAPE)
RANGE_BOUND_COUNT = len(DEFAULT_INLET_RANGE)
GRID_DIMENSION_COUNT = 2

RAW_REPOSITORY = "IDEALLab/MTO-2D"
RAW_REVISION = "72b4b00b5f1b5942537317214a4d8536c07fb5c8"
RAW_FILENAMES = {
    "design": "gamma_5666_half.npy",
    "conditions": "inp_paras_5666.npy",
    "mean_temperature": "meanT_5666.npy",
    "power_dissipation": "dissP_5666.npy",
    "source_case_id": "index_5666.npy",
}
RAW_SHA256 = {
    "design": "87aa4b2dfa0b8433eb808440502489d7311e9ac2d5967efdd82f12eaf3f2f753",
    "conditions": "f5eb9b65158a1a5f00f580f989458f76321c476963e5b5b5886cd0366a11e3af",
    "mean_temperature": "179f35854f74955d29c8760227f3d2adca3dc89a8571a118fff98a8a925a78ff",
    "power_dissipation": "3fd6dd841e901f17a061beb6957e1e96e95349750282e17032b1018374c1edc7",
    "source_case_id": "1ebb41a932a87dc91f479335173e840613d15f10d44a82d217f7ed035692fcd9",
}
RAW_ROW_COUNT = 5_666

GENERATED_PROVENANCE = "native OpenFOAM gamma field produced by MTO2D.optimize"
LEGACY_PROVENANCE = (
    "lossy reconstruction from the published 256x256 half-domain using "
    "PyTorch-compatible non-antialiased bicubic interpolation"
)
LEGACY_REQUIRED_FINITE_FIELDS = (
    "inlet_velocity",
    "max_power_dissipation",
    "volume_fraction",
    "mean_temperature",
    "power_dissipation",
    "power_constraint_residual_absolute",
    "power_constraint_residual_relative",
)


def condition_grid(
    shape: tuple[int, int, int] = DEFAULT_GRID_SHAPE,
    *,
    inlet_range: tuple[float, float] = DEFAULT_INLET_RANGE,
    power_range: tuple[float, float] = DEFAULT_POWER_RANGE,
    volume_range: tuple[float, float] = DEFAULT_VOLUME_RANGE,
) -> npt.NDArray[np.float64]:
    """Return a stable Cartesian condition grid.

    Columns are ``inlet_velocity``, ``max_power_dissipation``, and
    ``volume_fraction``. Ordering is inlet-major, then power, with volume
    changing fastest. The default shape is ``20 * 20 * 25 == 10_000``.
    """
    if len(shape) != CONDITION_COLUMN_COUNT or any(not isinstance(count, int) or count <= 0 for count in shape):
        raise ValueError("shape must contain three positive integers")

    ranges = (inlet_range, power_range, volume_range)
    if any(
        len(bounds) != RANGE_BOUND_COUNT or not np.all(np.isfinite(bounds)) or bounds[0] > bounds[1] for bounds in ranges
    ):
        raise ValueError("condition ranges must be finite (minimum, maximum) pairs")

    axes = tuple(
        np.linspace(bounds[0], bounds[1], count, dtype=np.float64) for bounds, count in zip(ranges, shape, strict=True)
    )
    meshes = np.meshgrid(*axes, indexing="ij")
    return np.ascontiguousarray(np.stack(meshes, axis=-1).reshape(-1, 3))


def deterministic_split_indices(
    row_count: int,
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    fractions: tuple[float, float, float] = DEFAULT_SPLIT_FRACTIONS,
) -> dict[str, npt.NDArray[np.int64]]:
    """Deterministically partition row positions into train, val, and test.

    The split is performed in two stages like the Beams3D dataset: first take
    the requested training fraction, then divide the held-out remainder
    between validation and test. With the default 80/15/5 fractions, 5,666
    rows yield 4,532/850/284 examples.
    """
    if not isinstance(row_count, int) or row_count < 0:
        raise ValueError("row_count must be a non-negative integer")
    fraction_array = np.asarray(fractions, dtype=np.float64)
    if fraction_array.shape != (3,) or np.any(fraction_array < 0.0):
        raise ValueError("fractions must contain three non-negative values")
    if not np.isclose(float(fraction_array.sum()), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("fractions must sum to 1")

    permutation = np.random.default_rng(seed).permutation(row_count).astype(np.int64, copy=False)
    train_count = int(row_count * fraction_array[0])
    held_out_count = row_count - train_count
    held_out_fraction = float(fraction_array[1] + fraction_array[2])
    test_count = (
        0
        if held_out_count == 0 or held_out_fraction == 0.0
        else math.ceil(held_out_count * fraction_array[2] / held_out_fraction - 1e-12)
    )
    val_count = held_out_count - test_count
    return {
        "train": permutation[:train_count],
        "val": permutation[train_count : train_count + val_count],
        "test": permutation[train_count + val_count :],
    }


def dataset_features() -> Any:
    """Return the common Hugging Face feature schema, importing lazily."""
    from datasets import Features  # noqa: PLC0415
    from datasets import Sequence as DatasetSequence  # noqa: PLC0415
    from datasets import Value  # noqa: PLC0415

    return Features(
        {
            "optimal_design": DatasetSequence(Value("float32")),
            "inlet_velocity": Value("float64"),
            "max_power_dissipation": Value("float64"),
            "volume_fraction": Value("float64"),
            "mean_temperature": Value("float32"),
            "power_dissipation": Value("float32"),
            "power_constraint_residual_absolute": Value("float64"),
            "power_constraint_residual_relative": Value("float64"),
            "volume_constraint_residual": Value("float64"),
            "source_case_id": Value("int64"),
            "source_row_index": Value("int64"),
            "optimization_steps": Value("int32"),
            "optimization_elapsed_time": Value("float64"),
            "evaluation_elapsed_time": Value("float64"),
            "source_dataset": Value("string"),
            "source_revision": Value("string"),
            "design_provenance": Value("string"),
            "design_is_exact": Value("bool"),
            "objectives_evaluated_on_design": Value("bool"),
        }
    )


def generation_jobs(  # noqa: PLR0913
    output_dir: str | Path,
    *,
    solver_config: Mapping[str, Any] | None = None,
    grid: npt.NDArray[np.float64] | None = None,
    start_index: int = 0,
    stop_index: int | None = None,
    evaluate_final: bool = True,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Build picklable keyword arguments for local or SLURM workers."""
    conditions = condition_grid() if grid is None else np.asarray(grid, dtype=np.float64)
    if (
        conditions.ndim != GRID_DIMENSION_COUNT
        or conditions.shape[1] != CONDITION_COLUMN_COUNT
        or not np.all(np.isfinite(conditions))
    ):
        raise ValueError("grid must be a finite array with shape (n_cases, 3)")

    stop = len(conditions) if stop_index is None else stop_index
    if not 0 <= start_index <= stop <= len(conditions):
        raise ValueError("indices must satisfy 0 <= start_index <= stop_index <= len(grid)")

    destination = str(Path(output_dir).expanduser().resolve())
    config = _json_safe_mapping(solver_config or {})
    return [
        {
            "case_id": case_id,
            "inlet_velocity": float(conditions[case_id, 0]),
            "max_power_dissipation": float(conditions[case_id, 1]),
            "volume_fraction": float(conditions[case_id, 2]),
            "output_dir": destination,
            "solver_config": config,
            "evaluate_final": evaluate_final,
            "force": force,
        }
        for case_id in range(start_index, stop)
    ]


def run_optimization_case(  # noqa: PLR0913, PLR0917
    case_id: int,
    inlet_velocity: float,
    max_power_dissipation: float,
    volume_fraction: float,
    output_dir: str,
    solver_config: Mapping[str, Any] | None = None,
    *,
    evaluate_final: bool = True,
    force: bool = False,
) -> str:
    """Optimize one grid point and atomically save a resumable shard.

    This top-level function is intentionally importable and picklable by
    :func:`engibench.utils.slurm.sbatch_map`. It returns only the small shard
    path, never the 400x200 design.
    """
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    shard_path = destination / f"case_{case_id:05d}.npz"
    if shard_path.exists() and not force:
        _validate_existing_shard(
            shard_path,
            case_id=case_id,
            conditions=(inlet_velocity, max_power_dissipation, volume_fraction),
        )
        return str(shard_path)

    from engibench.problems.mto2d.v0 import MTO2D  # noqa: PLC0415

    config = dict(solver_config or {})
    config.update(
        {
            "inlet_velocity": float(inlet_velocity),
            "max_power_dissipation": float(max_power_dissipation),
            "volume_fraction": float(volume_fraction),
        }
    )
    problem = MTO2D(seed=case_id, config=config)
    starting_design = problem.uniform_starting_design(volume_fraction)
    optimized_design, history = problem.optimize(starting_design)
    if not history:
        raise RuntimeError("MTO2D.optimize returned an empty history")

    optimization_run = problem.last_solver_run
    optimization_elapsed = (
        float(optimization_run.elapsed_time[-1])
        if optimization_run is not None and optimization_run.elapsed_time.size
        else float("nan")
    )
    if evaluate_final:
        evaluation = problem.simulate_verbose(optimized_design)
        mean_temperature, power_dissipation = (float(value) for value in evaluation.objective_values)
        volume_residual = float(evaluation.volume_constraint_residual)
        relative_power_residual = float(evaluation.power_constraint_residual)
        evaluation_elapsed = float(evaluation.elapsed_time)
    else:
        mean_temperature, power_dissipation = (float(value) for value in history[-1].obj_values)
        volume_residual = (
            float(optimization_run.volume_residual[-1])
            if optimization_run is not None and optimization_run.volume_residual.size
            else float("nan")
        )
        relative_power_residual = power_dissipation / max_power_dissipation - 1.0
        evaluation_elapsed = float("nan")

    row = {
        "optimal_design": np.asarray(optimized_design, dtype=np.float32),
        "inlet_velocity": float(inlet_velocity),
        "max_power_dissipation": float(max_power_dissipation),
        "volume_fraction": float(volume_fraction),
        "mean_temperature": mean_temperature,
        "power_dissipation": power_dissipation,
        "power_constraint_residual_absolute": power_dissipation - max_power_dissipation,
        "power_constraint_residual_relative": relative_power_residual,
        "volume_constraint_residual": volume_residual,
        "source_case_id": int(case_id),
        "source_row_index": int(case_id),
        "optimization_steps": len(history),
        "optimization_elapsed_time": optimization_elapsed,
        "evaluation_elapsed_time": evaluation_elapsed,
        "source_dataset": "MTO2D generated grid",
        "source_revision": "",
        "design_provenance": GENERATED_PROVENANCE,
        "design_is_exact": True,
        "objectives_evaluated_on_design": evaluate_final,
    }
    _write_shard_atomic(shard_path, row)
    return str(shard_path)


def generate_local(jobs: Sequence[Mapping[str, Any]]) -> list[str]:
    """Run generation jobs sequentially, returning only shard paths."""
    return [run_optimization_case(**dict(job)) for job in jobs]


def submit_slurm(  # noqa: PLR0913
    jobs: Sequence[Mapping[str, Any]],
    *,
    slurm_config: Any,
    group_size: int = 1,
    max_array_size: int = 1_000,
    work_dir: str | Path | None = None,
    wait: bool = False,
) -> list[Any]:
    """Submit shard-producing jobs in scheduler-safe SLURM array batches."""
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if max_array_size <= 0:
        raise ValueError("max_array_size must be positive")
    from engibench.utils import slurm  # noqa: PLC0415

    batch_size = group_size * max_array_size
    base_work_dir = None if work_dir is None else Path(work_dir).expanduser().resolve()
    submitted = []
    for batch_number, start in enumerate(range(0, len(jobs), batch_size)):
        batch_work_dir = None if base_work_dir is None else base_work_dir / f"batch_{batch_number:04d}"
        submitted.append(
            slurm.sbatch_map(
                f=run_optimization_case,
                args=[dict(job) for job in jobs[start : start + batch_size]],
                slurm_args=slurm_config,
                group_size=group_size,
                work_dir=None if batch_work_dir is None else str(batch_work_dir),
                wait=wait,
            )
        )
    return submitted


def discover_shards(
    shard_dir: str | Path,
    *,
    expected_count: int | None = None,
) -> list[Path]:
    """Find shards, reject duplicate IDs, and optionally require a complete range."""
    paths = sorted(Path(shard_dir).expanduser().resolve().glob("case_*.npz"))
    identities = [_shard_identity(path) for path in paths]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate source_case_id values found in shards")
    if expected_count is not None:
        if expected_count < 0:
            raise ValueError("expected_count must be non-negative")
        expected = set(range(expected_count))
        actual = set(identities)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise ValueError(
                f"incomplete shard set: expected IDs 0:{expected_count}; "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
    return [path for _, path in sorted(zip(identities, paths, strict=True))]


def assemble_shards(
    shard_dir: str | Path,
    *,
    expected_count: int | None = None,
    seed: int = DEFAULT_SPLIT_SEED,
    cache_dir: str | Path | None = None,
    writer_batch_size: int = DEFAULT_WRITER_BATCH_SIZE,
) -> Any:
    """Stream completed shards into a Hugging Face ``DatasetDict``."""
    from datasets import Dataset  # noqa: PLC0415
    from datasets import DatasetDict  # noqa: PLC0415

    paths = discover_shards(shard_dir, expected_count=expected_count)
    if writer_batch_size < 1:
        raise ValueError("writer_batch_size must be positive")
    splits = deterministic_split_indices(len(paths), seed=seed)
    features = dataset_features()
    datasets = {}
    for split_name, positions in splits.items():
        split_paths = [str(paths[int(position)]) for position in positions]
        dataset = Dataset.from_generator(
            _iter_shard_rows,
            features=features,
            cache_dir=None if cache_dir is None else str(Path(cache_dir).expanduser().resolve()),
            gen_kwargs={"paths": split_paths},
            split=split_name,
            writer_batch_size=writer_batch_size,
        )
        dataset.info.description = (
            "Native 400x200 MTO2D heat-sink designs generated on a Cartesian condition grid. "
            "Each row was assembled from one resumable solver shard."
        )
        datasets[split_name] = dataset
    return DatasetDict(datasets)


def raw_file_paths(raw_dir: str | Path) -> dict[str, Path]:
    """Resolve and validate the five raw NumPy files in a local directory."""
    directory = Path(raw_dir).expanduser().resolve()
    paths = {key: directory / filename for key, filename in RAW_FILENAMES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing raw MTO-2D files: {missing}")
    return paths


def verify_raw_file_hashes(paths: Mapping[str, str | Path]) -> None:
    """Verify the five raw files against the pinned Hugging Face LFS hashes."""
    missing_keys = sorted(set(RAW_SHA256) - set(paths))
    if missing_keys:
        raise ValueError(f"raw path mapping is missing hash keys: {missing_keys}")
    for key, expected_hash in RAW_SHA256.items():
        digest = hashlib.sha256()
        with Path(paths[key]).expanduser().open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        actual_hash = digest.hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"{key} SHA-256 mismatch: expected {expected_hash}, got {actual_hash} for {paths[key]}")


def download_raw_files(
    *,
    repo_id: str = RAW_REPOSITORY,
    revision: str = RAW_REVISION,
    cache_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Download the pinned raw files via ``huggingface_hub``.

    ``huggingface_hub`` is optional and imported only when this function is
    called. The large files are never downloaded merely by importing MTO2D.
    """
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except ImportError as error:
        raise ImportError("Install 'huggingface_hub' to download the raw MTO-2D dataset") from error

    resolved_cache = None if cache_dir is None else str(Path(cache_dir).expanduser().resolve())
    return {
        key: Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                revision=revision,
                cache_dir=resolved_cache,
            )
        )
        for key, filename in RAW_FILENAMES.items()
    }


def validate_raw_arrays(
    paths: Mapping[str, str | Path],
    *,
    expected_row_count: int | None = None,
) -> int:
    """Validate raw array shapes and return the common row count."""
    arrays = _open_raw_arrays(paths)
    row_count = len(arrays["conditions"])
    if expected_row_count is not None and row_count != expected_row_count:
        raise ValueError(f"raw dataset contains {row_count} rows; expected {expected_row_count}")
    expected_shapes = {
        "conditions": (row_count, CONDITION_COLUMN_COUNT),
        "mean_temperature": (row_count,),
        "power_dissipation": (row_count,),
        "source_case_id": (row_count,),
    }
    for key, expected_shape in expected_shapes.items():
        if arrays[key].shape != expected_shape:
            raise ValueError(f"{key} has shape {arrays[key].shape}; expected {expected_shape}")

    design_shape = arrays["design"].shape
    valid_design_shapes = {(row_count, *LEGACY_DESIGN_SHAPE), (row_count, 1, *LEGACY_DESIGN_SHAPE)}
    if design_shape not in valid_design_shapes:
        raise ValueError(f"design has shape {design_shape}; expected one of {sorted(valid_design_shapes)}")
    if not np.issubdtype(arrays["design"].dtype, np.floating):
        raise TypeError(f"design array must be floating point; got {arrays['design'].dtype}")
    if not np.issubdtype(arrays["source_case_id"].dtype, np.integer):
        raise TypeError(f"source_case_id array must be integer; got {arrays['source_case_id'].dtype}")
    if len(np.unique(arrays["source_case_id"])) != row_count:
        raise ValueError("source_case_id values must be unique")
    small_numeric_arrays = (
        arrays["conditions"],
        arrays["mean_temperature"],
        arrays["power_dissipation"],
    )
    if any(not np.all(np.isfinite(array)) for array in small_numeric_arrays):
        raise ValueError("raw conditions and objectives must contain only finite values")
    if np.any(arrays["conditions"][:, 1] <= 0.0):
        raise ValueError("max_power_dissipation values must be positive")
    return row_count


def convert_raw_arrays(  # noqa: PLR0913
    paths: Mapping[str, str | Path],
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    cache_dir: str | Path | None = None,
    source_dataset: str = RAW_REPOSITORY,
    source_revision: str = RAW_REVISION,
    writer_batch_size: int = DEFAULT_WRITER_BATCH_SIZE,
    verify_hashes: bool = True,
) -> Any:
    """Convert memory-mapped legacy arrays into a streamed ``DatasetDict``."""
    from datasets import Dataset  # noqa: PLC0415
    from datasets import DatasetDict  # noqa: PLC0415

    normalized_paths = {key: str(Path(path).expanduser().resolve()) for key, path in paths.items()}
    is_pinned_source = source_dataset == RAW_REPOSITORY and source_revision == RAW_REVISION
    if writer_batch_size < 1:
        raise ValueError("writer_batch_size must be positive")
    if is_pinned_source and verify_hashes:
        verify_raw_file_hashes(normalized_paths)
    expected_row_count = RAW_ROW_COUNT if is_pinned_source else None
    row_count = validate_raw_arrays(normalized_paths, expected_row_count=expected_row_count)
    splits = deterministic_split_indices(row_count, seed=seed)
    features = dataset_features()
    datasets = {}
    for split_name, positions in splits.items():
        dataset = Dataset.from_generator(
            _iter_raw_rows,
            features=features,
            cache_dir=None if cache_dir is None else str(Path(cache_dir).expanduser().resolve()),
            gen_kwargs={
                "paths": normalized_paths,
                "positions": positions.tolist(),
                "source_dataset": source_dataset,
                "source_revision": source_revision,
            },
            split=split_name,
            writer_batch_size=writer_batch_size,
        )
        dataset.info.description = (
            "IDEALLab/MTO-2D raw NumPy data converted to EngiBench's native 400x200 half-domain. "
            "The published 256x256 designs were reconstructed with a documented lossy bicubic transform."
        )
        dataset.info.license = "mit"
        datasets[split_name] = dataset
    return DatasetDict(datasets)


def legacy_row(  # noqa: PLR0913
    *,
    legacy_design: npt.NDArray,
    conditions: npt.NDArray,
    mean_temperature: float,
    power_dissipation: float,
    source_case_id: int,
    source_row_index: int,
    source_dataset: str = RAW_REPOSITORY,
    source_revision: str = RAW_REVISION,
) -> dict[str, Any]:
    """Convert one raw legacy row to the common schema."""
    design = np.asarray(legacy_design, dtype=np.float32)
    if design.shape == (1, *LEGACY_DESIGN_SHAPE):
        design = design[0]
    condition_array = np.asarray(conditions, dtype=np.float64)
    if condition_array.shape != (CONDITION_COLUMN_COUNT,) or not np.all(np.isfinite(condition_array)):
        raise ValueError("conditions must contain three finite values")
    inlet_velocity, max_power_dissipation, volume_fraction = (float(value) for value in condition_array)
    if max_power_dissipation <= 0.0:
        raise ValueError("max_power_dissipation must be positive")
    mean_temperature = float(mean_temperature)
    power_dissipation = float(power_dissipation)
    if not np.all(np.isfinite((mean_temperature, power_dissipation))):
        raise ValueError("objectives must be finite")
    absolute_residual = power_dissipation - max_power_dissipation
    relative_residual = power_dissipation / max_power_dissipation - 1.0
    return {
        "optimal_design": legacy_256_to_half(design).reshape(-1),
        "inlet_velocity": inlet_velocity,
        "max_power_dissipation": max_power_dissipation,
        "volume_fraction": volume_fraction,
        "mean_temperature": mean_temperature,
        "power_dissipation": power_dissipation,
        "power_constraint_residual_absolute": absolute_residual,
        "power_constraint_residual_relative": relative_residual,
        "volume_constraint_residual": None,
        "source_case_id": int(source_case_id),
        "source_row_index": int(source_row_index),
        "optimization_steps": None,
        "optimization_elapsed_time": None,
        "evaluation_elapsed_time": None,
        "source_dataset": source_dataset,
        "source_revision": source_revision,
        "design_provenance": LEGACY_PROVENANCE,
        "design_is_exact": False,
        "objectives_evaluated_on_design": False,
    }


def validate_legacy_dataset(
    dataset: Mapping[str, Any],
    *,
    row_count: int = RAW_ROW_COUNT,
    seed: int = DEFAULT_SPLIT_SEED,
    source_dataset: str = RAW_REPOSITORY,
    source_revision: str = RAW_REVISION,
) -> dict[str, int]:
    """Validate a converted legacy DatasetDict before publication."""
    expected_indices = deterministic_split_indices(row_count, seed=seed)
    expected_sizes = {name: len(indices) for name, indices in expected_indices.items()}
    if set(dataset) != set(expected_sizes):
        raise ValueError(f"dataset splits must be {sorted(expected_sizes)}; got {sorted(dataset)}")

    all_ids: list[int] = []
    reference_row: Mapping[str, Any] | None = None
    for split_name, expected_size in expected_sizes.items():
        split = dataset[split_name]
        source_ids, split_reference = _validate_legacy_split(
            split_name,
            split,
            expected_size,
            expected_indices[split_name],
            source_dataset=source_dataset,
            source_revision=source_revision,
        )
        all_ids.extend(source_ids)
        reference_row = split_reference or reference_row

    if len(all_ids) != row_count or len(set(all_ids)) != row_count:
        raise ValueError("source_case_id values must be unique across all splits")
    if source_dataset == RAW_REPOSITORY and source_revision == RAW_REVISION:
        _validate_retained_reference(reference_row)
    return expected_sizes


def _validate_legacy_split(  # noqa: PLR0913
    split_name: str,
    split: Any,
    expected_size: int,
    expected_positions: npt.NDArray[np.int64],
    *,
    source_dataset: str,
    source_revision: str,
) -> tuple[list[int], Mapping[str, Any] | None]:
    if len(split) != expected_size:
        raise ValueError(f"{split_name} contains {len(split)} rows; expected {expected_size}")
    if split.features != dataset_features():
        raise ValueError(f"{split_name} features do not match the MTO2D v0 schema")

    _validate_flat_design_column(split_name, _logical_arrow_column(split, "optimal_design"))
    _validate_finite_columns(split_name, split)
    _validate_residual_columns(split_name, split)
    _validate_legacy_metadata(split_name, split, source_dataset, source_revision)
    source_ids = [int(value) for value in split["source_case_id"]]
    actual_positions = np.asarray(split["source_row_index"], dtype=np.int64)
    if not np.array_equal(actual_positions, expected_positions):
        raise ValueError(f"{split_name}.source_row_index does not match the deterministic split membership and order")
    reference = split[source_ids.index(0)] if 0 in source_ids else None
    return source_ids, reference


def _validate_flat_design_column(split_name: str, designs: Any) -> None:
    import pyarrow.compute as pc  # type: ignore[import-untyped]  # noqa: PLC0415

    design_length = int(np.prod(HALF_DESIGN_SHAPE))
    for chunk in designs.chunks:
        if chunk.null_count or chunk.values.null_count:
            raise ValueError(f"{split_name} contains null designs or design values")
        length_bounds = pc.min_max(pc.list_value_length(chunk)).as_py()
        if length_bounds != {"min": design_length, "max": design_length}:
            raise ValueError(f"{split_name} contains a design whose flattened length is not {design_length}")
        values = chunk.values
        if not pc.all(pc.is_finite(values)).as_py():
            raise ValueError(f"{split_name} contains non-finite design values")
        value_bounds = pc.min_max(values).as_py()
        if value_bounds["min"] < 0.0 or value_bounds["max"] > 1.0:
            raise ValueError(f"{split_name} contains design values outside [0, 1]")


def _validate_finite_columns(split_name: str, split: Any) -> None:
    import pyarrow.compute as pc  # noqa: PLC0415

    for name in LEGACY_REQUIRED_FINITE_FIELDS:
        column = _logical_arrow_column(split, name).combine_chunks()
        if column.null_count or not pc.all(pc.is_finite(column)).as_py():
            raise ValueError(f"{split_name}.{name} must contain only finite, non-null values")


def _validate_residual_columns(split_name: str, split: Any) -> None:
    maximum = np.asarray(split["max_power_dissipation"], dtype=np.float64)
    measured = np.asarray(split["power_dissipation"], dtype=np.float64)
    absolute = np.asarray(split["power_constraint_residual_absolute"], dtype=np.float64)
    relative = np.asarray(split["power_constraint_residual_relative"], dtype=np.float64)
    if not np.allclose(absolute, measured - maximum, rtol=1e-6, atol=1e-5):
        raise ValueError(f"{split_name}.power_constraint_residual_absolute is inconsistent")
    if not np.allclose(relative, measured / maximum - 1.0, rtol=1e-6, atol=1e-7):
        raise ValueError(f"{split_name}.power_constraint_residual_relative is inconsistent")


def _validate_legacy_metadata(
    split_name: str,
    split: Any,
    source_dataset: str,
    source_revision: str,
) -> None:
    if any(value is not False for value in split["design_is_exact"]) or any(
        value is not False for value in split["objectives_evaluated_on_design"]
    ):
        raise ValueError(f"{split_name} must mark reconstructed designs and objectives as inexact")
    if set(split["source_dataset"]) != {source_dataset}:
        raise ValueError(f"{split_name} must record source dataset {source_dataset}")
    if set(split["source_revision"]) != {source_revision}:
        raise ValueError(f"{split_name} must record source revision {source_revision}")
    if set(split["design_provenance"]) != {LEGACY_PROVENANCE}:
        raise ValueError(f"{split_name} contains unexpected design provenance")
    for field in (
        "volume_constraint_residual",
        "optimization_steps",
        "optimization_elapsed_time",
        "evaluation_elapsed_time",
    ):
        if _logical_arrow_column(split, field).null_count != len(split):
            raise ValueError(f"{split_name}.{field} must be null because the source does not provide it")


def _logical_arrow_column(split: Any, name: str) -> Any:
    """Return an Arrow column while respecting any Dataset row indices."""
    return split.with_format("arrow")[name]


def _validate_retained_reference(reference_row: Mapping[str, Any] | None) -> None:
    if reference_row is None:
        raise ValueError("converted dataset does not contain retained reference source_case_id=0")
    conditions = [
        reference_row["inlet_velocity"],
        reference_row["max_power_dissipation"],
        reference_row["volume_fraction"],
    ]
    objectives = [reference_row["mean_temperature"], reference_row["power_dissipation"]]
    if not np.allclose(conditions, [-0.074, 63.1, 0.61], rtol=0.0, atol=1e-12):
        raise ValueError(f"retained reference conditions do not match: {conditions}")
    if not np.allclose(objectives, [9.45825, 62.2588], rtol=0.0, atol=1e-5):
        raise ValueError(f"retained reference objectives do not match: {objectives}")


def load_solver_config(path: str | Path | None) -> dict[str, Any]:
    """Load a JSON object containing MTO2D external solver configuration."""
    if path is None:
        return {}
    parsed = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError("solver configuration JSON must contain an object")
    return _json_safe_mapping(parsed)


def _json_safe_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    try:
        encoded = json.dumps(dict(values))
    except (TypeError, ValueError) as error:
        raise ValueError("solver_config must contain JSON-serializable values") from error
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise TypeError("solver_config must be a mapping")
    return decoded


def _write_shard_atomic(path: Path, row: Mapping[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.stem}-",
            suffix=".npz",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
        np.savez_compressed(temporary_path, **row)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _scalar(array: npt.NDArray, name: str) -> Any:
    if array.shape != ():
        raise ValueError(f"shard field {name!r} must be scalar; got {array.shape}")
    return array.item()


def _shard_identity(path: Path) -> int:
    with np.load(path, allow_pickle=False) as shard:
        if "source_case_id" not in shard:
            raise ValueError(f"shard is missing source_case_id: {path}")
        return int(_scalar(shard["source_case_id"], "source_case_id"))


def _validate_existing_shard(
    path: Path,
    *,
    case_id: int,
    conditions: tuple[float, float, float],
) -> None:
    row = _load_shard_row(path)
    if row["source_case_id"] != case_id:
        raise ValueError(f"existing shard {path} has source_case_id={row['source_case_id']}, expected {case_id}")
    stored_conditions = np.array(
        [
            row["inlet_velocity"],
            row["max_power_dissipation"],
            row["volume_fraction"],
        ],
        dtype=np.float64,
    )
    if not np.allclose(stored_conditions, conditions, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"existing shard {path} conditions {stored_conditions.tolist()} do not match "
            f"requested conditions {list(conditions)}; pass force=True to replace it"
        )


def _load_shard_row(path: str | Path) -> dict[str, Any]:
    feature_names = tuple(dataset_features())
    with np.load(path, allow_pickle=False) as shard:
        missing = [name for name in feature_names if name not in shard]
        if missing:
            raise ValueError(f"shard {path} is missing fields: {missing}")
        row = {
            name: (np.asarray(shard[name], dtype=np.float32) if name == "optimal_design" else _scalar(shard[name], name))
            for name in feature_names
        }
    design = np.asarray(row["optimal_design"], dtype=np.float32)
    if design.shape == (int(np.prod(HALF_DESIGN_SHAPE)),):
        native_design = design.reshape(HALF_DESIGN_SHAPE)
    elif design.shape == HALF_DESIGN_SHAPE:
        native_design = design
    else:
        raise ValueError(
            f"shard {path} design has shape {design.shape}; "
            f"expected {HALF_DESIGN_SHAPE} or {(int(np.prod(HALF_DESIGN_SHAPE)),)}"
        )
    if not np.all(np.isfinite(native_design)) or np.any((native_design < 0.0) | (native_design > 1.0)):
        raise ValueError(f"shard {path} design must contain finite values in [0, 1]")
    row["optimal_design"] = native_design.reshape(-1)
    return row


def _iter_shard_rows(paths: Sequence[str]) -> Iterator[dict[str, Any]]:
    for path in paths:
        yield _load_shard_row(path)


def _open_raw_arrays(paths: Mapping[str, str | Path]) -> dict[str, npt.NDArray]:
    missing_keys = sorted(set(RAW_FILENAMES) - set(paths))
    if missing_keys:
        raise ValueError(f"raw path mapping is missing keys: {missing_keys}")
    return {key: np.load(Path(paths[key]).expanduser(), mmap_mode="r", allow_pickle=False) for key in RAW_FILENAMES}


def _iter_raw_rows(
    paths: Mapping[str, str],
    positions: Sequence[int],
    source_dataset: str,
    source_revision: str,
) -> Iterator[dict[str, Any]]:
    arrays = _open_raw_arrays(paths)
    for position in positions:
        yield legacy_row(
            legacy_design=arrays["design"][position],
            conditions=arrays["conditions"][position],
            mean_temperature=float(arrays["mean_temperature"][position]),
            power_dissipation=float(arrays["power_dissipation"][position]),
            source_case_id=int(arrays["source_case_id"][position]),
            source_row_index=int(position),
            source_dataset=source_dataset,
            source_revision=source_revision,
        )
