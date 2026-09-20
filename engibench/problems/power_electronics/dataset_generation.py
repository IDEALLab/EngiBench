"""Generate reproducible Power Electronics v1 dataset shards from pinned v0 rows."""

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Protocol

from datasets import Dataset
from datasets import load_dataset
import numpy as np

from engibench.problems.power_electronics.utils.ngspice import NgSpiceIdentity
from engibench.problems.power_electronics.v1 import PowerElectronics
from engibench.problems.power_electronics.v1 import PowerElectronicsSimulationResult
from engibench.problems.power_electronics.v1 import SOURCE_VOLTAGE
from engibench.problems.power_electronics.v1 import TARGET_DC_GAIN

SOURCE_DATASET_ID = "IDEALLab/power_electronics_v0"
SOURCE_DATASET_REVISION = "5c4adb2ec5cfc71794988b1297a7ff8ffe59daa5"
CANONICAL_NGSPICE_VERSION = "44.2"
CANONICAL_PLATFORM_SYSTEM = "Linux"
CANONICAL_PLATFORM_MACHINES = ("x86_64", "amd64")
DATASET_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class GitState:
    """Git provenance for the EngiBench checkout running the simulation."""

    commit: str
    dirty: bool


@dataclass(frozen=True)
class ContainerIdentity:
    """Identity of the Apptainer image supplied to the generation command."""

    path: str | None
    sha256: str | None


class PowerElectronicsRunner(Protocol):
    """Narrow simulator interface required by the dataset generator."""

    @property
    def simulator_identity(self) -> NgSpiceIdentity:
        """Return the simulator identity."""

    @property
    def config(self) -> Any:
        """Return the configuration containing the source netlist path."""

    def simulate_verbose(self, design: np.ndarray) -> PowerElectronicsSimulationResult:
        """Simulate one design with detailed outputs."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it entirely in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def design_sha256(design: Sequence[float]) -> str:
    """Return a stable identifier for a source design row."""
    serialized = json.dumps(list(design), allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def read_git_state(repo_root: Path) -> GitState:
    """Read the exact commit and dirty state of a Git checkout."""
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return GitState(commit=commit, dirty=bool(status.strip()))


def resolve_container_identity(container_image: str | None, *, allow_uncontainerized: bool) -> ContainerIdentity:
    """Resolve and hash the exact container image used for a generation run."""
    supplied_path = container_image or os.environ.get("APPTAINER_CONTAINER")
    if supplied_path is None:
        if allow_uncontainerized:
            return ContainerIdentity(path=None, sha256=None)
        raise RuntimeError(
            "A pinned Apptainer image is required. Run inside Apptainer, pass --container-image, "
            "or use --allow-uncontainerized for local development only."
        )

    resolved_path = Path(supplied_path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Container image does not exist: {resolved_path}")
    return ContainerIdentity(path=str(resolved_path), sha256=sha256_file(resolved_path))


def validate_backend(
    *,
    simulator_version: str,
    simulator_sha256: str,
    simulator_system: str,
    simulator_machine: str,
    container_identity: ContainerIdentity,
    expected_simulator_sha256: str | None,
    expected_container_sha256: str | None,
    allow_noncanonical_backend: bool,
) -> bool:
    """Fail before simulation unless the backend matches the frozen canonical identity."""
    mismatches = []
    if simulator_version != CANONICAL_NGSPICE_VERSION:
        mismatches.append(f"ngspice version {simulator_version!r} != {CANONICAL_NGSPICE_VERSION!r}")
    if simulator_system != CANONICAL_PLATFORM_SYSTEM:
        mismatches.append(f"platform system {simulator_system!r} != {CANONICAL_PLATFORM_SYSTEM!r}")
    if simulator_machine.lower() not in CANONICAL_PLATFORM_MACHINES:
        mismatches.append(f"platform machine {simulator_machine!r} is not x86_64")
    if expected_simulator_sha256 is None:
        mismatches.append("--expected-ngspice-sha256 was not supplied")
    elif simulator_sha256 != expected_simulator_sha256:
        mismatches.append(f"ngspice SHA-256 {simulator_sha256!r} != {expected_simulator_sha256!r}")
    if expected_container_sha256 is None:
        mismatches.append("--expected-container-sha256 was not supplied")
    elif container_identity.sha256 != expected_container_sha256:
        mismatches.append(f"container SHA-256 {container_identity.sha256!r} != {expected_container_sha256!r}")

    if mismatches and not allow_noncanonical_backend:
        details = "\n- ".join(mismatches)
        raise RuntimeError(f"Backend is not the frozen Power Electronics v1 backend:\n- {details}")
    return not mismatches


def select_indices(
    row_count: int,
    *,
    explicit_indices: Sequence[int] | None,
    limit: int | None,
    shard_index: int,
    num_shards: int,
) -> list[int]:
    """Select source row indices deterministically without changing their split."""
    if row_count < 0:
        raise ValueError("row_count must be non-negative")
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if explicit_indices is not None and (num_shards != 1 or shard_index != 0):
        raise ValueError("Explicit indices cannot be combined with sharding")

    if explicit_indices is None:
        indices = list(range(shard_index, row_count, num_shards))
    else:
        indices = list(explicit_indices)
        if len(indices) != len(set(indices)):
            raise ValueError("Explicit indices must be unique")
        if any(index < 0 or index >= row_count for index in indices):
            raise IndexError(f"Explicit indices must be in [0, {row_count})")

    return indices if limit is None else indices[:limit]


def _nullable_float(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _base_record(
    *,
    split: str,
    source_index: int,
    source_row: Mapping[str, Any],
    git_state: GitState,
    source_revision: str,
    simulator_identity: NgSpiceIdentity,
    container_identity: ContainerIdentity,
    netlist_sha256: str,
) -> dict[str, Any]:
    design = [float(value) for value in source_row["initial_design"]]
    return {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "split": split,
        "source_index": source_index,
        "design_sha256": design_sha256(design),
        "initial_design": design,
        "v0_DcGain": float(source_row["DcGain"]),
        "v0_Voltage_Ripple": float(source_row["Voltage_Ripple"]),
        "source_dataset_id": SOURCE_DATASET_ID,
        "source_dataset_revision": source_revision,
        "problem_version": PowerElectronics.version,
        "engibench_git_commit": git_state.commit,
        "netlist_sha256": netlist_sha256,
        "simulator_version": simulator_identity.version,
        "simulator_sha256": simulator_identity.executable_sha256,
        "simulator_platform_system": simulator_identity.platform_system,
        "simulator_platform_machine": simulator_identity.platform_machine,
        "container_sha256": container_identity.sha256,
    }


def simulation_record(
    *,
    split: str,
    source_index: int,
    source_row: Mapping[str, Any],
    result: PowerElectronicsSimulationResult,
    git_state: GitState,
    source_revision: str,
    container_identity: ContainerIdentity,
    netlist_sha256: str,
) -> dict[str, Any]:
    """Build one successful or measurement-invalid output record."""
    record = _base_record(
        split=split,
        source_index=source_index,
        source_row=source_row,
        git_state=git_state,
        source_revision=source_revision,
        simulator_identity=result.simulator_identity,
        container_identity=container_identity,
        netlist_sha256=netlist_sha256,
    )
    record.update(
        {
            "output_voltage_mean": _nullable_float(result.output_voltage_mean),
            "output_voltage_peak_to_peak": _nullable_float(result.output_voltage_peak_to_peak),
            "dc_gain": _nullable_float(result.dc_gain),
            "dc_gain_error": _nullable_float(result.dc_gain_error),
            "relative_voltage_ripple": _nullable_float(result.relative_voltage_ripple),
            "simulation_valid": result.simulation_valid,
            "objectives_valid": result.objectives_valid,
            "simulation_status": result.status,
            "error_type": None,
            "error_message": None,
        }
    )
    return record


def error_record(
    *,
    split: str,
    source_index: int,
    source_row: Mapping[str, Any],
    error: Exception,
    git_state: GitState,
    source_revision: str,
    simulator_identity: NgSpiceIdentity,
    container_identity: ContainerIdentity,
    netlist_sha256: str,
) -> dict[str, Any]:
    """Build a row that preserves a simulator failure instead of dropping the design."""
    record = _base_record(
        split=split,
        source_index=source_index,
        source_row=source_row,
        git_state=git_state,
        source_revision=source_revision,
        simulator_identity=simulator_identity,
        container_identity=container_identity,
        netlist_sha256=netlist_sha256,
    )
    record.update(
        {
            "output_voltage_mean": None,
            "output_voltage_peak_to_peak": None,
            "dc_gain": None,
            "dc_gain_error": None,
            "relative_voltage_ripple": None,
            "simulation_valid": False,
            "objectives_valid": False,
            "simulation_status": "simulation_error",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
    )
    return record


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            json.dump(value, output_file, allow_nan=False, indent=2, sort_keys=True)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(json.dumps(record, allow_nan=False, separators=(",", ":")))
                output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def generate_shard(
    *,
    dataset: Dataset,
    split: str,
    indices: Sequence[int],
    output_path: Path,
    work_dir: Path,
    source_revision: str,
    git_state: GitState,
    container_identity: ContainerIdentity,
    ngspice_path: str | None,
    expected_simulator_sha256: str | None,
    expected_container_sha256: str | None,
    allow_noncanonical_backend: bool,
    overwrite: bool,
    problem_factory: Callable[..., PowerElectronicsRunner] = PowerElectronics,
) -> dict[str, Any]:
    """Simulate selected v0 rows and atomically write one v1 JSONL shard plus manifest."""
    manifest_path = output_path.with_suffix(f"{output_path.suffix}.manifest.json")
    if not overwrite and (output_path.exists() or manifest_path.exists()):
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path} or {manifest_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    problem = problem_factory(target_dir=str(work_dir), ngspice_path=ngspice_path)
    simulator_identity = problem.simulator_identity
    netlist_path = Path(problem.config.original_netlist_path)
    netlist_sha256 = sha256_file(netlist_path)
    canonical_backend_validated = validate_backend(
        simulator_version=simulator_identity.version,
        simulator_sha256=simulator_identity.executable_sha256,
        simulator_system=simulator_identity.platform_system,
        simulator_machine=simulator_identity.platform_machine,
        container_identity=container_identity,
        expected_simulator_sha256=expected_simulator_sha256,
        expected_container_sha256=expected_container_sha256,
        allow_noncanonical_backend=allow_noncanonical_backend,
    )
    records: list[dict[str, Any]] = []
    for source_index in indices:
        source_row = dataset[source_index]
        try:
            result = problem.simulate_verbose(np.asarray(source_row["initial_design"], dtype=np.float64))
            record = simulation_record(
                split=split,
                source_index=source_index,
                source_row=source_row,
                result=result,
                git_state=git_state,
                source_revision=source_revision,
                container_identity=container_identity,
                netlist_sha256=netlist_sha256,
            )
        except Exception as error:  # noqa: BLE001 - failures are dataset rows and must remain visible
            record = error_record(
                split=split,
                source_index=source_index,
                source_row=source_row,
                error=error,
                git_state=git_state,
                source_revision=source_revision,
                simulator_identity=simulator_identity,
                container_identity=container_identity,
                netlist_sha256=netlist_sha256,
            )
        records.append(record)

    _atomic_write_jsonl(output_path, records)
    output_sha256 = sha256_file(output_path)
    status_counts: dict[str, int] = {}
    for record in records:
        status = str(record["simulation_status"])
        status_counts[status] = status_counts.get(status, 0) + 1

    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset_id": SOURCE_DATASET_ID,
            "revision": source_revision,
            "split": split,
            "source_row_count": len(dataset),
            "selected_indices": list(indices),
            "dataset_fingerprint": dataset._fingerprint,  # noqa: SLF001 - HF exposes no public immutable fingerprint API
        },
        "problem": {
            "name": "power_electronics",
            "version": PowerElectronics.version,
            "source_voltage": SOURCE_VOLTAGE,
            "target_dc_gain": TARGET_DC_GAIN,
            "objectives": [
                {"name": "dc_gain_error", "direction": "minimize", "definition": "abs(dc_gain - 0.25)"},
                {
                    "name": "relative_voltage_ripple",
                    "direction": "minimize",
                    "definition": "output_voltage_peak_to_peak / abs(output_voltage_mean)",
                },
            ],
        },
        "engibench": asdict(git_state),
        "simulator": asdict(simulator_identity),
        "container": asdict(container_identity),
        "netlist": {"path": str(netlist_path), "sha256": netlist_sha256},
        "canonical_backend_validated": canonical_backend_validated,
        "runtime": {
            "python": sys.version,
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
        },
        "output": {
            "path": str(output_path.resolve()),
            "sha256": output_sha256,
            "record_count": len(records),
            "status_counts": status_counts,
        },
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


def _parse_indices(value: str | None) -> list[int] | None:
    if value is None:
        return None
    if not value.strip():
        raise argparse.ArgumentTypeError("--indices cannot be empty")
    try:
        return [int(index) for index in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("--indices must be comma-separated integers") from error


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for reproducible shard generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL shard")
    parser.add_argument("--work-dir", type=Path, required=True, help="Scratch directory for ngspice files")
    parser.add_argument("--source-revision", default=SOURCE_DATASET_REVISION)
    parser.add_argument("--indices", help="Comma-separated source row indices; cannot be combined with sharding")
    parser.add_argument("--limit", type=int, help="Limit rows after deterministic index selection")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--ngspice-path")
    parser.add_argument(
        "--expected-ngspice-sha256",
        help="Frozen SHA-256 of the ngspice executable; required for canonical generation",
    )
    parser.add_argument(
        "--expected-container-sha256",
        help="Frozen SHA-256 of the .sif image; required for canonical generation",
    )
    parser.add_argument("--container-image", help="Exact .sif path; defaults to APPTAINER_CONTAINER")
    parser.add_argument(
        "--allow-uncontainerized",
        action="store_true",
        help="Permit local development outside Apptainer; never use for the canonical dataset",
    )
    parser.add_argument(
        "--allow-dirty-code",
        action="store_true",
        help="Permit a dirty EngiBench checkout; never use for the canonical dataset",
    )
    parser.add_argument(
        "--allow-noncanonical-backend",
        action="store_true",
        help="Permit a backend other than ngspice 44.2 on Linux x86_64; comparison/development only",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate one deterministic Power Electronics v1 dataset shard."""
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[3]
    git_state = read_git_state(repo_root)
    if git_state.dirty and not args.allow_dirty_code:
        raise RuntimeError("EngiBench checkout is dirty; commit it or pass --allow-dirty-code for local development only.")
    container_identity = resolve_container_identity(
        args.container_image,
        allow_uncontainerized=args.allow_uncontainerized,
    )
    dataset = load_dataset(
        SOURCE_DATASET_ID,
        revision=args.source_revision,
        split=args.split,
    )
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected a Dataset for split {args.split!r}, got {type(dataset).__name__}")
    indices = select_indices(
        len(dataset),
        explicit_indices=_parse_indices(args.indices),
        limit=args.limit,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    manifest = generate_shard(
        dataset=dataset,
        split=args.split,
        indices=indices,
        output_path=args.output,
        work_dir=args.work_dir,
        source_revision=args.source_revision,
        git_state=git_state,
        container_identity=container_identity,
        ngspice_path=args.ngspice_path,
        expected_simulator_sha256=args.expected_ngspice_sha256,
        expected_container_sha256=args.expected_container_sha256,
        allow_noncanonical_backend=args.allow_noncanonical_backend,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest["output"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
