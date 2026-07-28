"""Retrieve and validate the exact solver-native MTO2D gamma fields.

The Zaratan source tree is treated as read-only. Retrieval uses ``rsync`` with
an explicit ``--files-from`` manifest, so only ``app/200/gamma`` is requested
for the 5,666 cases in the pinned ``index_5666.npy`` file. Interrupted
transfers can be resumed by rerunning the same command.

Examples:
    Prepare the authoritative transfer manifest::

        python -m engibench.problems.mto2d.model.retrieve_native_gammas prepare \
            --ids-npy /path/to/index_5666.npy \
            --output-dir /path/to/source-gammas

    Fetch the selected fields with interactive Zaratan authentication::

        python -m engibench.problems.mto2d.model.retrieve_native_gammas fetch \
            --output-dir /path/to/source-gammas

    Validate every downloaded field and produce a retry manifest::

        python -m engibench.problems.mto2d.model.retrieve_native_gammas validate \
            --output-dir /path/to/source-gammas
"""

import argparse
import csv
from dataclasses import asdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Any

import numpy as np
import numpy.typing as npt

from engibench.problems.mto2d.model.dataset import RAW_ROW_COUNT
from engibench.problems.mto2d.model.dataset import RAW_SHA256
from engibench.problems.mto2d.model.design_io import DESIGN_CELL_COUNT
from engibench.problems.mto2d.model.design_io import FIXED_CELL_COUNT
from engibench.problems.mto2d.model.design_io import GAMMA_CELL_COUNT
from engibench.problems.mto2d.model.design_io import parse_internal_field

DEFAULT_REMOTE = "adrake17@login.zaratan.umd.edu"
DEFAULT_REMOTE_ROOT = "/home/adrake17/scratch/warmstart/sd/template"
DEFAULT_LOGIN_NODE = "login-1.zaratan.umd.edu"
DEFAULT_HOST_KEY_ALIAS = "login.zaratan.umd.edu"
DEFAULT_TIMEOUT_SECONDS = 120

SOURCE_CASES_FILENAME = "source_cases.csv"
TRANSFER_FILES_FILENAME = "gamma-files.txt"
RETRIEVAL_METADATA_FILENAME = "retrieval_metadata.json"
VALIDATION_RECORDS_FILENAME = "gamma-validation.jsonl"
VALIDATION_SUMMARY_FILENAME = "gamma-validation-summary.json"
RETRY_FILES_FILENAME = "gamma-retry-files.txt"

PINNED_INDEX_SHA256 = RAW_SHA256["source_case_id"]
REFERENCE_CASE_ID = 9130
REFERENCE_CASE_SHA256 = "12288cce0e1cfe1397b6470a45ba8cfc8bfa63f35d44512eb568fb8d24217e3d"


@dataclass(frozen=True)
class SourceCase:
    """One selected source case in published row order."""

    source_row_index: int
    source_case_id: int
    relative_path: str


@dataclass(frozen=True)
class ValidationRecord:
    """Validation result for one expected native gamma field."""

    source_row_index: int
    source_case_id: int
    relative_path: str
    status: str
    byte_size: int | None
    sha256: str | None
    minimum: float | None
    maximum: float | None
    all_cell_mean: float | None
    fixed_tail_valid: bool | None
    error: str | None


class _GammaValidationError(ValueError):
    def __init__(self, message: str, *, fixed_tail_valid: bool | None) -> None:
        super().__init__(message)
        self.fixed_tail_valid = fixed_tail_valid


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_gamma_path(case_id: int) -> str:
    return f"2Dheatsink_{case_id}/app/200/gamma"


def load_source_ids(
    path: str | Path,
    *,
    verify_pinned_hash: bool = True,
) -> npt.NDArray[np.int64]:
    """Load and validate the authoritative source-case ID array."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source-case ID array does not exist: {source}")
    if verify_pinned_hash:
        actual_hash = _sha256(source)
        if actual_hash != PINNED_INDEX_SHA256:
            raise ValueError(f"source-case ID SHA-256 is {actual_hash}; expected pinned digest {PINNED_INDEX_SHA256}")

    values = np.load(source, allow_pickle=False)
    if values.ndim != 1:
        raise ValueError(f"source-case ID array must be one-dimensional; got shape {values.shape}")
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError(f"source-case ID array must contain integers; got {values.dtype}")
    case_ids = np.asarray(values, dtype=np.int64)
    if verify_pinned_hash and case_ids.shape != (RAW_ROW_COUNT,):
        raise ValueError(f"pinned source-case ID array must contain {RAW_ROW_COUNT} values; got {case_ids.size}")
    if np.any(case_ids < 0):
        raise ValueError("source-case IDs must be non-negative")
    if np.unique(case_ids).size != case_ids.size:
        raise ValueError("source-case IDs must be unique")
    return case_ids


def source_cases(case_ids: npt.NDArray[np.int64]) -> list[SourceCase]:
    """Create manifest entries in published source-row order."""
    return [
        SourceCase(
            source_row_index=row_index,
            source_case_id=int(case_id),
            relative_path=_relative_gamma_path(int(case_id)),
        )
        for row_index, case_id in enumerate(case_ids)
    ]


def prepare_manifest(
    ids_path: str | Path,
    output_dir: str | Path,
    *,
    verify_pinned_hash: bool = True,
) -> dict[str, Any]:
    """Write deterministic CSV and rsync manifests for selected source cases."""
    source = Path(ids_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    case_ids = load_source_ids(source, verify_pinned_hash=verify_pinned_hash)
    cases = source_cases(case_ids)

    cases_path = destination / SOURCE_CASES_FILENAME
    with cases_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source_row_index", "source_case_id", "relative_path"))
        writer.writeheader()
        writer.writerows(asdict(case) for case in cases)

    transfer_path = destination / TRANSFER_FILES_FILENAME
    transfer_path.write_text("".join(f"{case.relative_path}\n" for case in cases), encoding="utf-8")

    metadata = {
        "schema": "engibench-mto2d-native-gamma-retrieval-v1",
        "source_index_path": str(source),
        "source_index_sha256": _sha256(source),
        "source_index_is_pinned": bool(verify_pinned_hash),
        "row_count": len(cases),
        "source_case_id_min": int(case_ids.min()) if len(case_ids) else None,
        "source_case_id_max": int(case_ids.max()) if len(case_ids) else None,
        "remote_layout": "2Dheatsink_<source_case_id>/app/200/gamma",
        "source_cases_file": SOURCE_CASES_FILENAME,
        "transfer_files_file": TRANSFER_FILES_FILENAME,
    }
    (destination / RETRIEVAL_METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def read_source_cases(path: str | Path) -> list[SourceCase]:
    """Read and revalidate a prepared source-case CSV manifest."""
    manifest = Path(path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"source-case manifest does not exist: {manifest}")
    cases: list[SourceCase] = []
    with manifest.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_fields = {"source_row_index", "source_case_id", "relative_path"}
        if set(reader.fieldnames or ()) != expected_fields:
            raise ValueError(f"source-case manifest columns must be {sorted(expected_fields)}")
        for row in reader:
            row_index = int(row["source_row_index"])
            case_id = int(row["source_case_id"])
            relative_path = row["relative_path"]
            expected_path = _relative_gamma_path(case_id)
            if relative_path != expected_path:
                raise ValueError(f"source case {case_id} has relative path {relative_path!r}; expected {expected_path!r}")
            cases.append(SourceCase(row_index, case_id, relative_path))

    expected_positions = list(range(len(cases)))
    actual_positions = [case.source_row_index for case in cases]
    if actual_positions != expected_positions:
        raise ValueError("source-case manifest must be ordered by contiguous source_row_index")
    case_ids = [case.source_case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("source-case manifest contains duplicate source_case_id values")
    return cases


def _validate_transfer_list(
    path: Path,
    cases: list[SourceCase],
    *,
    require_all: bool,
) -> list[str]:
    paths = path.read_text(encoding="utf-8").splitlines()
    if any(not relative_path for relative_path in paths):
        raise ValueError(f"transfer list contains a blank path: {path}")
    if len(paths) != len(set(paths)):
        raise ValueError(f"transfer list contains duplicate paths: {path}")
    allowed_paths = {case.relative_path for case in cases}
    unexpected_paths = sorted(set(paths) - allowed_paths)
    if unexpected_paths:
        raise ValueError(f"transfer list contains paths outside the source-case manifest: {unexpected_paths[:3]}")
    if require_all and paths != [case.relative_path for case in cases]:
        raise ValueError("full transfer list must match the source-case manifest in source-row order")
    return paths


def build_rsync_command(  # noqa: PLR0913
    *,
    files_from: str | Path,
    output_dir: str | Path,
    remote: str = DEFAULT_REMOTE,
    remote_root: str = DEFAULT_REMOTE_ROOT,
    hostname: str = DEFAULT_LOGIN_NODE,
    host_key_alias: str = DEFAULT_HOST_KEY_ALIAS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    bwlimit_kib: int | None = None,
    checksum: bool = False,
    dry_run: bool = False,
    rsync_executable: str = "rsync",
) -> list[str]:
    """Build the remote-read-only rsync command used by ``fetch``."""
    transfer_list = Path(files_from).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if bwlimit_kib is not None and bwlimit_kib <= 0:
        raise ValueError("bwlimit_kib must be positive when supplied")

    ssh_command = shlex.join(
        [
            "ssh",
            "-4",
            "-o",
            f"Hostname={hostname}",
            "-o",
            f"HostKeyAlias={host_key_alias}",
            "-o",
            "ControlMaster=no",
            "-o",
            "ControlPath=none",
            "-o",
            "ConnectTimeout=20",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=4",
        ]
    )
    command = [
        rsync_executable,
        "-rltz",
        "--relative",
        f"--files-from={transfer_list}",
        "--partial",
        f"--timeout={timeout_seconds}",
        "--progress",
        "--stats",
    ]
    if bwlimit_kib is not None:
        command.append(f"--bwlimit={bwlimit_kib}")
    if checksum:
        command.append("--checksum")
    if dry_run:
        command.append("--dry-run")
    command.extend(
        [
            "-e",
            ssh_command,
            f"{remote}:{remote_root.rstrip('/')}/",
            f"{destination}/",
        ]
    )
    return command


def fetch(  # noqa: PLR0913
    *,
    output_dir: str | Path,
    retry_only: bool = False,
    remote: str = DEFAULT_REMOTE,
    remote_root: str = DEFAULT_REMOTE_ROOT,
    hostname: str = DEFAULT_LOGIN_NODE,
    host_key_alias: str = DEFAULT_HOST_KEY_ALIAS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    bwlimit_kib: int | None = None,
    checksum: bool = False,
    dry_run: bool = False,
    rsync_executable: str = "rsync",
) -> int:
    """Run one resumable manifest-driven rsync transfer."""
    destination = Path(output_dir).expanduser().resolve()
    files_from = destination / (RETRY_FILES_FILENAME if retry_only else TRANSFER_FILES_FILENAME)
    if not files_from.is_file():
        raise FileNotFoundError(f"transfer list does not exist: {files_from}")
    cases = read_source_cases(destination / SOURCE_CASES_FILENAME)
    transfer_paths = _validate_transfer_list(files_from, cases, require_all=not retry_only)
    if not transfer_paths:
        print(f"No files listed in {files_from}; nothing to retrieve.")
        return 0
    if shutil.which(rsync_executable) is None:
        raise FileNotFoundError(f"rsync executable is not available: {rsync_executable}")

    command = build_rsync_command(
        files_from=files_from,
        output_dir=destination,
        remote=remote,
        remote_root=remote_root,
        hostname=hostname,
        host_key_alias=host_key_alias,
        timeout_seconds=timeout_seconds,
        bwlimit_kib=bwlimit_kib,
        checksum=checksum,
        dry_run=dry_run,
        rsync_executable=rsync_executable,
    )
    print("Running remote-read-only transfer:")
    print(f"  {shlex.join(command)}")
    print("Rerun the same command after any timeout; completed files are skipped and partial files are reusable.")
    return subprocess.run(command, check=False).returncode


def _validated_gamma_values(
    content: str,
    *,
    case_id: int,
    digest: str,
) -> npt.NDArray[np.float64]:
    values = parse_internal_field(content, expected_count=GAMMA_CELL_COUNT)
    fixed_tail_valid = bool(
        np.array_equal(
            values[DESIGN_CELL_COUNT:],
            np.ones(FIXED_CELL_COUNT, dtype=values.dtype),
        )
    )
    if not fixed_tail_valid:
        raise _GammaValidationError(
            "the final 6,400 fixed/non-design cells are not all fluid",
            fixed_tail_valid=False,
        )
    if np.any((values < 0.0) | (values > 1.0)):
        raise _GammaValidationError("gamma values lie outside [0, 1]", fixed_tail_valid=True)
    if case_id == REFERENCE_CASE_ID and digest != REFERENCE_CASE_SHA256:
        raise _GammaValidationError(
            f"reference case {REFERENCE_CASE_ID} SHA-256 is {digest}; expected {REFERENCE_CASE_SHA256}",
            fixed_tail_valid=True,
        )
    return values


def validate_gamma(case: SourceCase, output_dir: str | Path) -> ValidationRecord:
    """Validate one native OpenFOAM gamma file and compute its local digest."""
    path = Path(output_dir).expanduser().resolve() / case.relative_path
    if not path.is_file():
        return ValidationRecord(
            **asdict(case),
            status="missing",
            byte_size=None,
            sha256=None,
            minimum=None,
            maximum=None,
            all_cell_mean=None,
            fixed_tail_valid=None,
            error="file is missing",
        )

    byte_size: int | None = None
    digest: str | None = None
    try:
        byte_size = path.stat().st_size
        digest = _sha256(path)
        content = path.read_text(encoding="ascii")
        values = _validated_gamma_values(content, case_id=case.source_case_id, digest=digest)
    except (OSError, ValueError) as error:
        fixed_tail_valid = error.fixed_tail_valid if isinstance(error, _GammaValidationError) else None
        return ValidationRecord(
            **asdict(case),
            status="invalid",
            byte_size=byte_size,
            sha256=digest,
            minimum=None,
            maximum=None,
            all_cell_mean=None,
            fixed_tail_valid=fixed_tail_valid,
            error=str(error),
        )

    return ValidationRecord(
        **asdict(case),
        status="valid",
        byte_size=byte_size,
        sha256=digest,
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        all_cell_mean=float(np.mean(values)),
        fixed_tail_valid=True,
        error=None,
    )


def validate_all(
    output_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate all expected fields and write records, summary, and retries."""
    destination = Path(output_dir).expanduser().resolve()
    cases = read_source_cases(manifest_path or destination / SOURCE_CASES_FILENAME)
    records_path = destination / VALIDATION_RECORDS_FILENAME
    records_temporary = records_path.with_suffix(records_path.suffix + ".tmp")
    retry_paths: list[str] = []
    counts = {"valid": 0, "missing": 0, "invalid": 0}
    byte_count = 0

    with records_temporary.open("w", encoding="utf-8") as handle:
        for position, case in enumerate(cases, start=1):
            record = validate_gamma(case, destination)
            counts[record.status] += 1
            byte_count += record.byte_size or 0
            if record.status != "valid":
                retry_paths.append(record.relative_path)
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
            if position % 100 == 0 or position == len(cases):
                print(
                    f"Validated {position:,}/{len(cases):,}: "
                    f"{counts['valid']:,} valid, {counts['missing']:,} missing, {counts['invalid']:,} invalid"
                )
    records_temporary.replace(records_path)

    (destination / RETRY_FILES_FILENAME).write_text(
        "".join(f"{path}\n" for path in retry_paths),
        encoding="utf-8",
    )
    summary = {
        "schema": "engibench-mto2d-native-gamma-validation-v1",
        "expected": len(cases),
        **counts,
        "validated_bytes": byte_count,
        "complete": counts["valid"] == len(cases),
        "records_file": VALIDATION_RECORDS_FILENAME,
        "retry_files_file": RETRY_FILES_FILENAME,
    }
    (destination / VALIDATION_SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="prepare the selected-case and rsync manifests")
    prepare.add_argument("--ids-npy", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument(
        "--verify-pinned-hash",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require the pinned IDEALLab/MTO-2D index_5666.npy digest",
    )

    fetch_parser = subparsers.add_parser("fetch", help="fetch selected gamma fields with resumable rsync")
    fetch_parser.add_argument("--output-dir", type=Path, required=True)
    fetch_parser.add_argument("--retry-only", action="store_true")
    fetch_parser.add_argument("--remote", default=DEFAULT_REMOTE)
    fetch_parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    fetch_parser.add_argument("--hostname", default=DEFAULT_LOGIN_NODE)
    fetch_parser.add_argument("--host-key-alias", default=DEFAULT_HOST_KEY_ALIAS)
    fetch_parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    fetch_parser.add_argument("--bwlimit-kib", type=int)
    fetch_parser.add_argument("--checksum", action="store_true")
    fetch_parser.add_argument("--dry-run", action="store_true")
    fetch_parser.add_argument("--rsync-executable", default="rsync")

    validate = subparsers.add_parser("validate", help="validate local fields and write a retry manifest")
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--manifest", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the native-gamma retrieval command line interface."""
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        metadata = prepare_manifest(
            args.ids_npy,
            args.output_dir,
            verify_pinned_hash=args.verify_pinned_hash,
        )
        print(f"Prepared {metadata['row_count']:,} exact gamma paths in {Path(args.output_dir).expanduser().resolve()}.")
        return 0
    if args.command == "fetch":
        return fetch(
            output_dir=args.output_dir,
            retry_only=args.retry_only,
            remote=args.remote,
            remote_root=args.remote_root,
            hostname=args.hostname,
            host_key_alias=args.host_key_alias,
            timeout_seconds=args.timeout_seconds,
            bwlimit_kib=args.bwlimit_kib,
            checksum=args.checksum,
            dry_run=args.dry_run,
            rsync_executable=args.rsync_executable,
        )

    summary = validate_all(args.output_dir, manifest_path=args.manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
