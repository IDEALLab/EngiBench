"""Tests for the MTO2D native-gamma retrieval workflow."""

import json
from pathlib import Path

import numpy as np
import pytest

from engibench.problems.mto2d.model.design_io import DESIGN_CELL_COUNT
from engibench.problems.mto2d.model.design_io import FIXED_CELL_COUNT
from engibench.problems.mto2d.model.retrieve_native_gammas import build_rsync_command
from engibench.problems.mto2d.model.retrieve_native_gammas import fetch
from engibench.problems.mto2d.model.retrieve_native_gammas import load_source_ids
from engibench.problems.mto2d.model.retrieve_native_gammas import prepare_manifest
from engibench.problems.mto2d.model.retrieve_native_gammas import read_source_cases
from engibench.problems.mto2d.model.retrieve_native_gammas import RETRY_FILES_FILENAME
from engibench.problems.mto2d.model.retrieve_native_gammas import SOURCE_CASES_FILENAME
from engibench.problems.mto2d.model.retrieve_native_gammas import TRANSFER_FILES_FILENAME
from engibench.problems.mto2d.model.retrieve_native_gammas import validate_all
from engibench.problems.mto2d.model.retrieve_native_gammas import VALIDATION_RECORDS_FILENAME


def _write_gamma(path: Path, *, fixed_tail: float = 1.0) -> None:
    values = np.concatenate(
        (
            np.full(DESIGN_CELL_COUNT, 0.25, dtype=np.float64),
            np.full(FIXED_CELL_COUNT, fixed_tail, dtype=np.float64),
        )
    )
    rendered = (
        "internalField nonuniform List<scalar>\n"
        f"{values.size}\n"
        "(\n" + "\n".join(str(value) for value in values) + "\n)\n;\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="ascii")


def test_prepare_manifest_preserves_source_row_order(tmp_path: Path) -> None:
    expected_ids = [9, 3, 17]
    ids_path = tmp_path / "ids.npy"
    np.save(ids_path, np.array(expected_ids, dtype=np.int64), allow_pickle=False)
    output = tmp_path / "fields"

    metadata = prepare_manifest(ids_path, output, verify_pinned_hash=False)
    cases = read_source_cases(output / SOURCE_CASES_FILENAME)

    assert metadata["row_count"] == len(expected_ids)
    assert [case.source_row_index for case in cases] == [0, 1, 2]
    assert [case.source_case_id for case in cases] == expected_ids
    assert (output / TRANSFER_FILES_FILENAME).read_text(encoding="utf-8").splitlines() == [
        "2Dheatsink_9/app/200/gamma",
        "2Dheatsink_3/app/200/gamma",
        "2Dheatsink_17/app/200/gamma",
    ]
    np.testing.assert_array_equal(load_source_ids(ids_path, verify_pinned_hash=False), expected_ids)


def test_build_rsync_command_is_remote_read_only_and_resumable(tmp_path: Path) -> None:
    files_from = tmp_path / TRANSFER_FILES_FILENAME
    files_from.write_text("2Dheatsink_7/app/200/gamma\n", encoding="utf-8")

    command = build_rsync_command(
        files_from=files_from,
        output_dir=tmp_path,
        bwlimit_kib=128,
        checksum=True,
        dry_run=True,
    )
    joined = " ".join(command)

    assert "--files-from=" in joined
    assert "--partial" in command
    assert "--checksum" in command
    assert "--dry-run" in command
    assert "--bwlimit=128" in command
    assert "Hostname=login-1.zaratan.umd.edu" in joined
    assert "ControlMaster=no" in joined
    assert "--delete" not in joined
    assert "--remove-source-files" not in joined
    assert command[-2] == ("adrake17@login.zaratan.umd.edu:/home/adrake17/scratch/warmstart/sd/template/")
    assert command[-1] == f"{tmp_path.resolve()}/"


def test_fetch_rejects_paths_outside_prepared_manifest(tmp_path: Path) -> None:
    ids_path = tmp_path / "ids.npy"
    np.save(ids_path, np.array([7], dtype=np.int64), allow_pickle=False)
    output = tmp_path / "fields"
    prepare_manifest(ids_path, output, verify_pinned_hash=False)
    (output / TRANSFER_FILES_FILENAME).write_text("../../other-file\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside the source-case manifest"):
        fetch(output_dir=output, dry_run=True, rsync_executable="true")


def test_validate_all_records_valid_missing_and_retry_files(tmp_path: Path) -> None:
    ids_path = tmp_path / "ids.npy"
    np.save(ids_path, np.array([7, 8], dtype=np.int64), allow_pickle=False)
    output = tmp_path / "fields"
    prepare_manifest(ids_path, output, verify_pinned_hash=False)
    _write_gamma(output / "2Dheatsink_7/app/200/gamma")

    summary = validate_all(output)
    records = [json.loads(line) for line in (output / VALIDATION_RECORDS_FILENAME).read_text(encoding="utf-8").splitlines()]

    assert summary == {
        "schema": "engibench-mto2d-native-gamma-validation-v1",
        "expected": 2,
        "valid": 1,
        "missing": 1,
        "invalid": 0,
        "validated_bytes": records[0]["byte_size"],
        "complete": False,
        "records_file": "gamma-validation.jsonl",
        "retry_files_file": "gamma-retry-files.txt",
    }
    assert [record["status"] for record in records] == ["valid", "missing"]
    assert (output / RETRY_FILES_FILENAME).read_text(encoding="utf-8") == "2Dheatsink_8/app/200/gamma\n"


def test_validate_all_retries_invalid_fixed_tail(tmp_path: Path) -> None:
    ids_path = tmp_path / "ids.npy"
    np.save(ids_path, np.array([7], dtype=np.int64), allow_pickle=False)
    output = tmp_path / "fields"
    prepare_manifest(ids_path, output, verify_pinned_hash=False)
    _write_gamma(output / "2Dheatsink_7/app/200/gamma", fixed_tail=0.0)

    summary = validate_all(output)
    record = json.loads((output / VALIDATION_RECORDS_FILENAME).read_text(encoding="utf-8"))

    assert summary["invalid"] == 1
    assert record["status"] == "invalid"
    assert record["fixed_tail_valid"] is False
    assert "fixed/non-design cells" in record["error"]
