import hashlib
import json
from pathlib import Path

from datasets import load_from_disk
import numpy as np
import pytest

from engibench.problems.mto2d.model.dataset import dataset_features
from engibench.problems.mto2d.model.dataset import LEGACY_SPLIT_ALGORITHM
from engibench.problems.mto2d.model.dataset import LEGACY_SPLIT_FRACTIONS
from engibench.problems.mto2d.model.dataset import legacy_split_indices
from engibench.problems.mto2d.model.dataset import LEGACY_SPLIT_POLICY
from engibench.problems.mto2d.model.dataset import RAW_FILENAMES
from engibench.problems.mto2d.model.design_io import DESIGN_CELL_COUNT
from engibench.problems.mto2d.model.design_io import FIXED_CELL_COUNT
from engibench.problems.mto2d.model.design_io import GAMMA_CELL_COUNT
from engibench.problems.mto2d.model.design_io import gamma_to_half_design
from engibench.problems.mto2d.model.reformat_native_gamma_dataset import CONVERSION_MANIFEST_FILENAME
from engibench.problems.mto2d.model.reformat_native_gamma_dataset import convert_and_save
from engibench.problems.mto2d.model.reformat_native_gamma_dataset import DATASET_CARD_FILENAME
from engibench.problems.mto2d.model.reformat_native_gamma_dataset import NATIVE_SOURCE_PROVENANCE
from engibench.problems.mto2d.model.reformat_native_gamma_dataset import native_source_row
from engibench.problems.mto2d.model.retrieve_native_gammas import prepare_manifest
from engibench.problems.mto2d.model.retrieve_native_gammas import validate_all
from engibench.problems.mto2d.model.retrieve_native_gammas import VALIDATION_RECORDS_FILENAME


def _gamma_values(row_index: int) -> np.ndarray:
    first = np.linspace(0.0, 0.8, DESIGN_CELL_COUNT, dtype=np.float64)
    design = np.mod(first + row_index * 0.03, 0.81)
    return np.concatenate((design, np.ones(FIXED_CELL_COUNT, dtype=np.float64)))


def _write_gamma(path: Path, values: np.ndarray) -> None:
    rendered = (
        "FoamFile { format ascii; }\n"
        "internalField nonuniform List<scalar>\n"
        f"{GAMMA_CELL_COUNT}\n(\n" + "\n".join(format(float(value), ".17g") for value in values) + "\n)\n;\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="ascii")


def _fixture_inputs(tmp_path: Path, *, row_count: int = 4) -> tuple[Path, Path, dict[str, np.ndarray]]:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    case_ids = np.arange(100, 100 + row_count, dtype=np.int64)
    arrays = {
        "conditions": np.column_stack(
            (
                np.linspace(-0.095, -0.025, row_count),
                np.linspace(50.0, 75.0, row_count),
                np.linspace(0.25, 0.70, row_count),
            )
        ),
        "mean_temperature": np.linspace(20.0, 23.0, row_count),
        "power_dissipation": np.linspace(49.0, 74.0, row_count),
        "source_case_id": case_ids,
    }
    for key, array in arrays.items():
        np.save(raw_dir / RAW_FILENAMES[key], array, allow_pickle=False)

    ids_path = tmp_path / "ids.npy"
    np.save(ids_path, case_ids, allow_pickle=False)
    gamma_dir = tmp_path / "gammas"
    prepare_manifest(ids_path, gamma_dir, verify_pinned_hash=False)
    for row_index, case_id in enumerate(case_ids):
        _write_gamma(
            gamma_dir / f"2Dheatsink_{case_id}/app/200/gamma",
            _gamma_values(row_index),
        )
    assert validate_all(gamma_dir)["complete"] is True
    return raw_dir, gamma_dir, arrays


def test_convert_exact_native_fields_preserves_order_labels_and_provenance(tmp_path: Path) -> None:
    raw_dir, gamma_dir, arrays = _fixture_inputs(tmp_path, row_count=20)
    output = tmp_path / "native-dataset"

    converted = convert_and_save(
        gamma_dir,
        raw_dir,
        output,
        source_dataset="test/MTO-2D",
        source_revision="fixture",
        verify_hashes=False,
        cache_dir=tmp_path / "cache",
        writer_batch_size=2,
    )
    reloaded = load_from_disk(str(output))
    expected_splits = legacy_split_indices(len(arrays["source_case_id"]))

    assert {name: len(split) for name, split in converted.items()} == {
        name: len(positions) for name, positions in expected_splits.items()
    }
    for split_name, positions in expected_splits.items():
        split = reloaded[split_name]
        assert split.features == dataset_features()
        assert split["source_row_index"] == positions.tolist()
        assert split["source_case_id"] == arrays["source_case_id"][positions].tolist()
        assert split["design_is_exact"] == [True] * len(split)
        assert split["objectives_evaluated_on_design"] == [False] * len(split)
        assert set(split["design_provenance"]) <= {NATIVE_SOURCE_PROVENANCE}

    source_position = 0
    split_name = next(name for name, positions in expected_splits.items() if source_position in positions)
    split_positions = expected_splits[split_name].tolist()
    row = reloaded[split_name][split_positions.index(source_position)]
    gamma = _gamma_values(source_position)

    np.testing.assert_array_equal(
        np.asarray(row["optimal_design"], dtype=np.float32),
        gamma_to_half_design(gamma).reshape(-1),
    )
    assert row["mean_temperature"] == pytest.approx(arrays["mean_temperature"][source_position])
    assert row["power_dissipation"] == pytest.approx(arrays["power_dissipation"][source_position])
    assert row["volume_constraint_residual"] == pytest.approx(
        float(np.mean(gamma) - arrays["conditions"][source_position, 2])
    )
    assert reloaded[split_name].info.license in (None, "")
    manifest = json.loads((output / CONVERSION_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["publication_ready"] is False
    assert manifest["redistribution_rights"] == "unverified"
    assert manifest["metadata_hashes_verified"] is False
    assert manifest["metadata_sha256"] is None
    assert manifest["split_policy"] == LEGACY_SPLIT_POLICY
    assert manifest["split_fractions"] == list(LEGACY_SPLIT_FRACTIONS)
    assert manifest["split_algorithm"] == LEGACY_SPLIT_ALGORITHM
    copied_records = output / VALIDATION_RECORDS_FILENAME
    assert copied_records.is_file()
    assert manifest["gamma_validation_records_sha256"] == hashlib.sha256(copied_records.read_bytes()).hexdigest()
    card = (output / DATASET_CARD_FILENAME).read_text(encoding="utf-8")
    assert "Publication blocked" in card
    assert "different solver states" in " ".join(card.split())
    assert "historical pre-update" in manifest["residual_semantics"]["power_constraint_residual"]
    assert "exact post-update" in manifest["residual_semantics"]["volume_constraint_residual"]


def test_conversion_rejects_gamma_changed_after_validation(tmp_path: Path) -> None:
    raw_dir, gamma_dir, arrays = _fixture_inputs(tmp_path)
    first_id = int(arrays["source_case_id"][0])
    gamma_path = gamma_dir / f"2Dheatsink_{first_id}/app/200/gamma"
    gamma_path.write_text(gamma_path.read_text(encoding="ascii") + "\n", encoding="ascii")

    with pytest.raises(ValueError, match="SHA-256 changed"):
        convert_and_save(
            gamma_dir,
            raw_dir,
            tmp_path / "output",
            source_dataset="test/MTO-2D",
            source_revision="fixture",
            verify_hashes=False,
        )
    assert not (tmp_path / "output").exists()


def test_native_source_row_rejects_nonfluid_fixed_tail() -> None:
    invalid = _gamma_values(0)
    invalid[-1] = 0.0

    with pytest.raises(ValueError, match="fixed/non-design"):
        native_source_row(
            gamma_values=invalid,
            conditions=np.array([-0.05, 60.0, 0.4]),
            mean_temperature=20.0,
            power_dissipation=59.0,
            source_case_id=1,
            source_row_index=0,
            source_dataset="test/MTO-2D",
            source_revision="fixture",
        )


def test_atomic_save_cleans_temporary_output_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir, gamma_dir, _arrays = _fixture_inputs(tmp_path)
    output = tmp_path / "native-dataset"

    def fail_after_partial_write(_self, dataset_path, *args, **kwargs) -> None:
        partial = Path(dataset_path)
        partial.mkdir(parents=True, exist_ok=True)
        (partial / "partial").write_text("incomplete", encoding="utf-8")
        raise RuntimeError("injected save failure")

    monkeypatch.setattr("datasets.DatasetDict.save_to_disk", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="injected save failure"):
        convert_and_save(
            gamma_dir,
            raw_dir,
            output,
            source_dataset="test/MTO-2D",
            source_revision="fixture",
            verify_hashes=False,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_conversion_rejects_validation_record_case_order_mismatch(tmp_path: Path) -> None:
    raw_dir, gamma_dir, _arrays = _fixture_inputs(tmp_path)
    records_path = gamma_dir / VALIDATION_RECORDS_FILENAME
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    records[0]["source_case_id"], records[1]["source_case_id"] = (
        records[1]["source_case_id"],
        records[0]["source_case_id"],
    )
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="identity mismatch"):
        convert_and_save(
            gamma_dir,
            raw_dir,
            tmp_path / "output",
            source_dataset="test/MTO-2D",
            source_revision="fixture",
            verify_hashes=False,
        )


def test_validation_record_fixture_digests_match_gamma_files(tmp_path: Path) -> None:
    _raw_dir, gamma_dir, arrays = _fixture_inputs(tmp_path, row_count=1)
    record = json.loads((gamma_dir / VALIDATION_RECORDS_FILENAME).read_text(encoding="utf-8"))
    case_id = int(arrays["source_case_id"][0])
    content = (gamma_dir / f"2Dheatsink_{case_id}/app/200/gamma").read_bytes()

    assert record["sha256"] == hashlib.sha256(content).hexdigest()
