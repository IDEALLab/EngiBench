import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from datasets import load_from_disk
import numpy as np
import pyarrow.parquet as pq
import pytest

from engibench.problems.mto2d.model import publish_native_dataset
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

CANONICAL_RAMP_Q = 0.01


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
    assert manifest["canonical_frozen_simulation"]["ramp_q"] == CANONICAL_RAMP_Q
    assert manifest["canonical_frozen_simulation"]["design_update"] is False
    assert manifest["historical_label_semantics"]["evaluated_on_stored_design"] is False
    copied_records = output / VALIDATION_RECORDS_FILENAME
    assert copied_records.is_file()
    assert manifest["gamma_validation_records_sha256"] == hashlib.sha256(copied_records.read_bytes()).hexdigest()
    card = (output / DATASET_CARD_FILENAME).read_text(encoding="utf-8")
    assert "Publication blocked" in card
    assert "different solver states" in " ".join(card.split())
    assert "canonical frozen simulation uses the source-matched final" in " ".join(card.split())
    assert "historical pre-update" in manifest["residual_semantics"]["power_constraint_residual"]
    assert "exact post-update" in manifest["residual_semantics"]["volume_constraint_residual"]

    candidate = publish_native_dataset.ValidatedDataset(
        path=output,
        dataset=reloaded,
        manifest=manifest,
        split_sizes={name: len(split) for name, split in reloaded.items()},
    )
    public = publish_native_dataset.public_dataset(candidate)
    for split in public.values():
        assert tuple(split.column_names) == publish_native_dataset.PUBLIC_COLUMNS
        assert split.features == publish_native_dataset.public_dataset_features()


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


def test_publication_cli_is_dry_run_by_default_and_requires_rights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate = publish_native_dataset.ValidatedDataset(
        path=tmp_path,
        dataset=object(),
        manifest={},
        split_sizes={"train": 4_249, "val": 283, "test": 1_134},
    )
    monkeypatch.setattr(publish_native_dataset, "validate_publication_dataset", lambda _path: candidate)

    publish_native_dataset.main(["--dataset-dir", str(tmp_path)])

    output = capsys.readouterr().out
    assert "Dry run complete; no Hub repository was created or modified." in output
    assert "--confirm-redistribution-rights --push" in output

    with pytest.raises(ValueError, match="--push requires --confirm-redistribution-rights"):
        publish_native_dataset.main(["--dataset-dir", str(tmp_path), "--push"])


def test_publication_cli_pushes_only_after_explicit_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = publish_native_dataset.ValidatedDataset(
        path=tmp_path,
        dataset=object(),
        manifest={},
        split_sizes={"train": 4_249, "val": 283, "test": 1_134},
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(publish_native_dataset, "validate_publication_dataset", lambda _path: candidate)

    def fake_publish(validated, **kwargs) -> publish_native_dataset.PublicationResult:
        captured["candidate"] = validated
        captured.update(kwargs)
        return publish_native_dataset.PublicationResult(
            commit_oid="abc123",
            commit_url="https://huggingface.co/datasets/IDEALLab/mto_2d_v0/commit/abc123",
            readme_sha256="0" * 64,
            data_files=(),
        )

    monkeypatch.setattr(publish_native_dataset, "publish_dataset", fake_publish)

    publish_native_dataset.main(
        [
            "--dataset-dir",
            str(tmp_path),
            "--repo-id",
            "IDEALLab/mto_2d_v0",
            "--confirm-redistribution-rights",
            "--push",
        ]
    )

    assert captured["candidate"] is candidate
    assert captured["repo_id"] == "IDEALLab/mto_2d_v0"
    assert captured["license_id"] == "mit"


def test_public_parquet_staging_uses_minimal_schema_and_expected_layout(tmp_path: Path) -> None:
    raw_dir, gamma_dir, _arrays = _fixture_inputs(tmp_path, row_count=20)
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
    candidate = publish_native_dataset.ValidatedDataset(
        path=output,
        dataset=converted,
        manifest={},
        split_sizes={name: len(split) for name, split in converted.items()},
    )

    staged = publish_native_dataset._stage_public_parquet(  # noqa: SLF001
        publish_native_dataset.public_dataset(candidate),
        tmp_path / "public-data",
    )

    assert [path.name for path in staged] == [
        "train-00000-of-00003.parquet",
        "train-00001-of-00003.parquet",
        "train-00002-of-00003.parquet",
        "val-00000-of-00001.parquet",
        "test-00000-of-00001.parquet",
    ]
    expected_schema = publish_native_dataset.public_dataset_features().arrow_schema
    assert all(pq.read_schema(path).equals(expected_schema, check_metadata=False) for path in staged)


def test_publication_commit_is_data_only_and_preserves_readme(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = publish_native_dataset.ValidatedDataset(
        path=tmp_path,
        dataset=object(),
        manifest={},
        split_sizes={"train": 4_249, "val": 283, "test": 1_134},
    )
    readme = b"---\nlicense: mit\n---\n\n# Human-edited card"
    desired_names = [
        "train-00000-of-00003.parquet",
        "train-00001-of-00003.parquet",
        "train-00002-of-00003.parquet",
        "val-00000-of-00001.parquet",
        "test-00000-of-00001.parquet",
    ]
    desired_paths = {f"data/{name}" for name in desired_names}
    captured: dict[str, object] = {}

    def fake_stage(_dataset, data_dir: Path) -> tuple[Path, ...]:
        data_dir.mkdir(parents=True)
        paths = tuple(data_dir / name for name in desired_names)
        for path in paths:
            path.write_bytes(b"parquet")
        return paths

    class FakeApi:
        def repo_info(self, **_kwargs):
            return SimpleNamespace(sha="before", private=False)

        def list_repo_files(self, *, revision, **_kwargs):
            if revision == "before":
                return [
                    ".gitattributes",
                    "README.md",
                    "gamma-validation.jsonl",
                    "data/obsolete.parquet",
                ]
            return [".gitattributes", "README.md", "gamma-validation.jsonl", *sorted(desired_paths)]

        def create_commit(self, *, operations, parent_commit, **_kwargs):
            captured["operations"] = operations
            captured["parent_commit"] = parent_commit
            assert all(
                Path(operation.path_or_fileobj).is_file()
                for operation in operations
                if hasattr(operation, "path_or_fileobj")
            )
            return SimpleNamespace(oid="after", commit_url="https://example.test/commit/after")

    monkeypatch.setattr(publish_native_dataset, "public_dataset", lambda _candidate: object())
    monkeypatch.setattr(publish_native_dataset, "_stage_public_parquet", fake_stage)
    monkeypatch.setattr(publish_native_dataset, "_read_remote_readme", lambda _repo, _revision: readme)
    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)

    result = publish_native_dataset.publish_dataset(
        candidate,
        repo_id="IDEALLab/mto_2d_v0",
        license_id="mit",
        private=False,
    )

    operations = captured["operations"]
    operation_paths = {operation.path_in_repo for operation in operations}
    assert captured["parent_commit"] == "before"
    assert operation_paths == desired_paths | {"data/obsolete.parquet"}
    assert all(path.startswith("data/") and path.endswith(".parquet") for path in operation_paths)
    assert "README.md" not in operation_paths
    assert "gamma-validation.jsonl" not in operation_paths
    assert result.readme_sha256 == hashlib.sha256(readme).hexdigest()
