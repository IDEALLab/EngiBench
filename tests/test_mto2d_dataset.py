from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from engibench.problems.mto2d import v0 as mto2d_module
from engibench.problems.mto2d.model.dataset import assemble_shards
from engibench.problems.mto2d.model.dataset import condition_grid
from engibench.problems.mto2d.model.dataset import convert_raw_arrays
from engibench.problems.mto2d.model.dataset import dataset_features
from engibench.problems.mto2d.model.dataset import deterministic_split_indices
from engibench.problems.mto2d.model.dataset import GENERATED_PROVENANCE
from engibench.problems.mto2d.model.dataset import generation_jobs
from engibench.problems.mto2d.model.dataset import legacy_row
from engibench.problems.mto2d.model.dataset import RAW_FILENAMES
from engibench.problems.mto2d.model.dataset import RAW_ROW_COUNT
from engibench.problems.mto2d.model.dataset import run_optimization_case
from engibench.problems.mto2d.model.design_io import HALF_DESIGN_SHAPE


def test_default_condition_grid_has_exact_size_and_stable_order() -> None:
    grid = condition_grid()

    assert grid.shape == (10_000, 3)
    assert grid.dtype == np.float64
    np.testing.assert_allclose(grid[0], [-0.095, 50.0, 0.25])
    np.testing.assert_allclose(grid[1], [-0.095, 50.0, 0.26875])
    np.testing.assert_allclose(grid[24], [-0.095, 50.0, 0.70])
    np.testing.assert_allclose(grid[25], [-0.095, 51.31578947368421, 0.25])
    np.testing.assert_allclose(grid[500], [-0.09131578947368421, 50.0, 0.25])
    np.testing.assert_allclose(grid[-1], [-0.025, 75.0, 0.70])


@pytest.mark.parametrize("invalid_shape", [(0, 20, 25), (20, -1, 25), (20, 25)])
def test_condition_grid_rejects_invalid_shape(invalid_shape) -> None:
    with pytest.raises(ValueError, match="three positive integers"):
        condition_grid(invalid_shape)


def test_legacy_split_has_requested_sizes_and_is_deterministic() -> None:
    first = deterministic_split_indices(RAW_ROW_COUNT, seed=1)
    second = deterministic_split_indices(RAW_ROW_COUNT, seed=1)

    assert {name: len(indices) for name, indices in first.items()} == {
        "train": 4_249,
        "val": 283,
        "test": 1_134,
    }
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    combined = np.concatenate(tuple(first.values()))
    np.testing.assert_array_equal(np.sort(combined), np.arange(RAW_ROW_COUNT))
    assert len(np.unique(combined)) == RAW_ROW_COUNT


def test_generation_jobs_selects_grid_rows_without_large_payloads(tmp_path: Path) -> None:
    small_grid = condition_grid((2, 2, 2))
    jobs = generation_jobs(
        tmp_path,
        solver_config={"max_iter": 1, "driver_command": ["external-driver"]},
        grid=small_grid,
        start_index=2,
        stop_index=4,
    )

    assert [job["case_id"] for job in jobs] == [2, 3]
    assert jobs[0]["volume_fraction"] == pytest.approx(0.25)
    assert jobs[0]["solver_config"] == {"max_iter": 1, "driver_command": ["external-driver"]}
    assert "optimal_design" not in jobs[0]


def test_case_worker_calls_optimize_and_resumes_from_atomic_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    case_id = 4

    class FakeMTO2D:
        def __init__(self, seed, config):
            calls.append(("init", seed, config))
            self.last_solver_run = None

        @staticmethod
        def uniform_starting_design(volume_fraction):
            return np.full(HALF_DESIGN_SHAPE, volume_fraction, dtype=np.float32)

        def optimize(self, starting_design):
            calls.append(("optimize", starting_design.shape))
            self.last_solver_run = SimpleNamespace(
                elapsed_time=np.array([12.0]),
                volume_residual=np.array([-0.001]),
            )
            history = [SimpleNamespace(obj_values=np.array([10.0, 59.0]))]
            return np.asarray(starting_design, dtype=np.float32), history

        @staticmethod
        def simulate_verbose(design):
            calls.append(("simulate", design.shape))
            return SimpleNamespace(
                objective_values=np.array([9.5, 58.0]),
                volume_constraint_residual=-0.002,
                power_constraint_residual=58.0 / 60.0 - 1.0,
                elapsed_time=1.5,
            )

    monkeypatch.setattr(mto2d_module, "MTO2D", FakeMTO2D)
    output_dir = tmp_path / "shards"
    path = run_optimization_case(
        case_id=case_id,
        inlet_velocity=-0.05,
        max_power_dissipation=60.0,
        volume_fraction=0.5,
        output_dir=str(output_dir),
        solver_config={"max_iter": 1},
    )

    assert [call[0] for call in calls] == ["init", "optimize", "simulate"]
    with np.load(path, allow_pickle=False) as shard:
        assert shard["optimal_design"].shape == HALF_DESIGN_SHAPE
        assert shard["mean_temperature"].item() == pytest.approx(9.5)
        assert shard["power_constraint_residual_absolute"].item() == pytest.approx(-2.0)
        assert shard["source_case_id"].item() == case_id

    calls.clear()
    assert (
        run_optimization_case(
            case_id=case_id,
            inlet_velocity=-0.05,
            max_power_dissipation=60.0,
            volume_fraction=0.5,
            output_dir=str(output_dir),
            solver_config={"max_iter": 1},
        )
        == path
    )
    assert calls == []
    with pytest.raises(ValueError, match="do not match requested conditions"):
        run_optimization_case(
            case_id=case_id,
            inlet_velocity=-0.05,
            max_power_dissipation=61.0,
            volume_fraction=0.5,
            output_dir=str(output_dir),
            solver_config={"max_iter": 1},
        )


def test_legacy_row_records_lossy_provenance_and_both_power_residuals() -> None:
    source_case_id = 17
    source_row_index = 3
    row = legacy_row(
        legacy_design=np.full((256, 256), 0.5, dtype=np.float32),
        conditions=np.array([-0.074, 63.1, 0.61]),
        mean_temperature=9.45825,
        power_dissipation=62.2588,
        source_case_id=source_case_id,
        source_row_index=source_row_index,
    )

    assert row["optimal_design"].shape == HALF_DESIGN_SHAPE
    assert row["optimal_design"].dtype == np.float32
    assert row["design_is_exact"] is False
    assert "lossy" in row["design_provenance"]
    assert row["source_case_id"] == source_case_id
    assert row["source_row_index"] == source_row_index
    assert row["objectives_evaluated_on_design"] is False
    assert row["power_constraint_residual_absolute"] == pytest.approx(62.2588 - 63.1)
    assert row["power_constraint_residual_relative"] == pytest.approx(62.2588 / 63.1 - 1.0)


def test_raw_conversion_streams_expected_schema_and_splits(tmp_path: Path) -> None:
    row_count = 20
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_arrays = {
        "design": np.linspace(0.0, 1.0, row_count, dtype=np.float32)[:, None, None]
        * np.ones((row_count, 256, 256), dtype=np.float32),
        "conditions": np.column_stack(
            (
                np.linspace(-0.095, -0.025, row_count),
                np.linspace(50.0, 75.0, row_count),
                np.linspace(0.25, 0.70, row_count),
            )
        ),
        "mean_temperature": np.linspace(8.0, 12.0, row_count),
        "power_dissipation": np.linspace(49.5, 74.5, row_count),
        "source_case_id": np.arange(100, 100 + row_count, dtype=np.int64),
    }
    paths = {}
    for name, filename in RAW_FILENAMES.items():
        path = raw_dir / filename
        np.save(path, raw_arrays[name])
        paths[name] = path

    converted = convert_raw_arrays(
        paths,
        seed=1,
        cache_dir=tmp_path / "cache",
        source_dataset="test/raw",
        source_revision="fixture",
    )

    assert {name: len(split) for name, split in converted.items()} == {"train": 15, "val": 1, "test": 4}
    assert converted["train"].features == dataset_features()
    row = converted["train"][0]
    assert np.asarray(row["optimal_design"]).shape == HALF_DESIGN_SHAPE
    assert row["source_dataset"] == "test/raw"
    assert row["source_revision"] == "fixture"
    assert row["design_is_exact"] is False


def _generated_row(case_id: int) -> dict:
    max_power = 60.0
    power = max_power - 0.5
    return {
        "optimal_design": np.full(HALF_DESIGN_SHAPE, case_id / 20.0, dtype=np.float32),
        "inlet_velocity": -0.05,
        "max_power_dissipation": max_power,
        "volume_fraction": 0.5,
        "mean_temperature": 10.0 + case_id,
        "power_dissipation": power,
        "power_constraint_residual_absolute": power - max_power,
        "power_constraint_residual_relative": power / max_power - 1.0,
        "volume_constraint_residual": 0.0,
        "source_case_id": case_id,
        "source_row_index": case_id,
        "optimization_steps": 1,
        "optimization_elapsed_time": 2.0,
        "evaluation_elapsed_time": 1.0,
        "source_dataset": "MTO2D generated grid",
        "source_revision": "",
        "design_provenance": GENERATED_PROVENANCE,
        "design_is_exact": True,
        "objectives_evaluated_on_design": True,
    }


def test_shard_assembly_streams_rows_and_checks_completeness(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    for case_id in range(20):
        np.savez_compressed(shard_dir / f"case_{case_id:05d}.npz", **_generated_row(case_id))

    converted = assemble_shards(shard_dir, expected_count=20, seed=1, cache_dir=tmp_path / "cache")

    assert {name: len(split) for name, split in converted.items()} == {"train": 15, "val": 1, "test": 4}
    case_ids = sorted(int(case_id) for split in converted.values() for case_id in split["source_case_id"])
    assert case_ids == list(range(20))

    (shard_dir / "case_00019.npz").unlink()
    with pytest.raises(ValueError, match="incomplete shard set"):
        assemble_shards(shard_dir, expected_count=20, cache_dir=tmp_path / "other-cache")
