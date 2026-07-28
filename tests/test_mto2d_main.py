"""Focused tests for the dataset-backed MTO2D demonstration entry point."""

import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from datasets import Dataset
from datasets import DatasetDict
from datasets import Features
from datasets import Sequence
from datasets import Value
import numpy as np
import pytest

from engibench.problems.mto2d import v0 as mto2d_module
from engibench.problems.mto2d.model.design_io import HALF_DESIGN_SHAPE
from engibench.problems.mto2d.model.runner import RunnerSettings
from engibench.problems.mto2d.model.runner import SolverRun

MPI_CORES = 3
SELECTED_INDEX = 2


class RecordingRunner:
    """Record calls and return a deterministic frozen-simulation result."""

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, RunnerSettings, str]] = []

    def run(self, design: np.ndarray, settings: RunnerSettings, *, kind: str) -> SolverRun:
        self.calls.append((design.copy(), settings, kind))
        return SolverRun(
            final_design=design.copy(),
            mean_temperature=np.array([9.25]),
            power_dissipation=np.array([61.5]),
            volume_residual=np.array([-0.002]),
            elapsed_time=np.array([12.0]),
            artifacts_path=None,
        )


def _row(design: np.ndarray, *, exact: bool = True) -> dict[str, Any]:
    return {
        "optimal_design": design,
        "inlet_velocity": -0.051,
        "max_power_dissipation": 62.0,
        "volume_fraction": 0.47,
        "mean_temperature": 9.5,
        "power_dissipation": 61.0,
        "design_provenance": "native fixture" if exact else "lossy fixture",
        "design_is_exact": exact,
        "objectives_evaluated_on_design": exact,
    }


@pytest.fixture
def saved_flat_dataset(tmp_path: Path) -> Path:
    """Save a beams3d-style flattened design DatasetDict."""
    native = np.linspace(0.0, 1.0, int(np.prod(HALF_DESIGN_SHAPE)), dtype=np.float32)
    row = _row(native)
    features = Features(
        {
            "optimal_design": Sequence(Value("float32")),
            "inlet_velocity": Value("float64"),
            "max_power_dissipation": Value("float64"),
            "volume_fraction": Value("float64"),
            "mean_temperature": Value("float64"),
            "power_dissipation": Value("float64"),
            "design_provenance": Value("string"),
            "design_is_exact": Value("bool"),
            "objectives_evaluated_on_design": Value("bool"),
        }
    )
    columns = {key: [value] for key, value in row.items()}
    split = Dataset.from_dict(columns, features=features)
    dataset = DatasetDict({"train": split, "val": split, "test": split})
    output = tmp_path / "mto_2d_v0"
    dataset.save_to_disk(str(output))
    return output


def test_main_uses_flat_dataset_row_conditions_and_opt_in_simulation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    native = np.linspace(0.0, 1.0, int(np.prod(HALF_DESIGN_SHAPE)), dtype=np.float32).reshape(HALF_DESIGN_SHAPE)
    dataset = {
        "train": [
            _row(np.zeros(HALF_DESIGN_SHAPE, dtype=np.float32)),
            _row(native.reshape(-1), exact=False),
        ]
    }
    runner = RecordingRunner()
    render_path = tmp_path / "selected.png"

    result = mto2d_module.main(
        dataset=dataset,
        split="train",
        index=1,
        solver_config={"mpi_cores": MPI_CORES, "inlet_velocity": -0.09},
        run_simulation=True,
        render_output=render_path,
        runner=runner,  # type: ignore[arg-type]
    )

    assert result is not None
    np.testing.assert_allclose(result.objective_values, [9.25, 61.5])
    assert render_path.is_file()
    assert len(runner.calls) == 1
    called_design, settings, kind = runner.calls[0]
    np.testing.assert_array_equal(called_design, native)
    assert called_design.shape == HALF_DESIGN_SHAPE
    assert settings.inlet_velocity == pytest.approx(-0.051)
    assert settings.max_power_dissipation == pytest.approx(62.0)
    assert settings.volume_fraction == pytest.approx(0.47)
    assert settings.mpi_cores == MPI_CORES
    assert kind == "simulate"

    output = capsys.readouterr().out
    assert "Selected split='train', index=1" in output
    assert "Stored objectives: mean_temperature=9.5, power_dissipation=61" in output
    assert "WARNING: this design was reconstructed lossily" in output
    assert "Simulated objectives: mean_temperature=9.25, power_dissipation=61.5" in output


def test_main_does_not_call_solver_without_explicit_flag(capsys: pytest.CaptureFixture[str]) -> None:
    runner = RecordingRunner()

    result = mto2d_module.main(
        dataset={"train": [_row(np.full(HALF_DESIGN_SHAPE, 0.25, dtype=np.float32))]},
        runner=runner,  # type: ignore[arg-type]
    )

    assert result is None
    assert runner.calls == []
    assert "Simulation skipped. Pass --simulate" in capsys.readouterr().out


def test_main_auto_uses_repository_dataset(
    saved_flat_dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(mto2d_module, "REPOSITORY_DATASET_PATH", saved_flat_dataset)

    result = mto2d_module.main(open_window=False)

    assert result is None
    output = capsys.readouterr().out
    assert f"Dataset: {saved_flat_dataset.resolve()}" in output
    assert "Selected split='train', index=0" in output


def test_cli_reads_json_solver_config_and_keeps_simulation_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "solver.json"
    config_path.write_text('{"backend": "command", "driver_command": ["driver"], "mpi_cores": 4}', encoding="utf-8")
    monkeypatch.setenv(mto2d_module.SOLVER_CONFIG_ENV_VAR, str(tmp_path / "missing-environment.json"))
    monkeypatch.setattr(mto2d_module, "LOCAL_RUNTIME_CONFIG_PATH", tmp_path / "missing-local.json")
    captured: dict[str, Any] = {}

    def fake_main(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(mto2d_module, "main", fake_main)
    mto2d_module._cli(  # noqa: SLF001
        [
            "--dataset",
            "local-dataset",
            "--split",
            "test",
            "--index",
            str(SELECTED_INDEX),
            "--no-show",
            "--simulate",
            "--solver-config",
            str(config_path),
        ]
    )

    assert captured["dataset_source"] == "local-dataset"
    assert captured["split"] == "test"
    assert captured["index"] == SELECTED_INDEX
    assert captured["run_simulation"] is True
    assert captured["open_window"] is False
    assert captured["solver_config"] == {
        "backend": "command",
        "driver_command": ["driver"],
        "mpi_cores": 4,
    }


def test_cli_auto_discovers_local_solver_config_only_for_simulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_checkout = tmp_path / "EngiBench"
    (source_checkout / ".git").mkdir(parents=True)
    local_config = tmp_path / "mto2d-docker.json"
    local_config.write_text(
        '{"backend": "container", "container_image": "auto-image", "case_template": "/auto/case", "mpi_cores": 4}',
        encoding="utf-8",
    )
    monkeypatch.setattr(mto2d_module, "SOURCE_CHECKOUT_PATH", source_checkout)
    monkeypatch.setattr(mto2d_module, "LOCAL_RUNTIME_CONFIG_PATH", local_config)
    monkeypatch.setenv(mto2d_module.SOLVER_CONFIG_ENV_VAR, "")
    monkeypatch.setenv("ENGIBENCH_MTO2D_CASE_TEMPLATE", "/environment/case")
    monkeypatch.setenv("ENGIBENCH_MTO2D_IMAGE", "environment-image")
    captured: dict[str, Any] = {}

    def fake_main(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(mto2d_module, "main", fake_main)
    mto2d_module._cli(["--simulate", "--no-show"])  # noqa: SLF001

    assert captured["solver_config"] == {
        "backend": "container",
        "container_image": "environment-image",
        "case_template": "/environment/case",
        "mpi_cores": 4,
    }

    captured.clear()
    mto2d_module._cli(["--no-show"])  # noqa: SLF001
    assert captured["solver_config"] == {}


def test_cli_environment_solver_config_wins_over_local_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment_config = tmp_path / "environment.json"
    environment_config.write_text(
        '{"backend": "container", "container_image": "configured-image", '
        '"case_template": "/configured/case", "mpi_cores": 2}',
        encoding="utf-8",
    )
    malformed_local = tmp_path / "local.json"
    malformed_local.write_text("not JSON", encoding="utf-8")
    monkeypatch.setenv(mto2d_module.SOLVER_CONFIG_ENV_VAR, str(environment_config))
    monkeypatch.setenv("ENGIBENCH_MTO2D_IMAGE", "lower-priority-image")
    monkeypatch.setattr(mto2d_module, "LOCAL_RUNTIME_CONFIG_PATH", malformed_local)
    captured: dict[str, Any] = {}

    def fake_main(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(mto2d_module, "main", fake_main)
    mto2d_module._cli(["--simulate", "--no-show"])  # noqa: SLF001

    assert captured["solver_config"] == {
        "backend": "container",
        "container_image": "configured-image",
        "case_template": "/configured/case",
        "mpi_cores": 2,
    }


def test_solver_config_auto_discovery_is_silent_when_local_artifact_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_checkout = tmp_path / "EngiBench"
    (source_checkout / ".git").mkdir(parents=True)
    monkeypatch.setattr(mto2d_module, "SOURCE_CHECKOUT_PATH", source_checkout)
    monkeypatch.setattr(mto2d_module, "LOCAL_RUNTIME_CONFIG_PATH", tmp_path / "missing.json")
    monkeypatch.delenv(mto2d_module.SOLVER_CONFIG_ENV_VAR, raising=False)

    assert mto2d_module._read_solver_config(None, auto_discover=True) == {}  # noqa: SLF001


def test_v0_file_runs_directly_with_local_dataset(saved_flat_dataset: Path, tmp_path: Path) -> None:
    script = Path(mto2d_module.__file__).resolve()
    rendering = tmp_path / "direct.png"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset",
            str(saved_flat_dataset),
            "--index",
            "0",
            "--no-show",
            "--render-output",
            str(rendering),
        ],
        cwd=script.parents[3],
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Stored objectives: mean_temperature=9.5, power_dissipation=61" in completed.stdout
    assert "Simulation skipped." in completed.stdout
    assert rendering.is_file()
