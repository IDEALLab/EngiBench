"""Tests for the MTO2D problem API and isolated runner."""

from pathlib import Path
import re
import sys
from typing import Any

from matplotlib import pyplot as plt
import numpy as np
import pytest

from engibench.problems.mto2d import MTO2D
from engibench.problems.mto2d.model.design_io import DESIGN_CELL_COUNT
from engibench.problems.mto2d.model.design_io import GAMMA_CELL_COUNT
from engibench.problems.mto2d.model.design_io import HALF_DESIGN_SHAPE
from engibench.problems.mto2d.model.design_io import parse_internal_field
from engibench.problems.mto2d.model.runner import RunnerSettings
from engibench.problems.mto2d.model.runner import SolverRun
from engibench.problems.mto2d.model.runner import SolverRunError

OPTIMIZATION_STEPS = 3


class FakeRunner:
    """Return deterministic histories while recording runner inputs."""

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, RunnerSettings, str]] = []

    def run(self, design: np.ndarray, settings: RunnerSettings, *, kind: str) -> SolverRun:
        """Record one call and provide a deterministic result."""
        self.calls.append((design.copy(), settings, kind))
        count = 1 if kind == "simulate" else settings.max_iter
        return SolverRun(
            final_design=np.full(HALF_DESIGN_SHAPE, 0.25, dtype=np.float32),
            mean_temperature=np.linspace(9.0, 8.0, count, dtype=np.float64),
            power_dissipation=np.linspace(61.0, 60.0, count, dtype=np.float64),
            volume_residual=np.linspace(-0.01, -0.02, count, dtype=np.float64),
            elapsed_time=np.linspace(5.0, 5.0 * count, count, dtype=np.float64),
            artifacts_path="fake-mto2d-artifacts",
        )


def _foam_gamma(values: np.ndarray) -> str:
    value_text = "\n".join(format(float(value), ".9g") for value in values)
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    location "200";
    object gamma;
}}
internalField nonuniform List<scalar>
{values.size}
(
{value_text}
);
boundaryField {{}}
"""


def _make_case_template(root: Path) -> Path:
    case = root / "template"
    app = case / "app"
    (app / "0").mkdir(parents=True)
    (app / "constant").mkdir()
    (app / "system").mkdir()
    (case / "src_TF").mkdir()

    gamma = np.concatenate(
        (
            np.zeros(DESIGN_CELL_COUNT, dtype=np.float64),
            np.ones(GAMMA_CELL_COUNT - DESIGN_CELL_COUNT, dtype=np.float64),
        )
    )
    (app / "0" / "gamma").write_text(_foam_gamma(gamma), encoding="ascii")
    (app / "9").mkdir()
    (app / "9" / "stale-output").write_text("stale", encoding="utf-8")
    (app / "processor0").mkdir()
    (app / "processor0" / "stale-output").write_text("stale", encoding="utf-8")
    (app / "0" / "U").write_text(
        """boundaryField
{
    inlet
    {
        type fixedValue;
        value uniform (0 -0.025 0);
    }
}
""",
        encoding="utf-8",
    )
    (app / "constant" / "transportProperties").write_text(
        """alphaMax alphaMax [0 0 -1 0 0 0 0] 2.5e3;
alphamax alphamax [0 0 -1 0 0 0 0] 2.5e3;
movlim 0.4;
voluse 0.5;
D1 50;
qu 0.005;
""",
        encoding="utf-8",
    )
    (app / "system" / "controlDict").write_text(
        "endTime 200;\nwriteInterval 200;\n",
        encoding="utf-8",
    )
    (app / "system" / "decomposeParDict").write_text(
        """numberOfSubdomains 4;
simpleCoeffs
{
    n (2 2 1);
    delta 0.001;
}
""",
        encoding="utf-8",
    )
    return case


def _make_driver(path: Path) -> Path:
    path.write_text(
        """from pathlib import Path
import os
import shutil

app = Path(os.environ["MTO2D_CASE_DIR"]) / "app"
latest = app / "1"
latest.mkdir()
shutil.copy2(app / "0" / "gamma", latest / "gamma")
for name, value in {
    "meanT.txt": 9.45825,
    "Disspower.txt": 62.2588,
    "Voluse.txt": -0.000671484,
    "Time.txt": 13713.0,
}.items():
    (app / name).write_text(f"{value}\\n", encoding="utf-8")
""",
        encoding="utf-8",
    )
    return path


def test_problem_simulation_and_optimization_objective_order() -> None:
    runner = FakeRunner()
    problem = MTO2D(config={"max_iter": OPTIMIZATION_STEPS}, runner=runner)  # type: ignore[arg-type]
    design = problem.uniform_starting_design(problem.conditions.volume_fraction)

    result = problem.simulate_verbose(design, {"max_power_dissipation": 62.0})

    np.testing.assert_array_equal(result.objective_values, np.array([9.0, 61.0]))
    assert result.power_constraint_residual == pytest.approx(61.0 / 62.0 - 1.0)
    assert result.volume_constraint_residual == pytest.approx(-0.01)
    assert runner.calls[0][1].max_iter == 1
    assert runner.calls[0][2] == "simulate"

    optimized, history = problem.optimize(design, {"mode": "warm"})

    assert optimized.shape == HALF_DESIGN_SHAPE
    assert optimized.dtype == np.float32
    assert len(history) == OPTIMIZATION_STEPS
    np.testing.assert_array_equal(history[0].obj_values, np.array([9.0, 61.0]))
    np.testing.assert_array_equal(history[-1].obj_values, np.array([8.0, 60.0]))
    assert [step.step for step in history] == [1, 2, 3]
    assert runner.calls[1][1].mode == "warm"
    assert runner.calls[1][2] == "optimize"


def test_uniform_start_and_render_use_native_half_domain() -> None:
    problem = MTO2D()
    design = problem.uniform_starting_design(0.61)

    assert problem.design_space.contains(design)
    assert problem.design_volume_residual(design, 0.61) == pytest.approx(0.0, abs=3e-8)
    fig, ax = problem.render(design)
    rendered = np.asarray(ax.images[0].get_array())
    assert rendered.shape == (400, 400)
    np.testing.assert_array_equal(rendered[:, :200], design)
    np.testing.assert_array_equal(rendered[:, 200:], np.fliplr(design))
    plt.close(fig)


def test_random_design_uses_injected_dataset_and_reset() -> None:
    first = np.full(HALF_DESIGN_SHAPE, 0.2, dtype=np.float32)
    second = np.full(HALF_DESIGN_SHAPE, 0.8, dtype=np.float32)
    dataset: dict[str, list[dict[str, Any]]] = {"train": [{"optimal_design": first}, {"optimal_design": second}]}
    problem = MTO2D(seed=7, dataset=dataset)

    design_a, index_a = problem.random_design()
    problem.reset(seed=7)
    design_b, index_b = problem.random_design()

    assert index_a == index_b
    np.testing.assert_array_equal(design_a, design_b)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"mode": "invalid"}, "mode"),
        ({"backend": "invalid"}, "backend"),
        ({"timeout": 0.0}, "timeout"),
        ({"max_iter": 5, "continuation_steps": 2}, "divisible"),
        ({"volume_fraction": 0.01}, "volume_fraction"),
    ],
)
def test_invalid_problem_config_is_rejected(config: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MTO2D(config=config)


def test_command_backend_prepares_and_parses_isolated_frozen_case(tmp_path: Path) -> None:
    template = _make_case_template(tmp_path)
    driver = _make_driver(tmp_path / "driver.py")
    problem = MTO2D(
        config={
            "case_template": str(template),
            "backend": "command",
            "driver_command": (sys.executable, str(driver)),
            "mpi_cores": 6,
            "work_dir": str(tmp_path),
            "retain_artifacts": True,
        }
    )
    design = np.linspace(0.0, 1.0, DESIGN_CELL_COUNT, dtype=np.float32).reshape(HALF_DESIGN_SHAPE)

    result = problem.simulate_verbose(design)

    np.testing.assert_allclose(result.objective_values, [9.45825, 62.2588])
    assert result.volume_constraint_residual == pytest.approx(-0.000671484)
    assert result.elapsed_time == pytest.approx(13713.0)
    assert result.artifacts_path is not None
    prepared = Path(result.artifacts_path) / "case" / "app"

    transport = (prepared / "constant" / "transportProperties").read_text(encoding="utf-8")
    assert re.search(r"alphaMax\s+alphaMax \[0 0 -1 0 0 0 0\] 5025200\.0;", transport)
    assert re.search(r"alphamax\s+alphamax \[0 0 -1 0 0 0 0\] 5025200\.0;", transport)
    assert "movlim 0.0;" in transport
    assert "voluse 0.61;" in transport
    assert "D1 63.1;" in transport
    assert "qu 0.019;" in transport

    inlet = (prepared / "0" / "U").read_text(encoding="utf-8")
    assert "value uniform (0 -0.074 0);" in inlet
    decomposition = (prepared / "system" / "decomposeParDict").read_text(encoding="utf-8")
    assert "numberOfSubdomains 6;" in decomposition
    assert "n (2 3 1);" in decomposition
    continuation = (prepared / "constant" / "continuationProperties").read_text(encoding="utf-8")
    assert "n_steps         1;" in continuation
    assert continuation.count("from           0.019;") == 1
    assert continuation.count("from           5025200;") == 1
    assert continuation.count("from           59.8;") == 1
    assert not (prepared / "9").exists()
    assert not (prepared / "processor0").exists()

    prepared_gamma = parse_internal_field(
        (prepared / "0" / "gamma").read_text(encoding="ascii"),
        expected_count=GAMMA_CELL_COUNT,
    )
    np.testing.assert_array_equal(prepared_gamma[DESIGN_CELL_COUNT:], 1.0)


def test_command_backend_one_step_optimization_smoke(tmp_path: Path) -> None:
    template = _make_case_template(tmp_path)
    driver = _make_driver(tmp_path / "driver.py")
    problem = MTO2D(
        config={
            "case_template": str(template),
            "backend": "command",
            "driver_command": (sys.executable, str(driver)),
            "max_iter": 1,
            "mode": "warm",
            "work_dir": str(tmp_path),
        }
    )
    starting_design = problem.uniform_starting_design(problem.conditions.volume_fraction)

    optimized_design, history = problem.optimize(starting_design)

    np.testing.assert_array_equal(optimized_design, starting_design)
    assert len(history) == 1
    np.testing.assert_allclose(history[0].obj_values, [9.45825, 62.2588])


def test_solver_failure_reports_retained_artifacts(tmp_path: Path) -> None:
    template = _make_case_template(tmp_path)
    problem = MTO2D(
        config={
            "case_template": str(template),
            "backend": "command",
            "driver_command": (sys.executable, "-c", "raise SystemExit(7)"),
            "work_dir": str(tmp_path),
            "retain_on_failure": True,
        }
    )
    design = problem.uniform_starting_design(problem.conditions.volume_fraction)

    with pytest.raises(SolverRunError, match="returned non-zero exit status 7") as error:
        problem.simulate(design)

    assert error.value.artifacts_path is not None
    assert (error.value.artifacts_path / "case" / "run.log").is_file()
