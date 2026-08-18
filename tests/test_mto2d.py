"""Tests for the MTO2D problem API and isolated runner."""

from collections.abc import Callable
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path
import re
import shutil
from typing import Any

from matplotlib import pyplot as plt
import numpy as np
import pytest

from engibench.constraint import Criticality
from engibench.problems.mto2d import MTO2D
from engibench.problems.mto2d.model.design_io import DESIGN_CELL_COUNT
from engibench.problems.mto2d.model.design_io import GAMMA_CELL_COUNT
from engibench.problems.mto2d.model.design_io import HALF_DESIGN_SHAPE
from engibench.problems.mto2d.model.design_io import parse_internal_field
from engibench.problems.mto2d.model.runner import MTO2DRunner
from engibench.problems.mto2d.model.runner import OptimizationSchedule
from engibench.problems.mto2d.model.runner import RunnerSettings
from engibench.problems.mto2d.model.runner import SCHEDULE_RUNTIME_MARKER
from engibench.problems.mto2d.model.runner import SCHEDULE_RUNTIME_VERSION
from engibench.problems.mto2d.model.runner import SolverRun
from engibench.problems.mto2d.model.runner import SolverRunError
from engibench.problems.mto2d.v0 import main as mto2d_main
from tests.problem_policies import PROBLEM_TEST_POLICIES

OPTIMIZATION_STEPS = 3
CONTINUATION_PARAMETER_COUNT = 3


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


def test_engibench_condition_and_objective_names_follow_topology_conventions() -> None:
    problem = MTO2D(runner=FakeRunner())  # type: ignore[arg-type]

    assert problem.conditions_keys == ["inlet_velocity", "max_power_dissipation", "volfrac"]
    assert problem.objectives_keys == ["mean_temperature", "power_dissipation"]
    assert ("optimal_design", *problem.conditions_keys, *problem.objectives_keys) == (
        "optimal_design",
        "inlet_velocity",
        "max_power_dissipation",
        "volfrac",
        "mean_temperature",
        "power_dissipation",
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
    (case / "src_TF" / "EXEC").write_text("fixture executable\n", encoding="utf-8")
    (case / SCHEDULE_RUNTIME_MARKER).write_text(f"{SCHEDULE_RUNTIME_VERSION}\n", encoding="ascii")

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
solid_area 0;
fluid_area 1;
test_PD 0;
D_normalization 1.57572e-7;
D0 90;
D1 50;
qu 0.005;
""",
        encoding="utf-8",
    )
    (app / "system" / "controlDict").write_text(
        "endTime 200;\nwriteInterval 200;\nwritePrecision 4;\n",
        encoding="utf-8",
    )
    (app / "system" / "blockMeshDict").write_text(
        """blocks
(
    hex (0 7 6 3 10 17 16 13)
    (160 400 1)
    simpleGrading (1 1 1)

    hex (7 1 2 6 17 11 12 16)
    zone_test
    (40 400 1)
    simpleGrading (1 1 1)

    hex (6 2 4 5 16 12 14 15)
    zone_fluid
    (40 80 1)
    simpleGrading (1 1 1)

    hex (1 7 8 9 11 17 18 19)
    zone_fluid
    (40 80 1)
    simpleGrading (1 1 1)
);
""",
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


def _fake_solver(
    *,
    mean_temperature: float = 9.45825,
    power_dissipation: float = 62.2588,
    volume_residual: float = -0.000671484,
    elapsed_time: float = 13713.0,
) -> Callable[[Path, RunnerSettings, str], None]:
    """Return an `execute` callable that writes the outputs a real solver would."""

    def execute(case_dir: Path, _settings: RunnerSettings, _kind: str) -> None:
        app = case_dir / "app"
        latest = app / "1"
        latest.mkdir()
        shutil.copy2(app / "0" / "gamma", latest / "gamma")
        for name, value in {
            "meanT.txt": mean_temperature,
            "Disspower.txt": power_dissipation,
            "Voluse.txt": volume_residual,
            "Time.txt": elapsed_time,
        }.items():
            (app / name).write_text(f"{value}\n", encoding="utf-8")

    return execute


def test_problem_simulation_and_optimization_objective_order() -> None:
    runner = FakeRunner()
    problem = MTO2D(max_iter=OPTIMIZATION_STEPS, runner=runner)  # type: ignore[arg-type]
    design = problem.uniform_starting_design(problem.conditions.volfrac)

    result = problem.simulate_verbose(design, {"max_power_dissipation": 62.0})

    np.testing.assert_array_equal(result.objective_values, np.array([9.0, 61.0]))
    assert result.power_constraint_residual == pytest.approx(61.0 / 62.0 - 1.0)
    assert result.volume_constraint_residual == pytest.approx(-0.01)
    assert runner.calls[0][1].max_iter == 1
    assert runner.calls[0][2] == "simulate"

    optimized, history = problem.optimize(
        design,
        {"mode": "warm", "optimization_schedule": "strict"},
    )

    assert optimized.shape == HALF_DESIGN_SHAPE
    assert optimized.dtype == np.float32
    assert len(history) == OPTIMIZATION_STEPS
    np.testing.assert_array_equal(history[0].obj_values, np.array([9.0, 61.0]))
    np.testing.assert_array_equal(history[-1].obj_values, np.array([8.0, 60.0]))
    assert [step.step for step in history] == [1, 2, 3]
    assert runner.calls[1][1].mode == "warm"
    assert runner.calls[1][1].optimization_schedule == "strict"
    assert runner.calls[1][2] == "optimize"


def test_optimization_history_carries_active_power_diagnostics() -> None:
    runner = FakeRunner()
    problem = MTO2D(
        max_iter=OPTIMIZATION_STEPS,
        max_power_dissipation=63.1,
        runner=runner,  # type: ignore[arg-type]
    )
    design = problem.uniform_starting_design(problem.conditions.volfrac)

    with pytest.warns(UserWarning, match="exact prefix|final active bound") as legacy_warnings:
        _optimized, cold_history = problem.optimize(design, {"mode": "cold"})
    warning_messages = [str(warning.message) for warning in legacy_warnings]
    assert any("exact prefix" in message for message in warning_messages)
    assert any("final active bound is 89.4" in message for message in warning_messages)

    cold_bounds = np.array([step.active_power_bound for step in cold_history])
    np.testing.assert_allclose(cold_bounds, [89.8, 89.6, 89.4])
    np.testing.assert_allclose(
        [step.power_constraint_residual for step in cold_history],
        [61.0 / 89.8 - 1.0, 60.5 / 89.6 - 1.0, 60.0 / 89.4 - 1.0],
    )
    np.testing.assert_allclose([step.volume_constraint_residual for step in cold_history], [-0.01, -0.015, -0.02])
    np.testing.assert_allclose([step.elapsed_time for step in cold_history], [5.0, 10.0, 15.0])

    warm_overrides = {"mode": "warm", "optimization_schedule": "strict"}
    _optimized, warm_history = problem.optimize(design, warm_overrides)
    np.testing.assert_allclose([step.active_power_bound for step in warm_history], [63.1, 63.1, 63.1])

    problem.simulate(design)
    np.testing.assert_allclose([step.active_power_bound for step in warm_history], [63.1, 63.1, 63.1])


def test_historical_uniform_start_and_render_use_native_half_domain() -> None:
    problem = MTO2D()
    design = problem.uniform_starting_design(0.61)

    assert problem.design_space.contains(design)
    np.testing.assert_array_equal(design, np.full(HALF_DESIGN_SHAPE, 0.61, dtype=np.float32))
    assert problem.design_volume_residual(design, 0.61) == pytest.approx(0.0288889, abs=1e-7)
    fig, ax = problem.render(design)
    rendered = np.asarray(ax.images[0].get_array())
    assert rendered.shape == (400, 400)
    np.testing.assert_array_equal(rendered[:, :200], design)
    np.testing.assert_array_equal(rendered[:, 200:], np.fliplr(design))
    plt.close(fig)


def test_check_constraints_warns_when_fluid_volume_exceeds_bound() -> None:
    problem = MTO2D(volfrac=0.25)
    feasible = np.zeros(HALF_DESIGN_SHAPE, dtype=np.float32)

    assert not problem.check_constraints(feasible, {})

    violations = problem.check_constraints(np.ones(HALF_DESIGN_SHAPE, dtype=np.float32), {})

    assert not violations.by_criticality(Criticality.Error)
    warnings = violations.by_criticality(Criticality.Warning)
    assert len(warnings) == 1
    assert "fluid_volume_bound" in str(warnings)
    assert "All-cell fluid fraction 1" in str(warnings)


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


def test_main_uses_sampled_conditions_without_hub_or_container_access(capsys: pytest.CaptureFixture[str]) -> None:
    runner = FakeRunner()
    render_open_window: list[bool] = []
    design = np.full(HALF_DESIGN_SHAPE, 0.2, dtype=np.float32)
    dataset = {
        "train": [
            {
                "optimal_design": design.ravel(),
                "inlet_velocity": -0.0569,
                "max_power_dissipation": 57.5,
                "volfrac": 0.26,
                "mean_temperature": 22.171801,
                "power_dissipation": 57.488201,
            }
        ]
    }

    class ExampleMTO2D(MTO2D):
        def __init__(self, seed: int = 0) -> None:
            super().__init__(seed=seed, dataset=dataset, runner=runner)  # type: ignore[arg-type]

        def render(self, design: np.ndarray, *, open_window: bool = False) -> tuple[Any, Any]:
            render_open_window.append(open_window)
            return None, None

    mto2d_main(ExampleMTO2D)

    output = capsys.readouterr().out
    _recorded_design, settings, kind = runner.calls[-1]
    assert kind == "simulate"
    assert settings.inlet_velocity == pytest.approx(-0.0569)
    assert settings.max_power_dissipation == pytest.approx(57.5)
    assert settings.volfrac == pytest.approx(0.26)
    assert render_open_window == [False]
    assert "Stored objectives: mean_temperature=22.171801, power_dissipation=57.488201" in output
    assert "Simulated objectives: mean_temperature=9, power_dissipation=61" in output
    assert "Simulation diagnostics:" in output


def test_default_solver_config_uses_configured_container_reference() -> None:
    runner = FakeRunner()
    problem = MTO2D(runner=runner)  # type: ignore[arg-type]
    design = problem.uniform_starting_design(problem.conditions.volfrac)

    problem.simulate(design)

    _recorded_design, settings, kind = runner.calls[-1]
    assert kind == "simulate"
    assert settings.max_iter == 1
    assert MTO2D()._runner.image == problem.container_id  # noqa: SLF001
    assert MTO2D()._runner.case_template is None  # noqa: SLF001
    assert asdict(problem._runner_settings(problem.config)) == asdict(problem.config)  # noqa: SLF001


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"mode": "invalid"}, "mode"),
        ({"optimization_schedule": "invalid"}, "optimization_schedule"),
        ({"continuation_profile": "quadratic"}, "continuation_profile"),
        ({"power_bound_start": 0.0}, "power_bound_start"),
        ({"volfrac": 0.01}, "volfrac"),
    ],
)
def test_invalid_problem_config_is_rejected(config: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MTO2D(**config)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"mode": "warm"}, "warm repair"),
        ({"max_iter": 201}, "shorter exact prefix"),
        ({"continuation_steps": 1}, "continuation_steps"),
        (
            {
                "max_iter": 5,
                "continuation_steps": 2,
                "optimization_schedule": "strict",
            },
            "divisible",
        ),
        (
            {
                "continuation_steps": 0,
                "optimization_schedule": "strict",
            },
            "between 1 and max_iter",
        ),
        ({"qu_start": 0.0}, "qu_start"),
        ({"alpha_max_start": -1.0}, "alpha_max_start"),
        ({"heaviside_start": 0.0}, "heaviside_start"),
    ],
)
def test_invalid_optimization_config_is_rejected_only_by_optimize(
    config: dict[str, Any],
    message: str,
) -> None:
    runner = FakeRunner()
    problem = MTO2D(**config, runner=runner)  # type: ignore[arg-type]
    design = problem.uniform_starting_design(problem.conditions.volfrac)

    problem.simulate(design)

    with pytest.raises(ValueError, match=message):
        problem.optimize(design)
    assert [kind for _design, _settings, kind in runner.calls] == ["simulate"]


def test_runner_prepares_and_parses_isolated_frozen_case(tmp_path: Path) -> None:
    template = _make_case_template(tmp_path)
    problem = MTO2D(
        mpi_cores=6,
        runner=MTO2DRunner(
            case_template=str(template),
            work_dir=str(tmp_path),
            retain_artifacts=True,
            execute=_fake_solver(),
        ),
    )
    design = np.linspace(0.0, 1.0, DESIGN_CELL_COUNT, dtype=np.float32).reshape(HALF_DESIGN_SHAPE)

    result = problem.simulate_verbose(design)

    np.testing.assert_allclose(result.objective_values, [9.45825, 62.2588])
    assert result.volume_constraint_residual == pytest.approx(-0.000671484)
    assert result.elapsed_time == pytest.approx(13713.0)
    assert result.artifacts_path is not None
    assert problem.last_solver_run is not None
    np.testing.assert_array_equal(problem.last_solver_run.final_design, design)
    prepared = Path(result.artifacts_path) / "case" / "app"

    transport = (prepared / "constant" / "transportProperties").read_text(encoding="utf-8")
    assert re.search(r"alphaMax\s+alphaMax \[0 0 -1 0 0 0 0\] 5025200\.0;", transport)
    assert re.search(r"alphamax\s+alphamax \[0 0 -1 0 0 0 0\] 5025200\.0;", transport)
    assert "movlim 0.0;" in transport
    assert "updateDesign false;" in transport
    assert "voluse 0.61;" in transport
    assert "D0 63.1;" in transport
    assert "D1 63.1;" in transport
    assert "qu 0.01;" in transport

    inlet = (prepared / "0" / "U").read_text(encoding="utf-8")
    assert "value uniform (0 -0.074 0);" in inlet
    decomposition = (prepared / "system" / "decomposeParDict").read_text(encoding="utf-8")
    assert "numberOfSubdomains 6;" in decomposition
    assert "method simple;" in decomposition
    assert "n (2 3 1);" in decomposition
    control = (prepared / "system" / "controlDict").read_text(encoding="utf-8")
    assert "writePrecision 12;" in control
    continuation = (prepared / "constant" / "continuationProperties").read_text(encoding="utf-8")
    assert "n_steps         1;" in continuation
    assert "optimizationSchedule strict;" in continuation
    assert continuation.count("from           0.01;") == 1
    assert continuation.count("from           5025200;") == 1
    assert continuation.count("from           59.8;") == 1
    assert not (prepared / "9").exists()
    assert not (prepared / "processor0").exists()

    prepared_gamma = parse_internal_field(
        (prepared / "0" / "gamma").read_text(encoding="ascii"),
        expected_count=GAMMA_CELL_COUNT,
    )
    np.testing.assert_array_equal(prepared_gamma[DESIGN_CELL_COUNT:], 1.0)


def test_legacy_schedule_prepares_source_initialization_and_endpoints(tmp_path: Path) -> None:
    template = _make_case_template(tmp_path)
    design = np.full(HALF_DESIGN_SHAPE, 0.5, dtype=np.float32)
    settings = RunnerSettings(
        inlet_velocity=-0.074,
        max_power_dissipation=63.1,
        volfrac=0.61,
        max_iter=1,
        optimization_schedule="legacy",
    )

    MTO2DRunner.validate_settings(settings, "optimize")
    MTO2DRunner()._prepare_case(template, design, settings, "optimize")  # noqa: SLF001

    transport = (template / "app" / "constant" / "transportProperties").read_text(encoding="utf-8")
    assert re.search(r"alphaMax\s+alphaMax \[0 0 -1 0 0 0 0\] 2500\.0;", transport)
    assert "qu 0.005;" in transport
    assert "D0 90.0;" in transport
    continuation = (template / "app" / "constant" / "continuationProperties").read_text(encoding="utf-8")
    assert "n_steps         1;" in continuation
    assert "optimizationSchedule legacy;" in continuation
    assert continuation.count("overallType    constant;") == CONTINUATION_PARAMETER_COUNT
    assert continuation.count("from           0.005;") == 1
    assert continuation.count("to             0.01;") == 1
    assert continuation.count("from           2500;") == 1
    assert continuation.count("to             5025226.63913;") == 1
    assert continuation.count("from           0.1;") == 1
    assert continuation.count("to             59.8;") == 1


@pytest.mark.parametrize("schedule", ["legacy", "strict"])
def test_optimization_schedule_requires_rebuilt_runtime_marker(
    tmp_path: Path,
    schedule: OptimizationSchedule,
) -> None:
    template = _make_case_template(tmp_path)
    marker = template / SCHEDULE_RUNTIME_MARKER
    marker.unlink()

    with pytest.raises(FileNotFoundError, match=f"optimization_schedule='{schedule}'"):
        MTO2DRunner._validate_runtime_marker(template, f"optimization_schedule={schedule!r}")  # noqa: SLF001

    marker.write_text("1\n", encoding="ascii")
    with pytest.raises(ValueError, match="expected '2'"):
        MTO2DRunner._validate_runtime_marker(template, f"optimization_schedule={schedule!r}")  # noqa: SLF001

    marker.write_text(f"{SCHEDULE_RUNTIME_VERSION}\n", encoding="ascii")
    MTO2DRunner._validate_runtime_marker(template, f"optimization_schedule={schedule!r}")  # noqa: SLF001


def test_frozen_simulation_requires_rebuilt_runtime_marker(tmp_path: Path) -> None:
    template = _make_case_template(tmp_path)
    (template / SCHEDULE_RUNTIME_MARKER).unlink()

    with pytest.raises(FileNotFoundError, match="frozen simulation requires"):
        MTO2DRunner._validate_runtime_marker(template, "frozen simulation")  # noqa: SLF001


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("type groovyBC;", "exactly one 'type fixedValue;'"),
        ("type fixedValue;\n        type fixedValue;", "exactly one 'type fixedValue;'"),
    ],
)
def test_inlet_velocity_requires_one_fixed_value_type(
    tmp_path: Path,
    replacement: str,
    message: str,
) -> None:
    template = _make_case_template(tmp_path)
    inlet = template / "app" / "0" / "U"
    inlet.write_text(
        inlet.read_text(encoding="utf-8").replace("type fixedValue;", replacement),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        MTO2DRunner._write_inlet_velocity(inlet, -0.074)  # noqa: SLF001


def test_inlet_velocity_rejects_duplicate_inlet_boundaries(tmp_path: Path) -> None:
    template = _make_case_template(tmp_path)
    inlet = template / "app" / "0" / "U"
    inlet.write_text(
        inlet.read_text(encoding="utf-8")
        + """
inlet
{
    type fixedValue;
    value uniform (0 -0.025 0);
}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly one inlet boundary"):
        MTO2DRunner._write_inlet_velocity(inlet, -0.074)  # noqa: SLF001


def test_case_template_rejects_wrong_gamma_cell_count(tmp_path: Path) -> None:
    template = _make_case_template(tmp_path)
    values = np.concatenate(
        (
            np.zeros(DESIGN_CELL_COUNT, dtype=np.float64),
            np.ones(GAMMA_CELL_COUNT - DESIGN_CELL_COUNT - 1, dtype=np.float64),
        )
    )
    (template / "app" / "0" / "gamma").write_text(_foam_gamma(values), encoding="ascii")

    with pytest.raises(ValueError, match="expected 86400"):
        MTO2DRunner(case_template=str(template))._resolve_case_template()  # noqa: SLF001


def test_case_template_requires_solver_executable(tmp_path: Path) -> None:
    template = _make_case_template(tmp_path)
    (template / "src_TF" / "EXEC").unlink()

    with pytest.raises(FileNotFoundError, match="compiled solver executable"):
        MTO2DRunner(case_template=str(template))._resolve_case_template()  # noqa: SLF001


def test_case_template_rejects_nonfluid_gamma_tail(tmp_path: Path) -> None:
    template = _make_case_template(tmp_path)
    values = np.concatenate(
        (
            np.zeros(DESIGN_CELL_COUNT, dtype=np.float64),
            np.ones(GAMMA_CELL_COUNT - DESIGN_CELL_COUNT, dtype=np.float64),
        )
    )
    values[-1] = 0.0
    (template / "app" / "0" / "gamma").write_text(_foam_gamma(values), encoding="ascii")

    with pytest.raises(ValueError, match="final 6,400 gamma cells"):
        MTO2DRunner(case_template=str(template))._resolve_case_template()  # noqa: SLF001


def test_case_template_rejects_incompatible_area_switch(tmp_path: Path) -> None:
    template = _make_case_template(tmp_path)
    transport = template / "app" / "constant" / "transportProperties"
    transport.write_text(
        transport.read_text(encoding="utf-8").replace("fluid_area 1;", "fluid_area 0;"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires fluid_area=1"):
        MTO2DRunner(case_template=str(template))._resolve_case_template()  # noqa: SLF001


def test_case_template_rejects_wrong_power_normalization(tmp_path: Path) -> None:
    template = _make_case_template(tmp_path)
    transport = template / "app" / "constant" / "transportProperties"
    transport.write_text(
        transport.read_text(encoding="utf-8").replace(
            "D_normalization 1.57572e-7;",
            "D_normalization 1.58e-7;",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"requires D_normalization=1\.57572e-07"):
        MTO2DRunner(case_template=str(template))._resolve_case_template()  # noqa: SLF001


def test_case_template_rejects_wrong_fixed_fluid_zone(tmp_path: Path) -> None:
    template = _make_case_template(tmp_path)
    block_mesh = template / "app" / "system" / "blockMeshDict"
    block_mesh.write_text(
        block_mesh.read_text(encoding="utf-8").replace("zone_fluid", "zone_solid", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ordered 80,000-cell design region"):
        MTO2DRunner(case_template=str(template))._resolve_case_template()  # noqa: SLF001


def test_case_template_rejects_wrong_block_mesh_cell_count(tmp_path: Path) -> None:
    template = _make_case_template(tmp_path)
    block_mesh = template / "app" / "system" / "blockMeshDict"
    block_mesh.write_text(
        block_mesh.read_text(encoding="utf-8").replace("(40 80 1)", "(40 79 1)", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly 86,400 cells"):
        MTO2DRunner(case_template=str(template))._resolve_case_template()  # noqa: SLF001


def test_update_design_dictionary_value_is_replaced_without_duplication(tmp_path: Path) -> None:
    dictionary = tmp_path / "transportProperties"
    dictionary.write_text("movlim 0.4;\nupdateDesign true;\n", encoding="utf-8")

    MTO2DRunner._upsert_plain_dictionary_value(  # noqa: SLF001
        dictionary,
        "updateDesign",
        value=False,
    )

    updated = dictionary.read_text(encoding="utf-8")
    assert updated.count("updateDesign") == 1
    assert "updateDesign false;" in updated


def test_dictionary_replacement_rejects_duplicate_keys(tmp_path: Path) -> None:
    dictionary = tmp_path / "transportProperties"
    dictionary.write_text("D1 50;\nD1 60;\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one 'D1'"):
        MTO2DRunner._replace_dictionary_value(dictionary, "D1", 63.1)  # noqa: SLF001


@pytest.mark.parametrize(
    ("mode", "configured_power_bound_start", "expected_power_bound_start"),
    [
        ("cold", None, 90.0),
        ("warm", None, 63.1),
        ("warm", 90.0, 90.0),
    ],
)
def test_one_step_optimization_smoke(
    tmp_path: Path,
    mode: str,
    configured_power_bound_start: float | None,
    expected_power_bound_start: float,
) -> None:
    template = _make_case_template(tmp_path)
    problem = MTO2D(
        max_iter=1,
        mode=mode,
        optimization_schedule="strict",
        power_bound_start=configured_power_bound_start,
        runner=MTO2DRunner(
            case_template=str(template),
            work_dir=str(tmp_path),
            retain_artifacts=True,
            execute=_fake_solver(),
        ),
    )
    starting_design = problem.uniform_starting_design(problem.conditions.volfrac)

    if expected_power_bound_start > problem.conditions.max_power_dissipation:
        with pytest.warns(UserWarning, match="final active bound is 89.8"):
            optimized_design, history = problem.optimize(starting_design)
    else:
        optimized_design, history = problem.optimize(starting_design)

    np.testing.assert_array_equal(optimized_design, starting_design)
    assert len(history) == 1
    np.testing.assert_allclose(history[0].obj_values, [9.45825, 62.2588])
    assert problem.last_solver_run is not None
    assert problem.last_solver_run.artifacts_path is not None
    transport = (
        Path(problem.last_solver_run.artifacts_path) / "case" / "app" / "constant" / "transportProperties"
    ).read_text(encoding="utf-8")
    assert f"D0 {expected_power_bound_start};" in transport
    assert "updateDesign true;" in transport


def test_container_backend_uses_isolated_writable_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = tmp_path / "case"
    (case / "app").mkdir(parents=True)
    (case / "src_TF").mkdir()
    captured: dict[str, Any] = {}

    def fake_container_run(*args: Any, **kwargs: Any) -> None:
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr("engibench.problems.mto2d.model.runner.container.run", fake_container_run)
    settings = RunnerSettings(
        inlet_velocity=-0.074,
        max_power_dissipation=63.1,
        volfrac=0.61,
    )
    runner = MTO2DRunner(image="engibench-mto2d:test")

    runner._run_container(case, settings, "simulate")  # noqa: SLF001

    assert (tmp_path / "container-home").is_dir()
    assert (tmp_path / "container-tmp").is_dir()
    serial_command = captured["args"][0][-1]
    assert "../src_TF/EXEC" in serial_command
    assert "decomposePar" not in serial_command
    assert "mpirun" not in serial_command
    assert "-parallel" not in serial_command
    assert "reconstructPar" not in serial_command
    assert captured["kwargs"]["mounts"] == ((str(tmp_path), "/work"),)
    assert captured["kwargs"]["env"] == {
        "HOME": "/work/container-home",
        "TMPDIR": "/work/container-tmp",
    }
    assert captured["kwargs"]["sync_uid"] is True

    runner._run_container(case, settings, "optimize")  # noqa: SLF001
    assert "reconstructPar" not in captured["args"][0][-1]

    runner._run_container(case, replace(settings, mpi_cores=4), "simulate")  # noqa: SLF001
    assert "reconstructPar -latestTime" in captured["args"][0][-1]

    runner._run_container(case, replace(settings, mpi_cores=4), "optimize")  # noqa: SLF001
    parallel_command = captured["args"][0][-1]
    assert "decomposePar" in parallel_command
    assert "mpirun -np 4" in parallel_command
    assert "reconstructPar -latestTime" in parallel_command


def test_container_backend_exports_image_case_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    case = run_root / "case"
    captured: dict[str, Any] = {}

    def fake_container_run(*args: Any, **kwargs: Any) -> None:
        captured["args"] = args
        captured["kwargs"] = kwargs
        (case / "app").mkdir(parents=True)
        (case / "src_TF").mkdir()

    monkeypatch.setattr("engibench.problems.mto2d.model.runner.container.run", fake_container_run)
    runner = MTO2DRunner(image="ghcr.io/ideallab/engibench-mto2d:test")

    runner._export_case_template(run_root, case)  # noqa: SLF001

    assert captured["args"] == (
        ["mto2d-export-case", "/work/case"],
        "ghcr.io/ideallab/engibench-mto2d:test",
    )
    assert captured["kwargs"]["mounts"] == ((str(run_root), "/work"),)
    assert captured["kwargs"]["env"] == {
        "HOME": "/work/container-home",
        "TMPDIR": "/work/container-tmp",
    }
    assert captured["kwargs"]["sync_uid"] is True


def test_runner_validates_a_case_exported_by_a_custom_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_container_run(*_args: Any, **kwargs: Any) -> None:
        case = Path(kwargs["mounts"][0][0]) / "case"
        (case / "app").mkdir(parents=True)
        (case / "src_TF").mkdir()

    monkeypatch.setattr("engibench.problems.mto2d.model.runner.container.run", fake_container_run)
    runner = MTO2DRunner(
        image="custom-mto2d:test",
        work_dir=str(tmp_path),
        retain_on_failure=False,
    )
    settings = RunnerSettings(inlet_velocity=-0.074, max_power_dissipation=63.1, volfrac=0.61)

    with pytest.raises(SolverRunError, match="compiled solver executable"):
        runner.run(np.full(HALF_DESIGN_SHAPE, 0.61, dtype=np.float32), settings, kind="simulate")


def test_frozen_output_validation_rejects_nonfinite_gamma(tmp_path: Path) -> None:
    app = tmp_path / "app"
    (app / "1").mkdir(parents=True)
    values = np.zeros(GAMMA_CELL_COUNT, dtype=np.float64)
    values[DESIGN_CELL_COUNT:] = 1.0
    values[0] = np.nan
    (app / "1" / "gamma").write_text(_foam_gamma(values), encoding="ascii")

    with pytest.raises(ValueError, match="non-finite or invalid gamma"):
        MTO2DRunner._validate_frozen_output(  # noqa: SLF001
            app,
            np.zeros(HALF_DESIGN_SHAPE, dtype=np.float32),
        )


def test_frozen_output_validation_rejects_finite_design_change(tmp_path: Path) -> None:
    app = tmp_path / "app"
    (app / "1").mkdir(parents=True)
    values = np.zeros(GAMMA_CELL_COUNT, dtype=np.float64)
    values[DESIGN_CELL_COUNT:] = 1.0
    values[0] = 2e-7
    (app / "1" / "gamma").write_text(_foam_gamma(values), encoding="ascii")

    with pytest.raises(ValueError, match="changed the design"):
        MTO2DRunner._validate_frozen_output(  # noqa: SLF001
            app,
            np.zeros(HALF_DESIGN_SHAPE, dtype=np.float32),
        )


def test_final_design_rejects_changed_fixed_fluid_cells(tmp_path: Path) -> None:
    app = tmp_path / "app"
    (app / "1").mkdir(parents=True)
    values = np.ones(GAMMA_CELL_COUNT, dtype=np.float64)
    values[-1] = 0.0
    (app / "1" / "gamma").write_text(_foam_gamma(values), encoding="ascii")

    with pytest.raises(ValueError, match="final 6,400 gamma cells"):
        MTO2DRunner._read_final_design(app)  # noqa: SLF001


def test_simulation_rejects_unpatched_solver_gamma(
    tmp_path: Path,
) -> None:
    template = _make_case_template(tmp_path)

    def fake_execute(case_dir: Path, _settings: RunnerSettings, _kind: str) -> None:
        app = case_dir / "app"
        (app / "1").mkdir()
        values = np.zeros(GAMMA_CELL_COUNT, dtype=np.float64)
        values[DESIGN_CELL_COUNT:] = 1.0
        values[0] = np.nan
        (app / "1" / "gamma").write_text(_foam_gamma(values), encoding="ascii")
        for filename, value in {
            "meanT.txt": 9.0,
            "Disspower.txt": 62.0,
            "Voluse.txt": -0.001,
            "Time.txt": 1.0,
        }.items():
            (app / filename).write_text(f"{value}\n", encoding="utf-8")

    problem = MTO2D(
        runner=MTO2DRunner(
            case_template=str(template),
            work_dir=str(tmp_path),
            retain_on_failure=True,
            execute=fake_execute,
        )
    )

    with pytest.raises(SolverRunError, match="likely lacks updateDesign support"):
        problem.simulate(problem.uniform_starting_design(0.61))


@pytest.mark.parametrize("retention", ["discard", "retain"])
def test_solver_failure_reports_log_tail_and_retained_artifacts(
    tmp_path: Path,
    retention: str,
) -> None:
    retain_on_failure = retention == "retain"
    template = _make_case_template(tmp_path)

    def failing_execute(case_dir: Path, _settings: RunnerSettings, _kind: str) -> None:
        (case_dir / "run.log").write_text("fatal solver marker\n", encoding="utf-8")
        raise RuntimeError("solver returned non-zero exit status 7")

    problem = MTO2D(
        runner=MTO2DRunner(
            case_template=str(template),
            work_dir=str(tmp_path),
            retain_on_failure=retain_on_failure,
            execute=failing_execute,
        )
    )
    design = problem.uniform_starting_design(problem.conditions.volfrac)

    with pytest.raises(SolverRunError, match="returned non-zero exit status 7") as error:
        problem.simulate(design)

    message = str(error.value)
    assert "Solver log:" in message
    assert "fatal solver marker" in message
    if retain_on_failure:
        assert error.value.artifacts_path is not None
        assert (error.value.artifacts_path / "case" / "run.log").is_file()
    else:
        assert error.value.artifacts_path is None


def test_shared_suite_policy_keeps_optimization_out() -> None:
    """Keep the expensive MTO2D optimizer and simulation out of the default suite."""
    policy = PROBLEM_TEST_POLICIES["problems.mto2d.v0.MTO2D"]

    assert not policy.exercise_optimization
    assert policy.supported_machines == ("x86_64", "amd64")
    assert policy.slow
