"""Tests for the MTO3D problem API and isolated runner (no container required)."""

from collections.abc import Callable
from dataclasses import asdict
from dataclasses import replace
import gzip
from pathlib import Path
import re
import shutil
from typing import Any

from matplotlib import pyplot as plt
import numpy as np
import pytest

from engibench.constraint import Criticality
from engibench.problems.mto3d import MTO3D
from engibench.problems.mto3d import v0
from engibench.problems.mto3d.model import design_io as dio
from engibench.problems.mto3d.model.runner import FROZEN_ITERATIONS
from engibench.problems.mto3d.model.runner import FROZEN_STATUS_FILE
from engibench.problems.mto3d.model.runner import MTO3DRunner
from engibench.problems.mto3d.model.runner import RunnerSettings
from engibench.problems.mto3d.model.runner import RUNTIME_MARKER
from engibench.problems.mto3d.model.runner import SolverRun
from engibench.problems.mto3d.model.runner import SolverRunError
from engibench.problems.mto3d.model.runner import XH_TEMPLATE
from engibench.problems.mto3d.v0 import main as mto3d_main
from tests.problem_policies import PROBLEM_TEST_POLICIES

OPTIMIZATION_STEPS = 3
FROZEN_NS_PASSES = 131
ROW_VOLFRAC = 0.32
FV_SOLUTION = (
    'solvers\n{\n     "(T)"\n    {\n        solver          GAMG;\n        smoother        GaussSeidel;\n'
    '        tolerance       1e-5;\n        relTol          0;\n    }\n     "(Ta|Tb)"\n    {\n'
    "        solver          PBiCGStab;\n        tolerance       1e-5;\n        relTol          0;\n    }\n}\n"
)


class FakeRunner:
    """Return deterministic results while recording runner inputs."""

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, RunnerSettings, str]] = []

    def run(self, design: np.ndarray, settings: RunnerSettings, *, kind: str) -> SolverRun:
        self.calls.append((design.copy(), settings, kind))
        count = 1 if kind == "simulate" else OPTIMIZATION_STEPS
        frozen = (
            {
                "mean_temperature": 0.718137,
                "power_dissipation": 21.5744,
                "volume_residual": -0.0567159,
                "elapsed_clock_time": 8.0,
                "ns_passes": 131.0,
                "ns_converged": 1.0,
            }
            if kind == "simulate"
            else None
        )
        return SolverRun(
            final_design=np.full(dio.HALF_DESIGN_SHAPE, 0.25, dtype=np.float32),
            mean_temperature=np.linspace(0.9, 0.8, count, dtype=np.float64),
            power_dissipation=np.linspace(21.0, 20.0, count, dtype=np.float64),
            volume_residual=np.linspace(-0.01, -0.02, count, dtype=np.float64),
            elapsed_time=np.linspace(10.0, 10.0 * count, count, dtype=np.float64),
            frozen_status=frozen,
            artifacts_path="fake-mto3d-artifacts",
        )


def _xh_text(tokens: list[str], location: str = "1922") -> str:
    return (
        "FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       volScalarField;\n"
        f'    location    "{location}";\n    object      xh;\n}}\n\ndimensions      [0 0 0 0 0 0 0];\n\n'
        f"internalField   nonuniform List<scalar> \n{len(tokens)}\n(\n" + "\n".join(tokens) + "\n)\n;\n\n"
        "boundaryField\n{\n    outlet\n    {\n        type            zeroGradient;\n    }\n"
        "    sym\n    {\n        type            symmetry;\n    }\n}\n"
    )


def _template_tokens(design_value: str = "0.5") -> list[str]:
    tokens = np.full(dio.XH_CELL_COUNT, design_value, dtype=object)
    tokens[dio.SOLID_INDEX] = "0"
    tokens[dio.FLUID_INDEX] = "1"
    return tokens.tolist()


def _make_case_template(root: Path) -> Path:
    case = root / "template"
    for sub in ("app/0", "app/constant", "app/system", "src"):
        (case / sub).mkdir(parents=True)
    (case / "src" / "EXEC").write_text("fixture executable\n", encoding="utf-8")
    (case / RUNTIME_MARKER).write_text("1\n", encoding="ascii")
    (case / XH_TEMPLATE).write_bytes(gzip.compress(_xh_text(_template_tokens()).encode(), mtime=0))
    (case / "app/0/U").write_text(
        "boundaryField\n{\n    inlet\n    {\n        type            fixedValue;\n"
        "        value           uniform (0 0 -0.121);\n    }\n    outlet\n    {\n"
        "        type            inletOutlet;\n        inletValue      uniform (0 0 0);\n    }\n}\n",
        encoding="utf-8",
    )
    (case / "app/constant/optProperties").write_text(
        "voluse              0.49;\nmovlim              0.4;\nPowerDissMax        21.6;\n"
        "continuation\n{\n    n_beginning      20;\n    n_levels      20;\n}\n",
        encoding="utf-8",
    )
    (case / "app/constant/transportProperties").write_text(
        "nu   nu [0 2 -1 0 0 0 0] 1.0e-6;\nNSmaxIter               150;//100;\n", encoding="utf-8"
    )
    (case / "app/system/controlDict").write_text(
        "endTime         3000;//comment\nwriteInterval   1;//comment\nwritePrecision  3;//comment\n", encoding="utf-8"
    )
    (case / "app/system/fvSolution").write_text(FV_SOLUTION, encoding="utf-8")
    (case / "app/system/decomposeParDict").write_text(
        "numberOfSubdomains 16;\nmethod          simple;\nsimpleCoeffs\n{\n    n               (2 2 4);\n"
        "    delta           0.001;\n}\nhierarchicalCoeffs\n{\n    n               (1 1 1);\n}\n",
        encoding="utf-8",
    )
    (case / "app/system/blockMeshDict").write_text(
        "blocks\n(\n    hex (0 1 2 3 4 5 6 7) fluid_area (100 20 132) simpleGrading (1 1 1)\n);\n", encoding="utf-8"
    )
    (case / "app/9").mkdir()
    (case / "app/processor0").mkdir()
    return case


def _fake_solver(
    *,
    steps: int = 1,
    converged: bool = True,
    change_design: bool = False,
    write_status: bool = True,
    stopped_early: bool = False,
    energy_converged: bool = True,
    fluid_value: str = "0.951",
    first_design_value: str | None = None,
    frozen_rows: int = FROZEN_ITERATIONS,
) -> Callable[[Path, RunnerSettings, str], None]:
    def execute(case_dir: Path, _settings: RunnerSettings, kind: str) -> None:
        app = case_dir / "app"
        rows = frozen_rows if kind == "simulate" else steps
        # A frozen evaluation and a run stopped by endTime write their final xh at endTime; a converged
        # continuation writes it two steps after the last history row.
        final = app / str(rows if kind == "simulate" or stopped_early else steps + 2)
        final.mkdir()
        if kind == "simulate":
            text = gzip.decompress((app / "0" / "xh.gz").read_bytes()).decode()
            if change_design:
                text = text.replace("\n0.25\n", "\n0.75\n", 1)
            (final / "xh.gz").write_bytes(gzip.compress(text.encode(), mtime=0))
            if write_status:
                (app / FROZEN_STATUS_FILE).write_text(
                    "iteration 1\nmean_temperature 0.71813685838686836\npower_dissipation 21.574406814452406\n"
                    "volume_residual -0.056715922051203072\nns_passes 1\nns_max_iter 20000\n"
                    f"ns_converged {1 if converged else 0}\nelapsed_clock_time 8\n",
                    encoding="ascii",
                )
            log = "GAMG:  Solving for p, Initial residual = 1\n" * FROZEN_NS_PASSES
            log += "GAMG:  Solving for T, Initial residual = 1\n" + "GAMG:  Solving for p, Initial residual = 1e-7\n"
            (case_dir / "run.log").write_text(log, encoding="utf-8")
        elif stopped_early:
            # Before convergence the written xh is the projected field: fixed cells not re-imposed.
            tokens = np.full(dio.XH_CELL_COUNT, "0.4", dtype=object)
            tokens[dio.SOLID_INDEX] = "0.0312"
            tokens[dio.FLUID_INDEX] = fluid_value
            if first_design_value is not None:
                tokens[dio.NATIVE_DESIGN_INDEX.reshape(-1)[0]] = first_design_value
            (final / "xh.gz").write_bytes(gzip.compress(_xh_text(tokens.tolist(), final.name).encode(), mtime=0))
        else:
            shutil.copy2(case_dir / XH_TEMPLATE, final / "xh.gz")
        for name, value in (("MeanT", 0.718137), ("PowerDiss", 21.5744), ("Voluse", -0.0567159), ("Time", 8)):
            values = [value] * rows
            if name == "MeanT" and kind == "simulate":
                # deferred correction: geometric approach to the converged value, written at 6 s.f.
                approach = (0.727345 - 0.0092 * 0.6**i if energy_converged else 0.70 + 1e-4 * i for i in range(rows))
                values = [float(f"{v:.6g}") for v in approach]
            (app / f"{name}.txt").write_text("".join(f"{i}\t{v}\n" for i, v in enumerate(values, 1)), encoding="utf-8")

    return execute


def _runner(tmp_path: Path, execute: Callable[[Path, RunnerSettings, str], None]) -> MTO3DRunner:
    return MTO3DRunner(
        case_template=str(_make_case_template(tmp_path)), work_dir=str(tmp_path), retain_artifacts=True, execute=execute
    )


def test_condition_and_objective_names_follow_mto2d() -> None:
    problem = MTO3D(runner=FakeRunner())  # type: ignore[arg-type]
    assert problem.conditions_keys == ["inlet_velocity", "max_power_dissipation", "volfrac"]
    assert problem.objectives_keys == ["mean_temperature", "power_dissipation"]


def test_simulation_uses_full_precision_frozen_status_and_optimization_history() -> None:
    runner = FakeRunner()
    problem = MTO3D(runner=runner)  # type: ignore[arg-type]
    design = problem.uniform_starting_design(problem.conditions.volfrac)
    result = problem.simulate_verbose(design, {"max_power_dissipation": 22.0})
    np.testing.assert_array_equal(result.objective_values, [0.718137, 21.5744])
    assert result.power_constraint_residual == pytest.approx(21.5744 / 22.0 - 1.0)
    assert result.ns_passes == FROZEN_NS_PASSES
    assert runner.calls[0][2] == "simulate"
    optimized, history = problem.optimize(design, {"mode": "warm"})
    assert optimized.shape == dio.HALF_DESIGN_SHAPE
    assert optimized.dtype == np.float32
    assert [step.step for step in history] == [1, 2, 3]
    np.testing.assert_allclose(history[-1].obj_values, [0.8, 20.0])
    assert runner.calls[1][1].mode == "warm"


def test_runner_prepares_and_parses_a_frozen_case(tmp_path: Path) -> None:
    problem = MTO3D(runner=_runner(tmp_path, _fake_solver()), inlet_velocity=-0.1547)
    design = np.full(dio.HALF_DESIGN_SHAPE, 0.25, dtype=np.float32)
    result = problem.simulate_verbose(design)
    np.testing.assert_array_equal(result.objective_values, [0.71813685838686836, 21.574406814452406])
    assert result.volume_constraint_residual == pytest.approx(-0.056715922051203072)
    assert result.artifacts_path is not None
    app = Path(result.artifacts_path) / "case" / "app"
    opt = (app / "constant/optProperties").read_text()
    assert "updateDesign no;" in opt
    assert "NoContinuation      1;" in opt
    assert "NSmaxIter               20000;" in (app / "constant/transportProperties").read_text()
    control = (app / "system/controlDict").read_text()
    assert f"endTime         {FROZEN_ITERATIONS};" in control
    assert f"writeInterval   {FROZEN_ITERATIONS};" in control
    assert "writePrecision  12;" in control
    fv_solution = (app / "system/fvSolution").read_text()
    assert fv_solution == FV_SOLUTION.replace("tolerance       1e-5;", "tolerance       1e-10;", 1).replace(
        "relTol          0;\n    }", "relTol          0;\n        maxIter         20000;\n    }", 1
    )
    assert result.ns_passes == FROZEN_NS_PASSES  # first iteration only, counted in the solver log
    assert "uniform (0 0 -0.155);" in (app / "0/U").read_text()  # 3 significant figures, like the campaigns
    assert "n               (2 2 4);" in (app / "system/decomposeParDict").read_text()
    assert not (app / "9").exists()
    assert not (app / "processor0").exists()
    written = dio.read_xh(app / "0/xh.gz")
    assert written.foam_file["location"] == "0"
    np.testing.assert_array_equal(dio.xh_to_half_design(written.values), design)


def test_legacy_cold_start_supplies_no_initial_design(tmp_path: Path) -> None:
    problem = MTO3D(runner=_runner(tmp_path, _fake_solver(steps=OPTIMIZATION_STEPS)), max_iter=OPTIMIZATION_STEPS)
    design = problem.uniform_starting_design(problem.conditions.volfrac)
    optimized, history = problem.optimize(design)
    assert len(history) == OPTIMIZATION_STEPS
    assert optimized.dtype == np.float32
    assert problem.last_solver_run is not None
    assert problem.last_solver_run.artifacts_path is not None
    app = Path(problem.last_solver_run.artifacts_path) / "case" / "app"
    assert not (app / "0/xh.gz").exists()
    opt = (app / "constant/optProperties").read_text()
    assert "updateDesign yes;" in opt
    assert "NoContinuation      0;" in opt
    assert "NSmaxIter               150;" in (app / "constant/transportProperties").read_text()
    control = (app / "system/controlDict").read_text()
    assert f"endTime         {OPTIMIZATION_STEPS};" in control
    assert "writeInterval   1;" in control
    assert (app / "system/fvSolution").read_text() == FV_SOLUTION


def test_optimization_stopped_before_convergence_returns_the_design_cells(tmp_path: Path) -> None:
    """The solver re-imposes the fixed cells only at the start of the next iteration."""
    execute = _fake_solver(steps=OPTIMIZATION_STEPS, stopped_early=True)
    problem = MTO3D(runner=_runner(tmp_path, execute), max_iter=OPTIMIZATION_STEPS)
    optimized, history = problem.optimize(problem.uniform_starting_design(problem.conditions.volfrac))
    assert len(history) == OPTIMIZATION_STEPS
    np.testing.assert_array_equal(optimized, np.full(dio.HALF_DESIGN_SHAPE, 0.4, dtype=np.float32))


def test_optimization_clips_filter_overshoot(tmp_path: Path) -> None:
    """The iterative PDE filter can overshoot [0, 1] by ~1e-8 (seen after one cold iteration of case 0)."""
    execute = _fake_solver(steps=1, stopped_early=True, fluid_value="1.0000000087", first_design_value="1.0000000005")
    problem = MTO3D(runner=_runner(tmp_path, execute), max_iter=1)
    optimized, _ = problem.optimize(problem.uniform_starting_design(problem.conditions.volfrac))
    assert optimized.flat[0] == np.float32(1.0)
    assert optimized.max() == np.float32(1.0)
    assert optimized.min() == np.float32(0.4)


def test_optimization_rejects_values_far_outside_unit_interval(tmp_path: Path) -> None:
    execute = _fake_solver(steps=1, stopped_early=True, fluid_value="1.01")
    problem = MTO3D(runner=_runner(tmp_path, execute), max_iter=1)
    with pytest.raises(SolverRunError, match="outside"):
        problem.optimize(problem.uniform_starting_design(problem.conditions.volfrac))


def test_warm_start_writes_the_design_and_final_physics(tmp_path: Path) -> None:
    problem = MTO3D(
        runner=_runner(tmp_path, _fake_solver(steps=OPTIMIZATION_STEPS)), max_iter=OPTIMIZATION_STEPS, mode="warm"
    )
    problem.optimize(np.full(dio.HALF_DESIGN_SHAPE, 0.3, dtype=np.float32))
    assert problem.last_solver_run is not None
    assert problem.last_solver_run.artifacts_path is not None
    app = Path(problem.last_solver_run.artifacts_path) / "case" / "app"
    assert (app / "0/xh.gz").exists()
    assert "NoContinuation      1;" in (app / "constant/optProperties").read_text()


def test_nominal_inlet_when_rounding_disabled(tmp_path: Path) -> None:
    problem = MTO3D(runner=_runner(tmp_path, _fake_solver()), inlet_velocity=-0.1547, inlet_significant_figures=None)
    result = problem.simulate_verbose(np.full(dio.HALF_DESIGN_SHAPE, 0.25, dtype=np.float32))
    assert result.artifacts_path is not None
    assert "uniform (0 0 -0.1547);" in (Path(result.artifacts_path) / "case/app/0/U").read_text()


@pytest.mark.parametrize(
    ("execute", "message"),
    [
        (_fake_solver(converged=False), "did not converge the flow"),
        (_fake_solver(change_design=True), "changed the design"),
        (_fake_solver(write_status=False), "frozen-evaluation patch"),
        (_fake_solver(energy_converged=False), "did not converge the energy equation"),
        # the solver stops after an iteration whose flow did not converge: one row, flow message first
        (_fake_solver(converged=False, frozen_rows=1), "did not converge the flow"),
    ],
)
def test_frozen_evaluation_failures_are_reported(tmp_path: Path, execute: Any, message: str) -> None:
    problem = MTO3D(runner=_runner(tmp_path, execute))
    with pytest.raises(SolverRunError, match=message):
        problem.simulate(np.full(dio.HALF_DESIGN_SHAPE, 0.25, dtype=np.float32))


def test_case_template_requires_runtime_marker(tmp_path: Path) -> None:
    runner = _runner(tmp_path, _fake_solver())
    assert runner.case_template is not None
    (Path(runner.case_template) / RUNTIME_MARKER).unlink()
    with pytest.raises(SolverRunError, match="solver patches"):
        MTO3D(runner=runner).simulate(np.full(dio.HALF_DESIGN_SHAPE, 0.25, dtype=np.float32))


@pytest.mark.parametrize(("config", "message"), [({"mode": "lukewarm"}, "mode"), ({"mpi_cores": 12}, "mpi_cores")])
def test_invalid_config_is_rejected(config: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MTO3D(**config)


def test_container_backend_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(*args: Any, **kwargs: Any) -> None:
        captured["args"], captured["kwargs"] = args, kwargs

    monkeypatch.setattr("engibench.problems.mto3d.model.runner.container.run", fake_run)
    case = tmp_path / "case"
    case.mkdir()
    settings = RunnerSettings(inlet_velocity=-0.121, max_power_dissipation=21.6, volfrac=0.49)
    runner = MTO3DRunner(image="engibench-mto3d:test")
    runner._run_container(case, settings, "simulate")  # noqa: SLF001
    command = captured["args"][0][-1]
    assert "mto3d-prepare-mesh ." in command
    assert "decomposePar" in command
    assert "mpirun --oversubscribe -np 16 ../src/EXEC -parallel" in command
    assert "reconstructPar -latestTime" in command
    assert captured["kwargs"]["mounts"] == ((str(tmp_path), "/work"),)
    assert captured["kwargs"]["sync_uid"] is True
    runner._run_container(case, replace(settings, mpi_cores=1), "simulate")  # noqa: SLF001
    assert "mpirun" not in captured["args"][0][-1]


def test_default_solver_config_uses_configured_container_reference() -> None:
    runner = FakeRunner()
    problem = MTO3D(runner=runner)  # type: ignore[arg-type]
    design = problem.uniform_starting_design(problem.conditions.volfrac)

    problem.simulate(design)

    _recorded_design, _settings, kind = runner.calls[-1]
    assert kind == "simulate"
    assert re.fullmatch(r"ghcr\.io/ideallab/engibench-mto3d@sha256:[0-9a-f]{64}", str(problem.container_id))
    assert MTO3D()._runner.image == problem.container_id  # noqa: SLF001
    assert MTO3D()._runner.case_template is None  # noqa: SLF001
    assert asdict(problem._runner_settings(problem.config)) == asdict(problem.config)  # noqa: SLF001


def test_render_draws_mirrored_fluid_cells_colored_by_depth() -> None:
    design = np.zeros(dio.HALF_DESIGN_SHAPE, dtype=np.float32)
    design[19, 0, 0] = 1.0  # one fluid cell: next to the heated wall, at the inlet, at the outer edge
    filled = v0._render_volume(design)  # noqa: SLF001
    assert filled.shape == (200, 20, 100)
    assert filled.sum() == 2 * design.sum()  # the cell and its mirror image
    assert filled[0, 0, 99]  # heated wall at depth 0, inlet at the top
    assert filled[199, 0, 99]

    fig, ax = MTO3D().render(design)
    (faces,) = ax.collections
    colors = faces.get_facecolors()
    assert len(colors) == 3 * filled.sum()  # three camera-facing faces per cell
    # The unshaded front faces of the cells next to the heated wall get the depth-0 color.
    front = [c for c in colors if np.allclose(c, plt.get_cmap("viridis")(0.0))]
    assert len(front) == filled.sum()
    plt.close(fig)


def test_camera_facing_voxel_faces_skips_hidden_and_interior_faces() -> None:
    filled = np.zeros((4, 3, 3), dtype=bool)
    filled[1:3, 1, 1] = True  # two cells side by side along x
    quads, depth, axis = v0._camera_facing_voxel_faces(filled, np.array([0.3, -0.9, 0.4]))  # noqa: SLF001
    # Visible: one +x end face, two front (-depth) faces and two top faces; the shared face is interior.
    assert sorted(axis.tolist()) == [0, 1, 1, 2, 2]
    assert quads.shape == (5, 4, 3)
    np.testing.assert_array_equal(depth[axis == 1], [1.0, 1.0])
    np.testing.assert_array_equal(quads[axis == 0][0][:, 0], [3.0, 3.0, 3.0, 3.0])


def test_check_constraints_warns_when_fluid_volume_exceeds_bound() -> None:
    problem = MTO3D(volfrac=0.25)
    assert not problem.check_constraints(np.zeros(dio.HALF_DESIGN_SHAPE, dtype=np.float32), {})
    violations = problem.check_constraints(np.ones(dio.HALF_DESIGN_SHAPE, dtype=np.float32), {})
    assert not violations.by_criticality(Criticality.Error)
    assert "fluid_volume_bound" in str(violations.by_criticality(Criticality.Warning))


def test_random_design_and_main_use_injected_dataset(capsys: pytest.CaptureFixture[str]) -> None:
    design = np.full(dio.HALF_DESIGN_SHAPE, 0.2, dtype=np.float32)
    row = {
        "optimal_design": design.ravel(),
        "inlet_velocity": -0.1547,
        "max_power_dissipation": 17.1,
        "volfrac": ROW_VOLFRAC,
        "mean_temperature": 0.9,
        "power_dissipation": 17.0,
    }
    dataset = {"train": [row, row]}
    runner = FakeRunner()

    class ExampleMTO3D(MTO3D):
        def __init__(self, seed: int = 0) -> None:
            super().__init__(seed=seed, dataset=dataset, runner=runner)  # type: ignore[arg-type]

        def render(self, design: np.ndarray, *, open_window: bool = False) -> tuple[Any, Any]:
            return None, None

    first, index_a = ExampleMTO3D(seed=7).random_design()
    second, index_b = ExampleMTO3D(seed=7).random_design()
    assert index_a == index_b
    np.testing.assert_array_equal(first, second)
    mto3d_main(ExampleMTO3D)
    _design, settings, kind = runner.calls[-1]
    assert kind == "simulate"
    assert settings.inlet_velocity == pytest.approx(-0.1547)
    assert settings.volfrac == ROW_VOLFRAC
    assert "Simulated objectives: mean_temperature=0.718137" in capsys.readouterr().out


def test_shared_suite_policy_keeps_expensive_runs_out() -> None:
    policy = PROBLEM_TEST_POLICIES["problems.mto3d.v0.MTO3D"]
    assert not policy.exercise_optimization
    assert policy.supported_machines == ("x86_64", "amd64")
    assert policy.slow
