from dataclasses import dataclass
from dataclasses import field
from math import ceil
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, ClassVar

from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
import pytest

from engibench import core as engibench_core
from engibench.constraint import Criticality
from engibench.constraint import THEORY
from engibench.core import ObjectiveDirection
from engibench.core import OptiStep
from engibench.core import Problem
from engibench.problems.beams3d.model import fem_model as beams3d_fem_model
from engibench.problems.beams3d.model.fem_model import _apply_filter
from engibench.problems.beams3d.model.fem_model import FeaModel3D
from engibench.problems.beams3d.v0 import Beams3D
from engibench.problems.beams3d.v0 import main as beams3d_main

EXPECTED_MAIN_RENDER_COUNT = 2


@dataclass
class _ExampleConditions:
    fixed_elements: np.ndarray = field(default_factory=lambda: np.zeros((2, 2, 2), dtype=np.int64))
    force_elements_z: np.ndarray = field(default_factory=lambda: np.ones((2, 2, 2), dtype=np.int64))
    volfrac: float = 0.3
    rmin: float = 1.5
    penal: float = 3.0


class _ExampleRows:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


class _ExampleSplit:
    def __init__(self) -> None:
        self.rows = [
            {
                "optimal_design": np.full((1, 1, 1), 0.3, dtype=np.float32),
                "volfrac": 0.3,
                "rmin": 1.5,
                "forcedist_x": 0.0,
                "forcedist_y": 1.0,
                "c": 1.25,
                "optimization_history": [1.25],
            }
        ]
        self.column_names = list(self.rows[0])

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[key]

    def select_columns(self, keys: list[str]) -> _ExampleRows:
        return _ExampleRows([{key: row[key] for key in keys} for row in self.rows])


class _ExampleProblem(Problem[np.ndarray]):
    version = 0
    objectives = (("c", ObjectiveDirection.MINIMIZE),)
    Conditions = _ExampleConditions
    Config = _ExampleConditions
    conditions = _ExampleConditions()
    design_space = spaces.Box(low=0.0, high=1.0, shape=(1, 1, 1), dtype=np.float32)
    dataset_id = "example"
    container_id = None
    last_instance: ClassVar["_ExampleProblem | None"] = None

    def __init__(self, seed: int = 0) -> None:
        self.dataset_accessed = False
        self.render_count = 0
        self.render_open_window_flags: list[bool] = []
        self.simulation_config: dict[str, Any] | None = None
        self.optimization_config: dict[str, Any] | None = None
        self.optimized_starting_point: np.ndarray | None = None
        self.reset_seeds: list[int | None] = []
        super().__init__(seed=seed)
        self.config = SimpleNamespace(nelx=1, nely=1, nelz=1, volfrac=0.3, rmin=1.5, penal=3.0)
        self.design_space = spaces.Box(low=0.0, high=1.0, shape=(1, 1, 1), dtype=np.float32)
        _ExampleProblem.last_instance = self

    def reset(self, seed: int | None = None) -> None:
        self.reset_seeds.append(seed)
        super().reset(seed)

    @property
    def dataset(self) -> Any:
        self.dataset_accessed = True
        return {"train": _ExampleSplit()}

    def random_design(self) -> tuple[np.ndarray, int]:
        return np.full((1, 1, 1), 0.3, dtype=np.float32), 0

    def render(self, design: np.ndarray, *, open_window: bool = False) -> np.ndarray:
        self.render_count += 1
        self.render_open_window_flags.append(open_window)
        return design

    def simulate(self, design: np.ndarray, config: dict[str, Any] | None = None) -> np.ndarray:
        self.simulation_config = config
        return np.array([1.25])

    def optimize(
        self,
        starting_point: np.ndarray,
        config: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, list[OptiStep]]:
        self.optimized_starting_point = starting_point
        self.optimization_config = config
        design = np.full((1, 1, 1), 0.25, dtype=np.float32)
        return design, [OptiStep(obj_values=np.array([0.5, 0.0]), step=1)]


def _direct_filter(values: np.ndarray, shape: tuple[int, int, int], rmin: float) -> np.ndarray:
    values_3d = values.reshape(shape)
    filtered = np.zeros(shape, dtype=np.float64)
    hs = np.zeros(shape, dtype=np.float64)
    rceil = ceil(rmin) - 1

    for axis0 in range(shape[0]):
        for axis1 in range(shape[1]):
            for axis2 in range(shape[2]):
                for neighbor0 in range(max(axis0 - rceil, 0), min(axis0 + rceil, shape[0] - 1) + 1):
                    for neighbor1 in range(max(axis1 - rceil, 0), min(axis1 + rceil, shape[1] - 1) + 1):
                        for neighbor2 in range(max(axis2 - rceil, 0), min(axis2 + rceil, shape[2] - 1) + 1):
                            dist = ((axis0 - neighbor0) ** 2 + (axis1 - neighbor1) ** 2 + (axis2 - neighbor2) ** 2) ** 0.5
                            weight = max(0.0, rmin - dist)
                            filtered[axis0, axis1, axis2] += weight * values_3d[neighbor0, neighbor1, neighbor2]
                            hs[axis0, axis1, axis2] += weight

    return filtered.reshape(-1, 1) / hs.reshape(-1, 1)


def test_matrix_free_filter_matches_direct_reference_for_non_cubic_grid() -> None:
    model = FeaModel3D()
    sensitivity_filter = model.get_filter(nelx=2, nely=4, nelz=3, rmin=4.0)
    values = np.arange(4 * 2 * 3, dtype=np.float64).reshape(-1, 1)

    actual = _apply_filter(values, sensitivity_filter)
    expected = _direct_filter(values, sensitivity_filter.shape, rmin=4.0)

    np.testing.assert_allclose(actual, expected)


def test_beams3d_public_objective_matches_dataset_key() -> None:
    problem = Beams3D()

    assert problem.objectives == (("c", ObjectiveDirection.MINIMIZE),)
    assert problem.objectives_keys == ["c"]


def test_beams3d_conditions_match_compact_dataset_schema() -> None:
    problem = Beams3D()

    assert problem.conditions_keys == ["volfrac", "rmin", "forcedist_x", "forcedist_y"]


def test_beams3d_config_builds_matching_non_cubic_default_masks() -> None:
    problem = Beams3D(config={"nelx": 2, "nely": 4, "nelz": 3})

    assert problem.config is not None
    assert problem.design_space.shape == (4, 2, 3)
    assert problem.config.fixed_elements.shape == (3, 5, 4)
    assert problem.config.force_elements_z.shape == (3, 5, 4)


def test_beams3d_config_builds_force_mask_from_forcedist() -> None:
    problem = Beams3D(config={"nelx": 4, "nely": 4, "nelz": 2, "forcedist_x": 0.25, "forcedist_y": 0.75})

    expected_force_elements_z = np.zeros((5, 5, 3), dtype=np.int64)
    expected_force_elements_z[1, 3, -1] = 1
    assert problem.config is not None
    np.testing.assert_array_equal(problem.config.force_elements_z, expected_force_elements_z)


def test_beams3d_non_cubic_instance_config_passes_constraints() -> None:
    problem = Beams3D(config={"nelx": 2, "nely": 4, "nelz": 3})
    design = np.full(problem.design_space.shape, 0.3, dtype=np.float32)

    violations = problem.check_constraints(design, {})

    assert violations.violations == []


def test_beams3d_rejects_empty_fixed_mask_in_constraints() -> None:
    problem = Beams3D(config={"nelx": 2, "nely": 4, "nelz": 3})
    design = np.full(problem.design_space.shape, 0.3, dtype=np.float32)
    fixed_elements = np.zeros((3, 5, 4), dtype=np.int64)

    violations = problem.check_constraints(design, {"fixed_elements": fixed_elements})

    assert "fixed_elements must contain at least one fixed node" in str(violations)


def test_beams3d_rejects_empty_load_mask_in_constraints() -> None:
    problem = Beams3D(config={"nelx": 2, "nely": 4, "nelz": 3})
    design = np.full(problem.design_space.shape, 0.3, dtype=np.float32)
    force_elements_z = np.zeros((3, 5, 4), dtype=np.int64)

    violations = problem.check_constraints(design, {"force_elements_z": force_elements_z})

    assert "force_elements_z must contain at least one loaded node" in str(violations)


def test_beams3d_rmin_below_one_is_theory_valid_but_warned() -> None:
    problem = Beams3D(config={"nelx": 2, "nely": 4, "nelz": 3})
    design = np.full(problem.design_space.shape, 0.3, dtype=np.float32)

    violations = problem.check_constraints(design, {"rmin": 0.5})

    theory_errors = violations.by_category(THEORY).by_criticality(Criticality.Error)
    assert theory_errors.violations == []
    assert "0.5" in str(violations)
    assert "[1.0" in str(violations)


def test_beams3d_warns_when_unsupported_xy_force_masks_are_ignored() -> None:
    problem = Beams3D(config={"nelx": 2, "nely": 4, "nelz": 3})
    starting_point = np.full(problem.design_space.shape, 0.3, dtype=np.float32)
    force_elements_x = np.ones((3, 5, 4), dtype=np.int64)
    force_elements_y = np.ones((3, 5, 4), dtype=np.int64)

    # max_iter=0 fires the boundary-condition merge (and its warning) without a solve.
    with pytest.warns(UserWarning, match="force_elements_x, force_elements_y"):
        problem.optimize(
            starting_point,
            config={"force_elements_x": force_elements_x, "force_elements_y": force_elements_y, "max_iter": 0},
        )


def test_beams3d_rebuilds_force_mask_from_per_call_forcedist_override(monkeypatch) -> None:
    problem = Beams3D(config={"nelx": 4, "nely": 4, "nelz": 2})
    captured: dict[str, Any] = {}

    def fake_run(_self, bcs, x_init=None):
        captured["bcs"] = bcs
        return {"structural_compliance": 0.0}

    monkeypatch.setattr(FeaModel3D, "run", fake_run)
    design = np.full(problem.design_space.shape, 0.3, dtype=np.float32)

    problem.simulate(design, config={"forcedist_x": 0.25, "forcedist_y": 0.75})

    expected_force_elements_z = np.zeros((5, 5, 3), dtype=np.int64)
    expected_force_elements_z[1, 3, -1] = 1
    np.testing.assert_array_equal(captured["bcs"]["force_elements_z"], expected_force_elements_z)


def test_beams3d_supported_cubic_dataset_ids_are_allowlisted() -> None:
    for resolution in (16, 32, 64):
        problem = Beams3D(config={"nelx": resolution, "nely": resolution, "nelz": resolution})

        assert problem.dataset_id == f"IDEALLab/beams_3d_{resolution}_v0"


def test_beams3d_unsupported_dataset_configs_fail_before_loading() -> None:
    problem = Beams3D(config={"nelx": 2, "nely": 4, "nelz": 3})

    assert problem.dataset_id == ""
    with pytest.raises(ValueError, match="dataset access is implemented only for cubic grids"):
        _ = problem.dataset
    with pytest.raises(ValueError, match="dataset access is implemented only for cubic grids"):
        problem.random_design()


def test_beams3d_missing_supported_dataset_error_propagates(monkeypatch) -> None:
    def fake_load_dataset(_dataset_id):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(engibench_core, "load_dataset", fake_load_dataset)
    problem = Beams3D(config={"nelx": 16, "nely": 16, "nelz": 16})

    with pytest.raises(FileNotFoundError, match="missing"):
        _ = problem.dataset


def test_beams3d_random_design_reshapes_flat_dataset_design(monkeypatch) -> None:
    flat_design = np.full(16 * 16 * 16, 0.3, dtype=np.float32)

    def fake_load_dataset(_dataset_id):
        return {"train": {"optimal_design": [flat_design]}}

    monkeypatch.setattr(engibench_core, "load_dataset", fake_load_dataset)
    problem = Beams3D(config={"nelx": 16, "nely": 16, "nelz": 16})

    design, idx = problem.random_design()

    assert idx == 0
    assert design.shape == problem.design_space.shape
    assert design.shape == (16, 16, 16)
    np.testing.assert_allclose(design, np.full((16, 16, 16), 0.3, dtype=np.float32))


def test_beams3d_main_runs_dataset_simulation_and_optimization_example(capsys) -> None:
    beams3d_main(_ExampleProblem)

    captured = capsys.readouterr()
    problem = _ExampleProblem.last_instance

    assert problem is not None
    assert problem.dataset_accessed
    assert problem.render_count == EXPECTED_MAIN_RENDER_COUNT
    assert problem.render_open_window_flags == [False, False]
    assert problem.reset_seeds == [0, 1]
    assert problem.optimized_starting_point is not None
    np.testing.assert_allclose(problem.optimized_starting_point, np.full((1, 1, 1), 0.3, dtype=np.float32))
    assert problem.simulation_config is not None
    assert problem.optimization_config is not None
    assert problem.simulation_config["forcedist_x"] == 0.0
    assert problem.simulation_config["forcedist_y"] == 1.0
    assert problem.optimization_config["forcedist_x"] == 0.0
    assert problem.optimization_config["forcedist_y"] == 1.0
    assert "Reference value: 1.2500" in captured.out
    assert "Final structural compliance: 0.5000" in captured.out


def test_beams3d_render_returns_matplotlib_3d_axes() -> None:
    problem = Beams3D(config={"nelx": 2, "nely": 3, "nelz": 4})
    design = np.full(problem.design_space.shape, 0.6, dtype=np.float32)

    fig, ax = problem.render(design)

    try:
        assert ax.name == "3d"
        assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_zlabel()) == ("x", "y", "z")
    finally:
        plt.close(fig)


def test_beams3d_import_does_not_eager_import_napari() -> None:
    script = "import sys; from engibench.problems.beams3d import Beams3D; assert 'napari' not in sys.modules"

    subprocess.run([sys.executable, "-c", script], check=True)


def test_beams3d_rejects_explicit_mismatched_boundary_mask() -> None:
    problem = Beams3D(config={"nelx": 2, "nely": 4, "nelz": 3})
    design = np.full(problem.design_space.shape, 0.3, dtype=np.float32)

    violations = problem.check_constraints(design, {"fixed_elements": np.zeros((17, 17, 17), dtype=np.int64)})

    assert "Invalid shape for fixed_elements" in str(violations)


def test_beams3d_max_iter_zero_runs_no_optimization_steps() -> None:
    fixed_elements = np.zeros((2, 2, 2), dtype=np.int64)
    fixed_elements[0, 0, 0] = 1
    force_elements_z = np.zeros((2, 2, 2), dtype=np.int64)
    force_elements_z[-1, -1, -1] = 1
    bcs = {
        "fixed_elements": fixed_elements,
        "force_elements_z": force_elements_z,
        "volfrac": 0.3,
        "rmin": 1.5,
        "penal": 3.0,
    }
    x_init = np.full((1, 1, 1), 0.3)

    results = FeaModel3D(eval_only=False, max_iter=0).run(bcs, x_init=x_init)

    assert results["opti_steps"] == []
    np.testing.assert_allclose(results["design"], x_init)


def test_beams3d_opti_steps_store_design_sensitivities_and_update(monkeypatch) -> None:
    def fake_fe_structural_bc_3d(*_args, **_kwargs) -> SimpleNamespace:
        return SimpleNamespace(
            um=np.ones(24, dtype=np.float64),
            edof24=np.arange(24, dtype=np.int64)[None, :],
            ex=np.array([0], dtype=np.int64),
            ey=np.array([0], dtype=np.int64),
            ez=np.array([0], dtype=np.int64),
        )

    def fake_mmasub(inputs):
        return np.full(inputs.n, 0.4, dtype=np.float64), inputs.low, inputs.upp

    monkeypatch.setattr(beams3d_fem_model, "fe_structural_bc_3d", fake_fe_structural_bc_3d)
    monkeypatch.setattr(beams3d_fem_model, "mmasub", fake_mmasub)

    fixed_elements = np.zeros((2, 2, 2), dtype=np.int64)
    fixed_elements[0, 0, 0] = 1
    force_elements_z = np.zeros((2, 2, 2), dtype=np.int64)
    force_elements_z[-1, -1, -1] = 1
    bcs = {
        "fixed_elements": fixed_elements,
        "force_elements_z": force_elements_z,
        "volfrac": 0.3,
        "rmin": 1.5,
        "penal": 3.0,
    }

    results = FeaModel3D(eval_only=False, max_iter=1).run(bcs, x_init=np.full((1, 1, 1), 0.3))

    step = results["opti_steps"][0]
    assert step.x is not None
    assert step.x_sensitivities is not None
    assert step.x_update is not None
    assert step.x.shape == (1, 1, 1)
    assert step.x_sensitivities.shape == (2, 1, 1, 1)
    assert step.x_update.shape == (1, 1, 1)
    np.testing.assert_allclose(step.obj_values_update, np.zeros_like(step.obj_values))
    np.testing.assert_allclose(step.x_update, np.full((1, 1, 1), 0.1))
