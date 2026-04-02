"""Tests for the engibench.transforms.xeltofab bridge module."""

import warnings

import numpy as np
import pytest

pytest.importorskip("xeltofab")

from xeltofab import PipelineParams

from engibench.transforms.xeltofab import PROBLEM_PRESETS
from engibench.transforms.xeltofab import save
from engibench.transforms.xeltofab import to_mesh
from engibench.transforms.xeltofab._validate import validate_input

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXPECTED_COLS = 3


def _make_circle_2d(shape: tuple[int, int] = (50, 100), radius: float = 0.3) -> np.ndarray:
    """Create a synthetic 2-D density field with a filled circle."""
    ny, nx = shape
    y, x = np.mgrid[:ny, :nx]
    cy, cx = ny / 2, nx / 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_dim = max(ny, nx)
    return np.where(dist < radius * max_dim, 1.0, 0.0)


def _make_sphere_3d(resolution: int = 20, radius: float = 0.35) -> np.ndarray:
    """Create a synthetic 3-D density field with a filled sphere."""
    coords = np.linspace(0, 1, resolution)
    z, y, x = np.meshgrid(coords, coords, coords, indexing="ij")
    dist = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2)
    return np.where(dist < radius, 1.0, 0.0)


def _make_fake_problem(name: str, volfrac: float | None = None):
    """Create a minimal stand-in for an EngiBench Problem.

    Returns a fresh class each time so ``type(obj).__name__`` is isolated.
    """

    class _Conditions:
        pass

    cond = _Conditions()
    if volfrac is not None:
        cond.volfrac = volfrac  # type: ignore[attr-defined]

    cls = type(name, (), {"conditions": cond})
    return cls()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidateInput:
    def test_rejects_non_array(self):
        with pytest.raises(TypeError, match="numpy ndarray"):
            validate_input([1, 2, 3])

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="2-D or 3-D"):
            validate_input(np.array([0.0, 1.0]))

    def test_rejects_4d(self):
        with pytest.raises(ValueError, match="2-D or 3-D"):
            validate_input(np.zeros((2, 2, 2, 2)))

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            validate_input(np.zeros((0, 10)))

    def test_rejects_nan(self):
        field = np.array([[0.5, float("nan")], [0.1, 0.2]])
        with pytest.raises(ValueError, match="non-finite"):
            validate_input(field)

    def test_rejects_non_numeric(self):
        field = np.array([["a", "b"], ["c", "d"]])
        with pytest.raises(TypeError, match="numeric dtype"):
            validate_input(field)

    def test_clips_out_of_range(self):
        field = np.array([[-0.1, 0.5], [0.8, 1.2]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_input(field)
            assert len(w) == 1
            assert "Clipping" in str(w[0].message)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_passes_valid(self):
        field = np.random.default_rng(0).random((10, 10))
        result = validate_input(field)
        np.testing.assert_array_equal(result, field)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


class TestPresets:
    def test_covers_all_density_problems(self):
        expected = {"Beams2D", "ThermoElastic2D", "ThermoElastic3D", "Photonics2D", "HeatConduction2D", "HeatConduction3D"}
        assert set(PROBLEM_PRESETS.keys()) == expected

    def test_all_presets_have_field_type_density(self):
        for name, preset in PROBLEM_PRESETS.items():
            assert preset.get("field_type") == "density", f"{name} preset missing field_type=density"


# ---------------------------------------------------------------------------
# to_mesh — 2-D
# ---------------------------------------------------------------------------


class TestToMesh2D:
    def test_synthetic_circle(self):
        problem = _make_fake_problem("Beams2D", volfrac=0.3)
        field = _make_circle_2d()
        state = to_mesh(problem, field, validate=False)
        assert state.contours is not None
        assert len(state.contours) > 0

    def test_with_validation(self):
        problem = _make_fake_problem("HeatConduction2D")
        field = _make_circle_2d()
        state = to_mesh(problem, field, validate=True, volume_tolerance=1.0)
        assert state.contours is not None


# ---------------------------------------------------------------------------
# to_mesh — 3-D
# ---------------------------------------------------------------------------


class TestToMesh3D:
    def test_synthetic_sphere(self):
        problem = _make_fake_problem("HeatConduction3D")
        field = _make_sphere_3d()
        state = to_mesh(problem, field, validate=False)
        assert state.vertices is not None
        assert state.faces is not None
        assert state.vertices.shape[1] == _EXPECTED_COLS
        assert state.faces.shape[1] == _EXPECTED_COLS

    def test_volume_preservation(self):
        problem = _make_fake_problem("HeatConduction3D")
        field = _make_sphere_3d(resolution=30, radius=0.35)
        state = to_mesh(problem, field, validate=True, volume_tolerance=0.15)
        assert state.vertices is not None

    def test_save_stl(self, tmp_path):
        problem = _make_fake_problem("HeatConduction3D")
        field = _make_sphere_3d()
        state = to_mesh(problem, field, validate=False)
        out = tmp_path / "test.stl"
        save(state, out)
        assert out.exists()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Parameter overrides
# ---------------------------------------------------------------------------

_CUSTOM_THRESHOLD = 0.3
_CUSTOM_SIGMA = 2.0


class TestParameterOverrides:
    def test_kwargs_override_preset(self):
        problem = _make_fake_problem("Beams2D")
        field = _make_circle_2d()
        state = to_mesh(problem, field, smooth_sigma=0.0, validate=False)
        assert state.params.smooth_sigma == 0.0

    def test_explicit_params_bypass_presets(self):
        problem = _make_fake_problem("Beams2D")
        field = _make_circle_2d()
        custom = PipelineParams(threshold=_CUSTOM_THRESHOLD, smooth_sigma=_CUSTOM_SIGMA)
        state = to_mesh(problem, field, params=custom, validate=False)
        assert state.params.threshold == _CUSTOM_THRESHOLD
        assert state.params.smooth_sigma == _CUSTOM_SIGMA

    def test_unknown_problem_uses_defaults(self):
        problem = _make_fake_problem("UnknownProblem")
        field = _make_circle_2d()
        state = to_mesh(problem, field, validate=False)
        assert state.contours is not None
