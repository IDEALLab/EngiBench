# ruff: noqa: SLF001
# These are whitebox tests: a few of them inspect the problem's internal geometry / parameters
# (``_bg_rho``, ``_design_region``, ``_epsr_min`` ...) to prove that simulate translates a design
# to permittivity by scaling only (no projection). Accessing those members is intentional here.

"""Tests for the Photonics2D v1 problem.

v1 guarantees that ``simulate`` and ``optimize`` are mutually consistent: ``simulate`` runs a design
as-is (no projection), and ``optimize`` returns the physical (projected) density, so
``simulate(optimize(x)[0]) == history[-1]`` and ``history[0] == simulate(x)``.

These tests use a small grid and few optimization steps to stay fast.
"""

import numpy as np
import pytest

from engibench.problems.photonics2d import Photonics2D
from engibench.problems.photonics2d.backend import design_to_epsr
from engibench.problems.photonics2d.backend import operator_proj
from engibench.problems.photonics2d.backend import poly_ramp
from engibench.problems.photonics2d.v0 import Photonics2D as Photonics2D_v0

NUM_X = 90
NUM_Y = 110
NUM_STEPS = 3
TOL = 1e-9


@pytest.fixture(scope="module")
def problem() -> Photonics2D:
    return Photonics2D(num_elems_x=NUM_X, num_elems_y=NUM_Y, seed=0)


@pytest.fixture(scope="module")
def start_design(problem: Photonics2D) -> np.ndarray:
    design, _ = problem.random_design(noise=0.001)
    return design


@pytest.fixture(scope="module")
def optimized(problem: Photonics2D, start_design: np.ndarray) -> tuple[np.ndarray, list]:
    """Run optimize once and reuse across tests (the expensive step)."""
    return problem.optimize(start_design, config={"num_optimization_steps": NUM_STEPS})


# --------------------------------------------------------------------- versioning


def test_exported_version_is_v1() -> None:
    assert Photonics2D.version == 1
    assert Photonics2D_v0.version == 0


# --------------------------------------------------------------------- consistency


def test_step0_equals_simulate(problem: Photonics2D, start_design: np.ndarray, optimized: tuple) -> None:
    """history[0] is the raw starting point run as-is, so it must equal simulate(start)."""
    _opt, history = optimized
    assert history[0].step == 0
    sim_start = float(problem.simulate(start_design)[0])
    assert history[0].obj_values[0] == pytest.approx(sim_start, rel=1e-6)


def test_roundtrip_simulate_equals_last_history(problem: Photonics2D, optimized: tuple) -> None:
    """simulate(returned design) reproduces the final optimization-history value."""
    opt, history = optimized
    sim_opt = float(problem.simulate(opt)[0])
    assert history[-1].obj_values[0] == pytest.approx(sim_opt, rel=1e-4)


def test_simulate_is_deterministic(problem: Photonics2D, optimized: tuple) -> None:
    """Re-simulating the same design gives the same value (no hidden state / drift)."""
    opt, _history = optimized
    assert float(problem.simulate(opt)[0]) == pytest.approx(float(problem.simulate(opt)[0]), rel=1e-9)


def test_simulate_does_not_project(problem: Photonics2D) -> None:
    """simulate must translate density -> permittivity by scaling only (no blur/projection)."""
    design = np.full((NUM_X, NUM_Y), 0.5, dtype=np.float64)
    problem.simulate(design)  # populates problem._last_epsr and the domain geometry
    expected_epsr = design_to_epsr(design, problem._bg_rho, problem._design_region, problem._epsr_min, problem._epsr_max)
    np.testing.assert_allclose(problem._last_epsr, expected_epsr)


# --------------------------------------------------------------------- bounds / dtype


def test_history_shape_and_steps(optimized: tuple) -> None:
    _opt, history = optimized
    assert [h.step for h in history] == list(range(NUM_STEPS + 1))
    for h in history:
        assert h.obj_values.shape == (1,)


def test_optimized_design_in_bounds(problem: Photonics2D, optimized: tuple) -> None:
    opt, _history = optimized
    assert opt.dtype == problem.design_space.dtype
    assert opt.min() >= 0.0
    assert opt.max() <= 1.0
    assert problem.design_space.contains(opt)


def test_simulate_rejects_out_of_bounds(problem: Photonics2D) -> None:
    bad = np.full((NUM_X, NUM_Y), 0.5, dtype=np.float64)
    bad[0, 0] = 2.0
    with pytest.raises(ValueError, match="constraint"):
        problem.simulate(bad)


def test_optimize_rejects_out_of_bounds(problem: Photonics2D) -> None:
    bad = np.full((NUM_X, NUM_Y), 0.5, dtype=np.float64)
    bad[0, 0] = -1.0
    with pytest.raises(ValueError, match="constraint"):
        problem.optimize(bad, config={"num_optimization_steps": 1})


# --------------------------------------------------------------------- backend units


def test_poly_ramp_endpoints() -> None:
    assert poly_ramp(0, max_iter=10, b0=1.0, bmax=300.0, degree=2) == pytest.approx(1.0)
    assert poly_ramp(10, max_iter=10, b0=1.0, bmax=300.0, degree=2) == pytest.approx(300.0)
    ramp = [poly_ramp(t, max_iter=10, b0=1.0, bmax=300.0, degree=2) for t in range(11)]
    assert all(ramp[i] <= ramp[i + 1] for i in range(len(ramp) - 1))  # monotonic non-decreasing


def test_operator_proj_maps_unit_interval() -> None:
    """Normalized tanh-Heaviside maps [0,1] -> [0,1] exactly at its anchors and stays inside."""
    assert operator_proj(np.array(0.0), eta=0.5, beta=50) == pytest.approx(0.0)
    assert operator_proj(np.array(0.5), eta=0.5, beta=50) == pytest.approx(0.5)
    assert operator_proj(np.array(1.0), eta=0.5, beta=50) == pytest.approx(1.0)
    grid = np.linspace(0.0, 1.0, 21)
    projected = operator_proj(grid, eta=0.5, beta=50)
    assert projected.min() >= -TOL
    assert projected.max() <= 1.0 + TOL


def test_operator_proj_escapes_outside_unit_interval() -> None:
    """Documents why the optimizer must clip rho: out-of-range input leaves [0,1].

    Uses a moderate beta (the early-optimization regime). At very large beta the tanh saturates
    and clamps instead, but during continuation beta is small and out-of-range rho escapes [0,1].
    """
    assert operator_proj(np.array(-0.2), eta=0.5, beta=2) < 0.0
    assert operator_proj(np.array(1.2), eta=0.5, beta=2) > 1.0


def test_design_to_epsr_is_linear_scaling(problem: Photonics2D) -> None:
    """design_to_epsr maps density 0/1 to epsr_min/epsr_max within the design region."""
    problem._setup_simulation(None)  # populate geometry
    zeros = np.zeros((NUM_X, NUM_Y))
    ones = np.ones((NUM_X, NUM_Y))
    epsr_zeros = design_to_epsr(zeros, problem._bg_rho, problem._design_region, problem._epsr_min, problem._epsr_max)
    epsr_ones = design_to_epsr(ones, problem._bg_rho, problem._design_region, problem._epsr_min, problem._epsr_max)
    design_mask = problem._design_region.astype(bool)
    assert np.allclose(epsr_zeros[design_mask], problem._epsr_min)
    assert np.allclose(epsr_ones[design_mask], problem._epsr_max)
