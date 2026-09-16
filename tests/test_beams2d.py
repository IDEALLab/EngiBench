"""Tests for the Beams2D problem.

`optimize` records, for every step, the design it was evaluated at, the filtered
sensitivities there, and the move the optimizer made from it. These tests pin each
field to the step it belongs to.

A small grid and few iterations keep this fast.
"""

from itertools import pairwise

import numpy as np
import pytest

from engibench.problems.beams2d import Beams2D

NELX = 20
NELY = 10
MAX_ITER = 4
VOLFRAC = 0.5
N_ELEMS = NELX * NELY


@pytest.fixture(scope="module")
def problem() -> Beams2D:
    return Beams2D(seed=0, config={"nelx": NELX, "nely": NELY, "volfrac": VOLFRAC})


@pytest.fixture(scope="module")
def optimized(problem: Beams2D) -> tuple[np.ndarray, list]:
    """Run optimize once and reuse across tests (the expensive step)."""
    problem.reset(seed=0)
    start = np.full(problem.design_space.shape, VOLFRAC, dtype=np.float64)
    # max_iter goes through optimize rather than the constructor: it lives on
    # Config, not SimulateConfig, so optimize rebuilds it from the default.
    return problem.optimize(start, config={"max_iter": MAX_ITER})


def test_history_is_numbered_from_zero(optimized: tuple) -> None:
    _opt, history = optimized
    assert 1 <= len(history) <= MAX_ITER
    assert [step.step for step in history] == list(range(len(history)))


def test_every_step_records_design_sensitivities_and_update(optimized: tuple) -> None:
    _opt, history = optimized
    for step in history:
        assert step.design.shape == (N_ELEMS,)
        # `x` is the core field every generic consumer reads; `design` is kept
        # for the callers that already use it.
        assert step.x is step.design
        # One sensitivity per design variable, so it is shaped like the design.
        assert step.x_sensitivities.shape == step.x.shape
        assert step.x_update.shape == (N_ELEMS,)


def test_sensitivities_are_nonpositive(optimized: tuple) -> None:
    """Compliance falls as material is added, and the optimizer clips dc at zero."""
    _opt, history = optimized
    for step in history:
        assert np.all(step.x_sensitivities <= 0.0)
        assert np.all(np.isfinite(step.x_sensitivities))


def test_update_is_the_move_to_the_next_design(optimized: tuple) -> None:
    """x_update is the step the optimizer took, so it lands on the next design."""
    _opt, history = optimized
    for step, following in pairwise(history):
        np.testing.assert_allclose(step.x + step.x_update, following.x, atol=1e-12)


def test_objective_delta_is_filled_in_except_on_the_last_step(optimized: tuple) -> None:
    """A step's delta needs the next step's objective, so the last one has none."""
    _opt, history = optimized
    if len(history) < 2:  # noqa: PLR2004 - a single step has no delta to check
        pytest.skip("optimization converged in one step")
    for step, following in pairwise(history):
        np.testing.assert_allclose(step.obj_values_update, following.obj_values - step.obj_values)
    assert history[-1].obj_values_update is None
