"""Tests for the Beams2D problem.

`optimize` records, for every step, the design it was evaluated at, the gradient
of the compliance there, and the move the optimizer made from it. These tests pin
each field to the step it belongs to.

A small grid and few iterations keep this fast.
"""

from itertools import pairwise

import numpy as np
import pytest

from engibench.problems.beams2d.v0 import Beams2D as Beams2D_v0
from engibench.problems.beams2d.v1 import Beams2D as Beams2D_v1

NELX = 20
NELY = 10
MAX_ITER = 4
VOLFRAC = 0.5
N_ELEMS = NELX * NELY


# Every test runs on both problem versions, with the overhang filter off and on.
# The overhang filter is where the recorded design and the optimizer's own
# variables differ the most, so it is the case the recorded fields must survive.
@pytest.fixture(
    scope="module",
    params=[(Beams2D_v0, False), (Beams2D_v0, True), (Beams2D_v1, False), (Beams2D_v1, True)],
    ids=["v0", "v0-overhang", "v1", "v1-overhang"],
)
def problem(request: pytest.FixtureRequest) -> Beams2D_v0:
    problem_class, overhang_constraint = request.param
    return problem_class(
        seed=0,
        config={"nelx": NELX, "nely": NELY, "volfrac": VOLFRAC, "overhang_constraint": overhang_constraint},
    )


@pytest.fixture(scope="module")
def optimized(problem: Beams2D_v0) -> tuple[np.ndarray, list]:
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
    """Compliance falls as material is added, so no entry of its gradient is positive."""
    _opt, history = optimized
    for step in history:
        assert np.all(step.x_sensitivities <= 0.0)
        assert np.all(np.isfinite(step.x_sensitivities))


def test_sensitivities_match_a_finite_difference_of_simulate(problem: Beams2D_v0, optimized: tuple) -> None:
    """The recorded sensitivities are the gradient of simulate() at the recorded design.

    Nudge the design a small distance along a random direction, both ways, and
    simulate each. The change in compliance per unit distance must equal the
    recorded sensitivities projected onto that direction.
    """
    _opt, history = optimized
    direction = np.random.default_rng(0).standard_normal(N_ELEMS)
    nudge = 1e-6
    for step in history:
        problem.reset(seed=0)
        compliance_ahead = problem.simulate(step.x + nudge * direction)
        problem.reset(seed=0)
        compliance_behind = problem.simulate(step.x - nudge * direction)
        measured_slope = (compliance_ahead - compliance_behind) / (2 * nudge)
        recorded_slope = step.x_sensitivities @ direction
        np.testing.assert_allclose(measured_slope, recorded_slope, rtol=1e-4)


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
