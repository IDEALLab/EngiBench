"""Per-problem policy for the shared problem suite.

This is a plain module rather than a test module so that a problem-specific test
can assert its own policy without importing -- and therefore executing -- the
shared suite, which imports every builtin problem and all of their dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engibench.core import Problem


@dataclass(frozen=True)
class ProblemTestPolicy:
    """Cost and portability policy for one problem in the shared suite."""

    exercise_optimization: bool = True
    """Whether `test_python_problem_impl` should also call `optimize()`."""

    optimization_reason: str = ""
    """Why optimization is skipped, printed when it is."""

    supported_machines: tuple[str, ...] | None = None
    """`platform.machine()` values the published runtime supports, or None for any."""

    slow: bool = False
    """Mark the problem's dataset and simulation tests as `slow`."""


DEFAULT_TEST_POLICY = ProblemTestPolicy()

PROBLEM_TEST_POLICIES = {
    "problems.airfoil.v0.Airfoil": ProblemTestPolicy(
        exercise_optimization=False,
        optimization_reason="optimization is not part of the shared Airfoil smoke test",
    ),
    "problems.mto2d.v0.MTO2D": ProblemTestPolicy(
        exercise_optimization=False,
        optimization_reason="the external 200-step optimization is too expensive for the shared smoke test",
        supported_machines=("x86_64", "amd64"),
        slow=True,
    ),
}


def problem_id(problem_class: type[Problem]) -> str:
    """Return the stable versioned identifier used by reference files and test policy."""
    return problem_class.__module__.removeprefix("engibench.") + "." + problem_class.__name__


def problem_test_policy(problem_class: type[Problem]) -> ProblemTestPolicy:
    """Return explicit shared-suite policy without broad module-prefix matching."""
    return PROBLEM_TEST_POLICIES.get(problem_id(problem_class), DEFAULT_TEST_POLICY)
