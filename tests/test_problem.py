from dataclasses import dataclass
from typing import Annotated

from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray
import pytest

from engibench.constraint import bounded
from engibench.constraint import constraint
from engibench.constraint import Criticality
from engibench.constraint import IMPL
from engibench.constraint import THEORY
from engibench.constraint import Violations
from engibench.core import Problem


class FakeProblem(Problem[NDArray[np.float64]]):
    version = 1
    objectives = ()
    conditions = ()
    design_space = spaces.Box(low=0.0, high=1.0, shape=(2, 3), dtype=np.float64)
    container_id = None

    @dataclass
    class Config:
        x: Annotated[int, bounded(lower=10)]
        y: Annotated[float, bounded(upper=-1.0)] = -1.0


def causes(violations: Violations) -> list[str]:
    return [v.cause for v in violations.violations]


def test_check_constraints_detects_violations() -> None:
    design = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    config = {"x": 1, "y": 0.0}
    violations = FakeProblem().check_constraints(design, config)
    assert causes(violations) == ["Config.x: 1 ∉ [10, ∞]", "Config.y: 0.0 ∉ [-∞, -1.0]", "design ∉ design_space"]
    expected_n_constraints = 3
    assert violations.n_constraints == expected_n_constraints


def test_check_constraints_detects_invalid_parameters() -> None:
    design = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    config = {"x": 10, "y": -1.0, "z": None}
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'z'"):
        FakeProblem().check_constraints(design, config)


def test_check_constraints_detects_missing_parameters() -> None:
    design = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    config = {"y": -1.0}
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'x'"):
        FakeProblem().check_constraints(design, config)


def test_check_constraints_handles_constraints_on_design_and_config() -> None:
    @constraint(categories=THEORY, criticality=Criticality.Warning)
    def volume_fraction_bound(design: NDArray, volfrac: float) -> None:
        """Constraint for volume fraction of the design."""
        actual_volfrac = design.mean()
        tolerance = 1e-6
        assert abs(actual_volfrac - volfrac) <= tolerance, (
            f"Volume fraction {actual_volfrac:.4f} does not match target {volfrac:.4f}"
        )

    class FakeProblem(Problem[NDArray[np.float64]]):
        version = 1
        objectives = ()
        conditions = ()
        design_space = spaces.Box(low=0.0, high=1.0, shape=(2, 3), dtype=np.float64)
        container_id = None

        design_constraints = (volume_fraction_bound,)

        @dataclass
        class Config:
            volfrac: Annotated[
                float,
                bounded(lower=0.0, upper=1.0).category(THEORY),
                bounded(lower=0.1, upper=0.9).warning().category(IMPL),
            ] = 0.35

    design = np.full((2, 3), 1.0)
    config = {"volfrac": 2.0}
    violations = FakeProblem().check_constraints(design, config)
    messages = causes(violations)
    messages[2] = messages[2].replace("np.float64(1.0)", "1.0")
    assert messages == [
        "Config.volfrac: 2.0 ∉ [0.0, 1.0]",
        "Config.volfrac: 2.0 ∉ [0.1, 0.9]",
        "Volume fraction 1.0000 does not match target 2.0000\nassert 1.0 <= 1e-06\n +  where 1.0 = abs((1.0 - 2.0))",
    ]
    expected_n_constraints = 4
    assert violations.n_constraints == expected_n_constraints
