from dataclasses import dataclass
from typing import Annotated

from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray
import pytest

from engibench.constraint import bounded
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
