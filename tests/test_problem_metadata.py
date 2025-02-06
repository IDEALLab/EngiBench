from __future__ import annotations

from typing import get_args, get_origin

import pytest

from engibench.core import Problem
from engibench.utils.all_problems import BUILTIN_PROBLEMS


@pytest.mark.parametrize("problem", BUILTIN_PROBLEMS.values())
def test_problem_metadata(problem: type[Problem]) -> None:
    """Check that all builtin problems define all metadata class attributes."""
    # Check generic parameters of Problem[]:
    (base,) = getattr(problem, "__orig_bases__", (None,))
    assert (
        get_origin(base) is Problem
    ), f"Problem {problem.__name__} does not specify generic parameters for the base class `Problem`"
    type_vars = Problem.__parameters__  # type: ignore[attr-defined]
    generics = get_args(base)
    assert len(generics) == len(
        type_vars
    ), f"Problem {problem.__name__} must specify {len(type_vars)} generic parameters for the base class `Problem`"

    # Check if all metadata fields are populated and have correct type:
    for attr in Problem.__annotations__:
        assert hasattr(problem, attr), f"Problem {problem.__name__} does not have an attribute {attr}"
