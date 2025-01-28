"""This file contains tests making sure the implemented problems respect the API."""

from __future__ import annotations

import inspect
from typing import get_args, get_origin

import gymnasium
import pytest

from engibench import Problem
from engibench.utils.all_problems import BUILTIN_PROBLEMS


@pytest.mark.parametrize("problem_class", BUILTIN_PROBLEMS.values())
def test_problem_impl(problem_class: type[Problem]) -> None:
    """Check that all builtin problems define all required class attributes and methods."""
    # Check generic parameters of Problem[]:
    (base,) = getattr(problem_class, "__orig_bases__", (None,))
    assert (
        get_origin(base) is Problem
    ), f"Problem {problem_class.__name__} does not specify generic parameters for the base class `Problem`"
    type_vars = Problem.__parameters__  # type: ignore[attr-defined]
    generics = get_args(base)
    assert len(generics) == len(
        type_vars
    ), f"Problem {problem_class.__name__} must specify {len(type_vars)} generic parameters for the base class `Problem`"

    problem: Problem = problem_class()
    # Test the attributes
    assert isinstance(
        problem.design_space, gymnasium.Space
    ), f"Problem {problem_class.__name__}: The design_space attribute should be a gymnasium.Space object."

    assert (
        len(problem.possible_objectives) >= 1
    ), f"Problem {problem_class.__name__}: The possible_objectives attribute should not be empty."
    assert all(
        isinstance(obj[0], str) and len(obj[0]) > 0 for obj in problem.possible_objectives
    ), f"Problem {problem_class.__name__}: The first element of each objective should be a non-emtpy string."

    assert (
        problem.dataset_id is not None and len(problem.dataset_id) > 0
    ), f"Problem {problem_class.__name__}: The dataset_id should be defined."

    # Test the required methods are implemented
    class_methods = {
        name
        for name, member in inspect.getmembers(type(problem))
        if inspect.isfunction(member) and member.__qualname__.startswith(type(problem).__name__ + ".")
    }
    assert "simulate" in class_methods, f"Problem {problem_class.__name__}: The simulate method should be implemented."
    assert "render" in class_methods, f"Problem {problem_class.__name__}: The render method should be implemented."
    assert (
        "random_design" in class_methods
    ), f"Problem {problem_class.__name__}: The random_design method should be implemented."
    assert "reset" in class_methods, f"Problem {problem_class.__name__}: The reset method should be implemented."
    # optimize is optional, thus not checked
