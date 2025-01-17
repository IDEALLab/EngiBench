"""This modules contains tests making sure the implemented problems respect the API."""

import inspect
import types

import gymnasium
import pytest

from engibench import Problem
import engibench.utils.all_problems


@pytest.mark.parametrize("problem_module", engibench.utils.all_problems.all_problems.values())
def test_api(problem_module: types.ModuleType) -> None:
    """Test the API of the given problem."""
    problem: Problem = problem_module.build()
    # Test the attributes
    assert isinstance(
        problem.design_space, gymnasium.Space
    ), "The design_space attribute should be a gymnasium.Space object."

    assert len(problem.possible_objectives) >= 1, "The possible_objectives attribute should not be empty."
    assert all(
        obj[1] == "minimize" or obj[1] == "maximize" for obj in problem.possible_objectives
    ), "The second element of each objective should be either 'minimize' or 'maximize'."
    assert all(
        isinstance(obj[0], str) for obj in problem.possible_objectives
    ), "The first element of each objective should be a string."

    assert problem.dataset_id is not None, "The dataset_id attribute should not be None."

    # Test the methods are implemented
    class_methods = {
        name
        for name, member in inspect.getmembers(type(problem))
        if inspect.isfunction(member) and member.__qualname__.startswith(type(problem).__name__ + ".")
    }
    assert "simulate" in class_methods, "The simulate method should be implemented."
    assert "render" in class_methods, "The render method should be implemented."
    assert "random_design" in class_methods, "The random_design method should be implemented."
    assert "reset" in class_methods, "The reset method should be implemented."
    # optimize is optional, thus not checked
