"""This file contains tests making sure the implemented problems respect the API."""

import dataclasses
import inspect
import os
import sys
from typing import Any, get_args, get_origin

import gymnasium
from gymnasium import spaces
import numpy as np
import pytest

from engibench import Problem
from engibench.utils.all_problems import BUILTIN_PROBLEMS


@pytest.mark.parametrize("problem_class", BUILTIN_PROBLEMS.values())
def test_problem_impl(problem_class: type[Problem]) -> None:
    """Check that all builtin problems define all required class attributes and methods."""
    print(f"Testing {problem_class.__name__}...")
    # Check generic parameters of Problem[]:
    (base,) = getattr(problem_class, "__orig_bases__", (None,))
    assert get_origin(base) is Problem, (
        f"Problem {problem_class.__name__} does not specify generic parameters for the base class `Problem`"
    )
    type_vars = Problem.__parameters__  # type: ignore[attr-defined]
    generics = get_args(base)
    assert len(generics) == len(type_vars), (
        f"Problem {problem_class.__name__} must specify {len(type_vars)} generic parameters for the base class `Problem`"
    )

    problem: Problem = problem_class()
    # Test the attributes
    assert isinstance(problem.design_space, gymnasium.Space), (
        f"Problem {problem_class.__name__}: The design_space attribute should be a gymnasium.Space object."
    )

    assert len(problem.objectives) >= 1, (
        f"Problem {problem_class.__name__}: The possible_objectives attribute should not be empty."
    )
    assert all(isinstance(obj[0], str) and len(obj[0]) > 0 for obj in problem.objectives), (
        f"Problem {problem_class.__name__}: The first element of each objective should be a non-emtpy string."
    )

    assert problem.dataset_id is not None, f"Problem {problem_class.__name__}: The dataset_id should be defined."
    assert len(problem.dataset_id) > 0, f"Problem {problem_class.__name__}: The dataset_id should positive."

    # Test the required methods are implemented
    class_methods = {
        name
        for name, member in inspect.getmembers(type(problem))
        if inspect.isfunction(member) and member.__qualname__.startswith(type(problem).__name__ + ".")
    }
    assert "simulate" in class_methods, f"Problem {problem_class.__name__}: The simulate method should be implemented."
    assert "render" in class_methods, f"Problem {problem_class.__name__}: The render method should be implemented."
    assert "random_design" in class_methods, (
        f"Problem {problem_class.__name__}: The random_design method should be implemented."
    )
    assert "reset" in class_methods, f"Problem {problem_class.__name__}: The reset method should be implemented."
    # optimize is optional, thus not checked

    # Test the dataset has the required splits
    dataset = problem.dataset
    assert "train" in dataset, f"Problem {problem_class.__name__}: The dataset should contain a 'train' split."
    assert "test" in dataset, f"Problem {problem_class.__name__}: The dataset should contain a 'test' split."
    assert "val" in dataset, f"Problem {problem_class.__name__}: The dataset should contain a 'val' split."
    # Test the dataset fields match `optimal_design`, `problem.conditions`, and `problem.objectives`
    if len(problem.objectives) > 1:
        for o, _ in problem.objectives:
            assert o in dataset["train"].column_names, (
                f"Problem {problem_class.__name__}: The dataset should contain the field {o}."
            )

    for f in dataclasses.fields(problem.Conditions):
        assert f.name in dataset["train"].column_names, (
            f"Problem {problem_class.__name__}: The dataset should contain the field {f.name}."
        )
    if problem_class.__module__.startswith("engibench.problems.power_electronics"):
        print(f"Skipping optimal design test for power electronics problem {problem_class.__name__}")
        return

    assert "optimal_design" in dataset["train"].column_names, (
        f"Problem {problem_class.__name__}: The dataset should contain the field 'optimal_design'."
    )
    print(f"Done testing {problem_class.__name__}.")


def problem_id(problem_class: type[Problem]) -> str:
    id_, _ = problem_class.__module__.removeprefix("engibench.problems.").split(".", 1)
    return id_


@pytest.mark.parametrize("problem_class", BUILTIN_PROBLEMS.values())
def test_python_problem_impl(problem_class: type[Problem]) -> None:
    """Check that all problems defined in Python files respect the API.

    This test verifies that:
    1. The problem simulation returns the correct number of objectives
    2. The optimization produces valid designs within the design space
    3. The optimization history contains valid objective values
    """
    if problem_class.container_id is not None and not sys.platform.startswith("linux"):
        pytest.skip(f"Skipping containerized problem {problem_class.__name__} on non-linux platform")
    print(f"Testing optimization and simulation for {problem_class.__name__}...")
    # Initialize problem and get a random design
    problem = problem_class(seed=1)
    design, _ = problem.random_design()

    # Test simulation outputs
    print(f"Simulating {problem_class.__name__}...")
    if problem_class.__module__.startswith("engibench.problems.airfoil"):
        objs = problem.simulate(design, mpicores=1)  # type: ignore[call-arg]
    else:
        objs = problem.simulate(design)

    expected_obj_count = len(problem.objectives)
    assert objs.shape[0] == expected_obj_count, (
        f"Problem {problem_class.__name__}: Simulation returned {objs.shape[0]} objectives "
        f"but expected {expected_obj_count}"
    )
    print(f"Done simulating {problem_class.__name__}.")
    # Test optimization outputs
    print(f"Optimizing {problem_class.__name__}...")
    # Skip optimization test for power electronics, airfoil, and heat conduction problems
    if problem_id(problem_class) == "airfoil":
        print(f"Skipping optimization test for {problem_class.__name__}")
        return
    problem.reset(seed=1)
    default_max_iter = 10
    max_iter = os.environ.get("ENGIBENCH_MAX_ITER", default_max_iter)
    max_iter_config: dict[str, Any] = {
        key: max_iter for key in ("max_iter", "num_optimization_steps") if hasattr(problem_class.Config, key)
    }
    try:
        optimal_design, history = problem.optimize(starting_point=design, config=max_iter_config)
    except NotImplementedError:
        print("Problem class {problem_class.__name__} does not implement optimize - Skipping optimize")
        return
    if isinstance(problem.design_space, spaces.Box):
        assert np.all(optimal_design >= problem.design_space.low), (
            f"Problem {problem_class.__name__}: The optimal design should be within the design space."
        )
        assert np.all(optimal_design <= problem.design_space.high), (
            f"Problem {problem_class.__name__}: The optimal design should be within the design space."
        )
        assert optimal_design.shape == problem.design_space.shape, (
            f"Problem {problem_class.__name__}: The optimal design should have the same shape as the design space."
        )
        assert np.can_cast(optimal_design.dtype, problem.design_space.dtype), (
            f"Problem {problem_class.__name__}: The optimal design should have the same dtype as the design space."
        )
    assert problem.design_space.contains(optimal_design), (
        f"Problem {problem_class.__name__}: The optimal design should be within the design space."
    )

    # Verify optimization history
    for step_num, optistep in enumerate(history):
        assert len(optistep.obj_values) == expected_obj_count, (
            f"Problem {problem_class.__name__}: Step {step_num} has {len(optistep.obj_values)} objectives "
            f"but expected {expected_obj_count}"
        )
    print(f"Done testing {problem_class.__name__}.")
