"""Registry of all problems in EngiBench."""

import importlib
import os
import pkgutil
from typing import Any

from engibench.core import Problem
import engibench.problems


def list_problems(base_module: Any = engibench.problems) -> dict[str, type[Problem]]:
    """Return a dict containing all available Problem instances defined in submodules of `base_module`."""
    module_path = os.path.dirname(base_module.__file__)
    modules = pkgutil.iter_modules(path=[module_path], prefix="")
    return {m.name: extract_problem(importlib.import_module(f"{base_module.__package__}.{m.name}")) for m in modules}


def extract_problem(module: Any) -> type[Problem]:
    """Get a `Problem` class defined in a module.

    Raises an exception if the module contains multiple `Problem` classes.
    """
    problem_types = [
        o for o in vars(module).values() if isinstance(o, type) and issubclass(o, Problem) and o is not Problem
    ]
    try:
        (p,) = problem_types
    except ValueError:
        msg = f"Only one problem per module is allowed. Got {', '.join(p.__name__ for p in problem_types)}"
        raise ValueError(msg) from None
    return p


BUILTIN_PROBLEMS = list_problems()
