"""Registry of all problems in EngiBench."""

import importlib
import os
import pkgutil
from typing import Any

from engibench.core import Problem
import engibench.problems


def list_problems(base_module: Any = engibench.problems) -> dict[str, type[Problem]]:
    """Return a dict containing all available Problem classes defined in submodules of `base_module`."""
    module_path = os.path.dirname(base_module.__file__)
    modules = pkgutil.iter_modules(path=[module_path], prefix="")

    out: dict[str, type[Problem]] = {}
    for m in modules:
        modname = f"{base_module.__package__}.{m.name}"
        try:
            module = importlib.import_module(modname)
        except (ModuleNotFoundError, ImportError):
            # Optional dependency missing (e.g., ceviche) or other import error
            continue

        p = extract_problem(module)
        if p is None:
            continue

        out[m.name] = p

    return out


def extract_problem(module: Any) -> type[Problem] | None:
    """Get a `Problem` class defined in a module.

    Returns None if the module contains no `Problem` classes.
    Raises an exception if the module contains multiple `Problem` classes.
    """
    problem_types = [
        o for o in vars(module).values() if isinstance(o, type) and issubclass(o, Problem) and o is not Problem
    ]

    if len(problem_types) == 0:
        return None

    if len(problem_types) > 1:
        msg = f"Only one problem per module is allowed. Got {', '.join(p.__name__ for p in problem_types)}"
        raise ValueError(msg)

    return problem_types[0]

BUILTIN_PROBLEMS = list_problems()
