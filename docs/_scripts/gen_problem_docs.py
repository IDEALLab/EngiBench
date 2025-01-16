"""
Generates markdown files for each problem in the problems directory.
"""

import os
import sys

from tqdm import tqdm

from engibench.core import Problem
from engibench.utils.all_problems import BUILTIN_PROBLEMS


def trim(docstring: str) -> str:
    if not docstring:
        return ""
    # Convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    lines = docstring.expandtabs().splitlines()
    # Determine minimum indentation (first line doesn't count):
    indent = 232323
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < 232323:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)
    # Return a single string:
    return "\n".join(trimmed)


def module(problem: type[Problem]) -> str:
    module = sys.modules[problem.__module__]
    if module.__package__ is None:
        return problem.__module__
    # Check if there is a shortcut alias in the parent module:
    parent_module = sys.modules[module.__package__]
    if getattr(parent_module, problem.__name__) is problem:
        return module.__package__
    return problem.__module__


# Only produces the latest version of each problem
filtered_problems: dict[str, type[Problem]] = {}
for problem_id, problem in BUILTIN_PROBLEMS.items():
    if problem_id not in filtered_problems or problem.version > filtered_problems[problem_id].version:
        filtered_problems[problem_id] = problem

print(filtered_problems)


# Now generates the markdown files
for problem_id, problem in tqdm(filtered_problems.items()):
    print(f"Generating docs for {problem_id}, {problem.version}, {problem}")

    docstring = problem.__doc__
    docstring = trim(docstring) if docstring is not None else None

    problem_title = f"{problem_id} {problem.version}".title()

    objective_str = ""
    for obj, direction in problem.possible_objectives:
        objective_str += f"{obj}: &uarr;<br>" if direction == "maximize" else f"{obj}: &darr;<br>"
    objective_str = objective_str[:-4]  # remove the last <br>

    boundary_str = ""
    for cond, value in problem.boundary_conditions:
        boundary_str += f"{cond}: {value}<br>"
    boundary_str = boundary_str[:-4]  # remove the last <br>

    title = f"# {problem_title}"
    prob_table = "|  |  |\n| --- | --- |\n"
    prob_table += f"| Design space | {problem.design_space} |\n"
    prob_table += f"| Objectives | {objective_str} |\n"
    prob_table += f"| Boundary conditions | {boundary_str} |\n"
    prob_table += f"| Dataset | [{problem.dataset_id}](https://huggingface.co/datasets/{problem.dataset_id}) |\n"
    prob_table += f"| Container | `{problem.container_id}` |\n"
    prob_table += f"| Import | `from {module(problem)} import {problem.__name__}` |\n"

    all_text = f"""{title}
{prob_table}
{docstring}
"""
    v_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "problems",
        problem_id + ".md",
    )
    with open(v_path, "w+", encoding="utf-8") as stream:
        stream.write(all_text)
