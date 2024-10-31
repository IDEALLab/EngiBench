"""
Generates markdown files for each problem in the problems directory.
"""

import os

from tqdm import tqdm

from engibench.utils.all_problems import all_problems


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


for problem_id, constructor in tqdm(all_problems.items()):
    print(f"Generating docs for {problem_id}, {constructor}")

    docstring = constructor.__doc__
    if not docstring:
        docstring = constructor.__class__.__doc__
    docstring = trim(docstring)

    problem_title = problem_id.replace("_", " ").title()
    problem_pkg = problem_id.replace("_", "").lower()

    objective_str = ""
    for obj, direction in constructor.possible_objectives:
        objective_str += f"{obj}: &uarr;<br>" if direction == "maximize" else f"{obj}: &darr;<br>"
    objective_str = objective_str[:-4]  # remove the last <br>

    title = f"# {problem_title}"
    prob_table = "|  |  |\n| --- | --- |\n"
    prob_table += f"| Objectives | {objective_str} |\n"
    prob_table += f"| Design space | {constructor.design_space} |\n"
    prob_table += f"| Dataset | [{constructor.dataset_id}](https://huggingface.co/datasets/{constructor.dataset_id}) |\n"
    prob_table += f"| Container | `{constructor.container_id}` |\n"
    prob_table += f"| Import | `from engibench.problems.{problem_pkg} import {constructor.__name__}` |\n"

    all_text = f"""{title}
{prob_table}
{docstring}
"""
    v_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "problems",
        problem_pkg + ".md",
    )
    file = open(v_path, "w+", encoding="utf-8")
    file.write(all_text)
    file.close()
