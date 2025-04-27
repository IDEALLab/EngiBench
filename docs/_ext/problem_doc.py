"""Sphinx extension to extract metadata from problems and autogenerate docs.

Example usage in md docs:
`````md
``` {problem}
airfoil
```
`````
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import contextlib
import importlib.abc
import importlib.machinery
import inspect
import sys
from types import ModuleType
from typing import Any
import unittest.mock

from docutils import nodes
from sphinx.application import Sphinx
from gymnasium import spaces # noqa: F401 this is used to import spaces before mock_imports in BUILTIN_PROBLEMS
from sphinx.util.docutils import SphinxDirective


def setup(app: Sphinx) -> None:
    """Add extension to sphinx."""
    app.add_directive("problem", ProblemDirective)


class ProblemDirective(SphinxDirective):
    required_arguments = 1

    def run(self) -> list[Any]:
        with mock_imports():
            from engibench.core import ObjectiveDirection
            from engibench.utils.all_problems import BUILTIN_PROBLEMS

        problem_id = self.arguments[0].strip()
        problem = BUILTIN_PROBLEMS[problem_id]
        docstring = unindent(problem.__doc__) if problem.__doc__ is not None else ""
        docstring = inspect.cleandoc(docstring)

        image = nodes.image(uri=f"../_static/img/problems/{problem_id}.png", width="450px", align="center")

        objectives = [f"{obj}: ↑" if direction == ObjectiveDirection.MAXIMIZE else f"{obj}: ↓" for obj, direction in problem.objectives]
        conditions = [f"{cond}: {value}" for cond, value in problem.conditions]

        tab_data = [
            ("Version", str(problem.version)),
            ("Design space", make_code(repr(problem.design_space))),
            ("Objectives", make_multiline(objectives)),
            ("Conditions", make_multiline(conditions)),
            ("Dataset", make_link(problem.dataset_id, f"https://huggingface.co/datasets/{problem.dataset_id}")),
            ("Container", make_code(problem.container_id) if problem.container_id is not None else None),
            ("Import", make_code(f"from {problem.__module__} import {problem.__name__}")),
        ]

        # Very ugly hack to retain the order of children
        # (`self.parse_text_to_nodes` will directly add subsections to the doctree
        # and will only return leading paragraphs):
        sec = self.state.document.children[0]
        header = sec.deepcopy()
        body = [] if docstring is None else self.parse_text_to_nodes(docstring, allow_section_headings=True)
        body += sec.deepcopy().children[len(header) :]
        sec.clear()
        sec.extend(header)

        return [image, make_table(tab_data), *body]


def make_section(title: str, section_id: str, body: list[Any]) -> nodes.section:
    sec = nodes.section(ids=[section_id])
    sec += nodes.title(text=title)
    for element in body:
        sec += element
    return sec


def make_link(text: str, uri: str) -> nodes.paragraph:
    link = nodes.reference(refuri=uri, text=text)
    return nodes.paragraph("", "", link)


def make_multiline(lines: list[str]) -> list[nodes.paragraph]:
    return [nodes.paragraph(text=text) for text in lines]


def make_code(text: str) -> nodes.paragraph:
    return nodes.paragraph("", "", nodes.literal(text=text))


def make_table(tab_data: list[tuple[str, Any]]) -> nodes.table:
    table = nodes.table()
    tgroup = nodes.tgroup(cols=2)
    table += tgroup
    tgroup += nodes.colspec()
    tgroup += nodes.colspec()
    tbody = nodes.tbody()
    for key, val in tab_data:
        if val is None:
            continue
        row = nodes.row()
        row += nodes.entry("", nodes.paragraph(text=key))
        p = [nodes.paragraph(text=val)] if isinstance(val, str) else val
        if not isinstance(p, list):
            p = [p]
        row += nodes.entry("", *p)
        tbody += row
    tgroup += tbody
    return table


def unindent(docstring: str) -> str:
    if not docstring:
        return ""
    # Convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    lines = docstring.expandtabs().splitlines()
    # Determine minimum indentation (first line doesn't count):
    line_indents = (line_indent(line) for line in lines[1:])
    indent = 0 if not line_indents else min(indent for indent in line_indents if indent is not None)
    # Remove indentation (first line is special):
    trimmed = [lines[0].strip(), *(line[indent:].rstrip() for line in lines[1:])]
    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)
    # Return a single string:
    return "\n".join(trimmed)


def line_indent(line: str) -> int | None:
    stripped = line.lstrip()
    if stripped:
        return len(line) - len(stripped)
    return None


@contextlib.contextmanager
def mock_imports() -> Iterator[None]:
    """Add an import hook just after the builtin modules hook and the frozen module hook:
    https://docs.python.org/3/reference/import.html#the-meta-path
    """
    sys.meta_path.insert(2, MockFinder())
    yield
    del sys.meta_path[2]


class MockFinder(importlib.abc.MetaPathFinder):
    """Import hook which loads a mock instead of a module if the module is not engibench."""

    def find_spec(
        self, fullname: str, _path: Sequence[str] | None, _target: ModuleType | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname.split(".", 1)[0] == "engibench":
            return None
        return importlib.machinery.ModuleSpec(fullname, MockLoader())


class MockLoader(importlib.abc.Loader):
    """Module loader, preparing the mocks."""

    def create_module(self, _spec: importlib.machinery.ModuleSpec) -> ModuleType:
        # The mock must have the __path__ attribute, otherwise the import
        # system will fail when trying a submodule of the mock module:
        return unittest.mock.Mock(spec=ModuleType, __path__=unittest.mock.Mock())

    def exec_module(self, module: ModuleType) -> None:
        pass
