"""Sphinx extension to extract metadata from problems and autogenerate docs.

Example usage in md docs:
`````md
``` {problem}
airfoil
```
`````
"""

from collections.abc import Iterable, Iterator, Sequence
import contextlib
import dataclasses
import importlib.abc
import importlib.machinery
import inspect
import sys
from types import ModuleType
from typing import Any, ClassVar, get_type_hints, TYPE_CHECKING
import unittest.mock

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.domains import Domain
from sphinx.util.docutils import SphinxDirective
from sphinx.util.typing import ExtensionMetadata

if TYPE_CHECKING:
    from sphinx.builders import Builder
    from sphinx.environment import BuildEnvironment

MODULE_WHITELIST = frozenset(["engibench"])
MODULE_EXTRA_MEMBERS = {"networkx": ["Graph"], "gymnasium": ["spaces"]}


def setup(app: Sphinx) -> ExtensionMetadata:
    """Add extension to sphinx."""
    app.add_domain(ProblemDomain)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }


class Lead:
    """Option to specify a lead in the problem directive."""

    caption = "Lead"

    def __init__(self, value: str) -> None:
        self.handle = None
        self.name, self.handle = value.split(" @", 1) if " @" in value else (value, None)

    def to_node(self) -> nodes.Node:
        p = nodes.Text(self.name)
        if self.handle:
            node = nodes.paragraph()
            node += [
                p,
                nodes.Text(" "),
                nodes.reference(refuri=f"https://github.com/{self.handle}", text="@" + self.handle),
            ]
            return node
        return p


class ProblemTableDirective(SphinxDirective):
    option_spec: ClassVar[dict[str, Any]] = {"lead": Lead, "problem_id": str}

    def run(self) -> list[Any]:
        problem_id = self.options.get("problem_id") or problem_id_from_docname(self.env.docname)
        problem = import_problem(problem_id)
        ObjectiveDirection = import_objective_direction()  # noqa: N806

        docstring = inspect.getdoc(problem)

        image = nodes.image(uri=f"../_static/img/problems/{problem_id}.png", width="450px", align="center")

        objectives = [
            f"{obj}: ↑" if direction == ObjectiveDirection.MAXIMIZE else f"{obj}: ↓"
            for obj, direction in problem.objectives
        ]
        conditions = read_dataclass(problem.Conditions)

        lead = self.options.get("lead")

        tab_data = [
            ("Version", str(problem.version)),
            ("Design space", make_code(repr(problem.design_space))),
            ("Objectives", make_multiline(objectives)),
            ("Conditions", make_simple_field_list(conditions)),
            ("Dataset", make_link(problem.dataset_id, f"https://huggingface.co/datasets/{problem.dataset_id}")),
            ("Container", make_code(problem.container_id) if problem.container_id is not None else None),
            ("Import", make_code(f"from {problem.__module__} import {problem.__name__}")),
            *([("Lead", lead.to_node())] if lead is not None else []),
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

        return [image, *body, make_table(tab_data)]


def problem_id_from_docname(docname: str) -> str:
    _, problem_id = docname.rsplit("/", 1)
    return problem_id


class ConditionsDirective(SphinxDirective):
    option_spec: ClassVar[dict[str, Any]] = {"problem_id": str, "defaults": bool}

    def run(self) -> list[Any]:
        problem_id = self.options.get("problem_id") or problem_id_from_docname(self.env.docname)
        problem = import_problem(problem_id)

        conditions = read_dataclass(problem.Conditions)
        return [make_simple_field_list(conditions, defaults=self.options.get("defaults", False))]


class ProblemDomain(Domain):
    name = "problem"
    label = "Engibench Problem"

    directives: ClassVar[dict[str, SphinxDirective]] = {
        "table": ProblemTableDirective,
        "conditions": ConditionsDirective,
    }

    def resolve_any_xref(  # noqa: PLR0913
        self,
        env: "BuildEnvironment",  # noqa: ARG002
        fromdocname: str,  # noqa: ARG002
        builder: "Builder",  # noqa: ARG002
        target: str,  # noqa: ARG002
        node: addnodes.pending_xref,  # noqa: ARG002
        contnode: nodes.Element,  # noqa: ARG002
    ) -> list[tuple[str, nodes.reference]]:
        return []


def import_objective_direction() -> type[Any]:
    """Import the ObjectiveDirection enum without requiring engibench dependencies."""
    with mock_imports(MODULE_WHITELIST, extra_members=MODULE_EXTRA_MEMBERS):
        from engibench.core import ObjectiveDirection  # noqa: PLC0415

    return ObjectiveDirection


def import_problem(problem_id: str) -> Any:
    """Import problem metadata without requiring engibench dependencies."""
    with mock_imports(MODULE_WHITELIST, extra_members=MODULE_EXTRA_MEMBERS):
        from engibench.utils.all_problems import BUILTIN_PROBLEMS  # noqa: PLC0415

        return BUILTIN_PROBLEMS[problem_id]


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


@dataclasses.dataclass
class Field:
    name: str
    type: type | None
    default: Any
    doc: str | None


def make_field_list(fields: list[Field]) -> nodes.Node:
    node = addnodes.desc()
    for f in fields:
        f_node = addnodes.desc()
        node.append(f_node)
        signode = addnodes.desc_signature("", "")
        f_node.append(signode)
        signode += addnodes.desc_name(f.name, f.name)
        if f.type is not None:
            signode += addnodes.desc_annotation(
                directives.unchanged,
                "",
                addnodes.desc_sig_punctuation("", ": "),
                addnodes.desc_sig_space(),
                nodes.Text(f.type.__name__ if isinstance(f.type, type) else str(f.type)),
            )
        if f.default is not dataclasses.MISSING:
            signode += addnodes.desc_annotation(
                directives.unchanged,
                "",
                addnodes.desc_sig_punctuation("", " ="),
                addnodes.desc_sig_space(),
                nodes.Text(f.default),
            )
        if f.doc is not None:
            f_node.append(addnodes.desc_content("", nodes.Text(f.doc)))

    return node


def make_simple_field_list(fields: list[Field], *, defaults: bool = False) -> nodes.Node:
    node = nodes.bullet_list()
    for f in fields:
        item = nodes.list_item()
        node += item
        p = nodes.paragraph()
        p += nodes.literal(text=f.name)
        text_pieces = [f.doc, f"(default: {f.default})" if f.default is not dataclasses.MISSING and defaults else None]
        if f.doc is not None:
            p += nodes.Text(": " + " ".join([piece for piece in text_pieces if piece]))
        item += p

    return node


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
    """Determine the indent of a lines"""
    stripped = line.lstrip()
    if stripped:
        return len(line) - len(stripped)
    return None


def read_dataclass(c: type) -> list[Field]:
    """Read the fields of a dataclass including docstrings for attributes."""
    docs = read_field_docstrings(c)
    types = get_type_hints(c)
    fields = dataclasses.fields(c)
    return [Field(name=f.name, default=f.default, doc=docs.get(f.name), type=types.get(f.name)) for f in fields]


def read_field_docstrings(c: type) -> dict[str, str]:  # noqa C903
    """Read field docstrings from a dataclass."""
    src = inspect.getsource(c)
    indent = ((line_indent(src) or 0) + 4) * " "

    def find_line_start(src: str) -> str | None:
        pos = src.find("\n" + indent)
        return None if pos == -1 else src[pos + len(indent) + 1 :]

    def field_name(line: str) -> tuple[str, str | None]:
        try:
            name, rest = line.split(": ", 1)
        except ValueError:
            return line, None
        return (rest, name) if name.isidentifier() else (line, None)

    def docstr(line: str) -> tuple[str, str | None]:
        if not line.startswith('"""'):
            return line, None
        pos = line.find('"""', 3)
        if pos == -1:
            raise ValueError("Unterminated docstring found")
        return line[pos + 3 :], line[3:pos]

    def tokenize(src: str) -> Iterable[tuple[str, str]]:
        rest: str | None = src
        f_name: str | None = None
        while rest:
            rest = find_line_start(rest)
            if rest is None:
                break
            rest, new_f_name = field_name(rest)
            if new_f_name is not None:
                f_name = new_f_name
                continue
            if f_name is not None:
                rest, docstring = docstr(rest)
                if docstring is not None:
                    yield f_name, docstring
                    f_name = None

    return dict(tokenize(src))


@contextlib.contextmanager
def mock_imports(whitelist: frozenset[str], extra_members: dict[str, list[str]] | None = None) -> Iterator[None]:
    """Add an import hook just after the builtin modules hook and the frozen module hook:
    https://docs.python.org/3/reference/import.html#the-meta-path
    """
    sys.meta_path.insert(2, MockFinder(whitelist, extra_members))
    yield
    del sys.meta_path[2]


class MockFinder(importlib.abc.MetaPathFinder):
    """Import hook which loads a mock instead of a module if the module is not engibench."""

    def __init__(self, whitelist: frozenset[str], extra_members: dict[str, list[str]] | None = None) -> None:
        self.whitelist = whitelist
        self.mock_loader = MockLoader(extra_members)
        super().__init__()

    def find_spec(
        self, fullname: str, _path: Sequence[str] | None, _target: ModuleType | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname.split(".", 1)[0] in self.whitelist:
            return None
        return importlib.machinery.ModuleSpec(fullname, self.mock_loader)


class MockLoader(importlib.abc.Loader):
    """Module loader, preparing the mocks."""

    def __init__(self, extra_members: dict[str, list[str]] | None = None) -> None:
        self.extra_members = extra_members or {}
        super().__init__()

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        # The mock must have the __path__ attribute, otherwise the import
        # system will fail when trying a submodule of the mock module:
        extra_args = {key: unittest.mock.Mock() for key in self.extra_members.get(spec.name, [])}
        return unittest.mock.Mock(spec=ModuleType, __path__=unittest.mock.Mock(), **extra_args)

    def exec_module(self, module: ModuleType) -> None:
        pass
