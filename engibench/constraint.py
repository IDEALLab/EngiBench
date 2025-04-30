"""Constraints for parameters of Problem classes."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import dataclasses
from dataclasses import dataclass
from dataclasses import is_dataclass
from enum import auto
from enum import Enum
from enum import Flag
import inspect
import typing
from typing import Any, ClassVar, overload, TypeVar

import numpy as np

Check = Callable[..., None]


class Category(Flag):
    """Category of a constraint."""

    Theory = auto()
    Implementation = auto()


IMPL = Category.Implementation
THEORY = Category.Theory
UNCATEGORIZED = Category(0)


class Criticality(Enum):
    """Criticality of a constraint violation."""

    Error = auto()
    Warning = auto()


@dataclass
class Constraint:
    """Constraint for parameters passed to e.g. :method:`engibench.core.Problem.optimize()`."""

    check: Check
    """Check callback raising an AssertError if the constraint is violated."""
    categories: Category = UNCATEGORIZED
    """Categories of the constraint."""
    criticality: Criticality = Criticality.Error
    """Criticality of a violation of the constraint."""

    def category(self, category: Category) -> Constraint:
        """Return a copy of the constraint which has the specified category added."""
        return Constraint(check=self.check, criticality=self.criticality, categories=self.categories | category)

    def warning(self) -> Constraint:
        """Return a copy of the constraint with the criticality level set to "warning"."""
        return Constraint(check=self.check, criticality=Criticality.Warning, categories=self.categories)

    def check_dict(self, parameter_args: dict[str, Any]) -> Violation | None:
        """Check for a violation of the given constraint for the given parameters."""
        # We first inspect the arguments of check callback:
        sig = inspect.signature(self.check)
        required_args = {
            p.name
            for p in sig.parameters.values()
            if p.default is p.empty and p.kind not in {p.VAR_KEYWORD, p.VAR_POSITIONAL}
        }
        missing_args = frozenset(required_args) - frozenset(parameter_args)
        if missing_args:
            msg = f"Missing argument(s) {', '.join(missing_args)} for constraint {self.check.__name__}"
            raise TypeError(msg)

        # If self.check() accepts `**kwargs`, we feed in **parameter_args:
        if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
            constraint_args = parameter_args
        # Otherwise, we only feed in parameters defined in the signature:
        else:
            constraint_args = {param: value for param, value in parameter_args.items() if param in sig.parameters}
        try:
            self.check(**constraint_args)
        except AssertionError as e:
            return Violation(self, str(e))
        return None

    def check_value(self, value: Any) -> Violation | None:
        """Check for a violation for the given single positional value."""
        try:
            self.check(value)
        except AssertionError as e:
            return Violation(self, str(e))
        return None


@overload
def constraint(check: Check, /) -> Constraint: ...


@overload
def constraint(
    *, categories: Category = UNCATEGORIZED, criticality: Criticality = Criticality.Error
) -> Callable[[Check], Constraint]: ...


def constraint(
    check: Check | None = None,
    /,
    *,
    categories: Category = UNCATEGORIZED,
    criticality: Criticality = Criticality.Error,
) -> Callable[[Check], Constraint] | Constraint:
    """Decorator for check callbacks to convert the callback to a :class:`Constraint`."""
    if check is not None:
        return Constraint(_unwrap_staticmethod(check))

    def decorator(check: Check) -> Constraint:
        return Constraint(_unwrap_staticmethod(check), categories=categories, criticality=criticality)

    return decorator


def _unwrap_staticmethod(f: Check | staticmethod) -> Check:
    """This is needed for Python 3.9."""
    if isinstance(f, staticmethod):
        return f.__func__
    return f


class Var:
    """Helper class to bind variable names to a constraint."""

    def __init__(self, *names: str) -> None:
        self.names = names

    def check(self, constraint: Constraint) -> Constraint:
        """Bind the variable names to `constraint`."""
        name = "_".join([*self.names, constraint.check.__name__])

        def extracting_check(**kwargs) -> None:
            try:
                values = [kwargs[name] for name in self.names]
            except KeyError as e:
                msg = f"Missing argument {e} for constraint {name}"
                raise AssertionError(msg) from None
            constraint.check(*values)

        extracting_check.__name__ = name
        return dataclasses.replace(constraint, check=extracting_check)


@dataclass
class Violation:
    """Representation of a violation of a constraint."""

    constraint: Constraint
    cause: str

    def __str__(self) -> str:
        return f"{self.constraint.check.__name__}: {self.cause}"


class Violations:
    """Filterable collection of :class:`Violation` instances returned by :function:`check_constraints`."""

    def __init__(self, violations: list[Violation], n_constraints: int) -> None:
        self.violations = violations
        self.n_constraints = n_constraints

    def by_category(self, category: Category) -> Violations:
        """Filter the violations by the category of the constraint causing the violation."""
        return Violations(
            [violation for violation in self.violations if category in violation.constraint.categories], self.n_constraints
        )

    def by_criticality(self, criticality: Criticality) -> Violations:
        """Filter the violations by criticality."""
        return Violations(
            [violation for violation in self.violations if violation.constraint.criticality == criticality],
            self.n_constraints,
        )

    def __bool__(self) -> bool:
        return bool(self.violations)

    def __len__(self) -> int:
        return len(self.violations)

    def __str__(self) -> str:
        return "\n".join(str(v) for v in self.violations)


T = TypeVar("T", int, float)


def bounded(*, lower: T | None = None, upper: T | None = None) -> Constraint:
    """Create a constraint which checks that the specified parameter is contained in an interval `[lower, upper]`."""

    def check(value: T) -> None:
        msg = f"{value} ∉ [{lower if lower is not None else '-∞'}, {upper if upper is not None else '∞'}]"
        assert lower is None or np.all(lower <= value), msg
        assert upper is None or np.all(value <= upper), msg

    return Constraint(check)


def greater_than(lower: T, /) -> Constraint:
    """Create a constraint which checks that the specified parameter is greater than `lower`."""

    def check(value: T) -> None:
        assert np.all(value > lower), f"{value} ∉ ({lower}, ∞)"

    return Constraint(check)


def less_than(upper: T, /) -> Constraint:
    """Create a constraint which checks that the specified parameter is less than `upper`."""

    def check(value: T) -> None:
        assert np.all(value < upper), f"{value} ∉ (-∞, {upper})"

    return Constraint(check)


def check_optimize_constraints(
    constraints: Iterable[Constraint],
    design: Any,
    config: dict[str, Any],
) -> Violations:
    """Specifically check the arguments of :method:`engibench.core.Problem.optimize()`."""
    return check_constraints(constraints, {"design": design, **config})


def check_constraints(
    constraints: Iterable[Constraint],
    parameter_args: dict[str, Any],
) -> Violations:
    """Check for violations of the given constraints for the given parameters."""
    constraints = list(constraints)
    violations = [
        violation
        for violation in (constraint.check_dict(parameter_args) for constraint in constraints)
        if violation is not None
    ]
    return Violations(violations, len(constraints))


def check_field_constraints(
    data: Any,
) -> Violations:
    """Check for violations of constraints for fields of a dataclass which declare constraints via :function:`field_constraints`."""
    assert is_dataclass(data)
    assert not isinstance(data, type)
    violations = []
    n_constraints = 0
    for f_name, constraint in field_constraints(data):
        n_constraints += 1
        violation = (
            constraint.check_value(getattr(data, f_name))
            if f_name is not None
            else constraint.check_dict(dataclasses.asdict(data))
        )
        if violation is not None:
            if f_name is not None:
                violation = Violation(violation.constraint, f"{type(data).__name__}.{f_name}: {violation.cause}")
            violations.append(violation)

    return Violations(violations, n_constraints)


def field_constraints(data: Any) -> Iterable[tuple[str | None, Constraint]]:
    """Iterate over all constraints declared on the dataclass instance `data`."""
    assert is_dataclass(data)
    # Check for annotated ClassVar:
    for f_name, f in data.__annotations__.items():
        if typing.get_origin(f) is not ClassVar:
            continue
        try:
            (annotation,) = typing.get_args(f)
        except TypeError:
            continue
        yield from ((f_name, c) for c in getattr(annotation, "__metadata__", ()) if isinstance(c, Constraint))
    for f in dataclasses.fields(data):
        # Check for typing.Annotated:
        yield from ((f.name, c) for c in getattr(f.type, "__metadata__", ()) if isinstance(c, Constraint))
    yield from ((None, c) for c in vars(type(data)).values() if isinstance(c, Constraint))


def count_field_constraints(data: Any) -> int:
    """Return the number of constraints declared on the dataclass `data`."""
    return len(list(field_constraints(data)))
