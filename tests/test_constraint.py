from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
import pytest

from engibench import constraint
from engibench.constraint import bounded
from engibench.constraint import greater_than
from engibench.constraint import IMPL
from engibench.constraint import THEORY
from engibench.constraint import Var

RR_MAX = 9.0


def test_check_constraints_detects_violations() -> None:
    @constraint.constraint
    def radius(x: float, y: float) -> None:
        assert x**2 + y**2 < RR_MAX, "Radius >= 3"

    z_positive = Var("z").check(greater_than(0.0))
    args = {"x": 1, "y": 4, "z": -1.0}
    violations = constraint.check_constraints([radius, z_positive], args)
    assert violations.violations == [
        constraint.Violation(radius, "Radius >= 3\nassert ((1 ** 2) + (4 ** 2)) < 9.0"),
        constraint.Violation(z_positive, "-1.0 ∉ (0.0, ∞)"),
    ]


def test_check_constraints_complains_about_missing_args() -> None:
    @constraint.constraint
    def radius(x: float, y: float) -> None:
        assert x**2 + y**2 < RR_MAX, "Radius >= 3"

    args = {"x": 1}
    with pytest.raises(TypeError):
        constraint.check_constraints([radius], args)


def test_check_constraints_passes_for_no_violations() -> None:
    @constraint.constraint
    def radius(x: float, y: float) -> None:
        assert x**2 + y**2 < RR_MAX, "Radius >= 3"

    args = {"x": 1, "y": 1}
    violations = constraint.check_constraints([radius, Var("y").check(greater_than(0.0))], args)
    assert not violations


def test_check_constraints_with_warnings() -> None:
    @constraint.constraint(criticality=constraint.Criticality.Warning)
    def radius(x: float, y: float) -> None:
        assert x**2 + y**2 < RR_MAX, "R >= 3"

    args = {"x": 5, "y": 1}
    violations = constraint.check_constraints([radius], args)
    assert radius.criticality == constraint.Criticality.Warning
    assert violations.violations == [constraint.Violation(radius, "R >= 3\nassert ((5 ** 2) + (1 ** 2)) < 9.0")]


def test_violation_error_is_filterable_by_criticality() -> None:
    z_positive = Var("z").check(greater_than(0.0))
    y_positive = Var("y").check(greater_than(0.0)).warning()
    z_violation = constraint.Violation(z_positive, "z < 0")
    violations = constraint.Violations(
        [
            z_violation,
            constraint.Violation(y_positive, "y < 0"),
        ],
        2,
    )
    errors = violations.by_criticality(constraint.Criticality.Error)
    assert errors.violations == [z_violation]


def test_violation_error_is_filterable_by_category() -> None:
    z_positive = Var("z").check(greater_than(0.0)).category(constraint.Category.Theory)
    y_positive = Var("y").check(greater_than(0.0)).warning().category(constraint.Category.Implementation)
    z_violation = constraint.Violation(z_positive, "z < 0")
    violations = constraint.Violations(
        [
            z_violation,
            constraint.Violation(y_positive, "y < 0"),
        ],
        2,
    )
    theory_violations = violations.by_category(constraint.Category.Theory)
    assert theory_violations.violations == [z_violation]


def test_violation_error_is_filterable_by_category_and_criticality() -> None:
    z_positive = Var("z").check(greater_than(0.0)).category(constraint.Category.Theory)
    y_positive = Var("y").check(greater_than(0.0)).warning().category(constraint.Category.Implementation)
    x_positive = Var("x").check(greater_than(0.0)).warning().category(constraint.Category.Theory)
    z_violation = constraint.Violation(z_positive, "z < 0")
    violations = constraint.Violations(
        [
            z_violation,
            constraint.Violation(y_positive, "y < 0"),
            constraint.Violation(x_positive, "x < 0"),
        ],
        3,
    )
    errors = violations.by_category(constraint.Category.Theory).by_criticality(constraint.Criticality.Error)
    assert errors.violations == [z_violation]


def test_is_bounded() -> None:
    c = Var("z").check(bounded(lower=-1.0, upper=1.0))
    violations = [constraint.check_constraints([c], {"z": z}) for z in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]]

    def violation(z: float) -> constraint.Violation:
        return constraint.Violation(c, f"{z} ∉ [-1.0, 1.0]")

    assert violations[0].violations == [violation(-2.0)]
    assert violations[1].violations == [violation(-1.5)]
    assert all((not v.violations) for v in violations[2:7])
    assert violations[7].violations == [violation(1.5)]
    assert violations[8].violations == [violation(2.0)]


def test_is_bounded_works_for_arrays() -> None:
    c = Var("z").check(bounded(lower=-1.0, upper=1.0))
    violations = [constraint.check_constraints([c], {"z": z}) for z in [np.array([0.0, 0.0]), np.array([2.0, 0.0])]]

    def violation(z: NDArray[np.float64]) -> constraint.Violation:
        return constraint.Violation(c, f"{z} ∉ [-1.0, 1.0]")

    assert not violations[0].violations
    assert violations[1].violations == [violation(np.array([2.0, 0.0]))]


def test_field_constraints() -> None:
    @dataclass
    class Params:
        nelx: Annotated[int, bounded(lower=1).category(THEORY), bounded(lower=10, upper=1000).warning().category(IMPL)] = 0
        nely: int = 0
        volfrac: Annotated[float, bounded(lower=0.1, upper=0.9).warning().category(IMPL)] = 0.0
        penal: float = 0.0
        rmin: float = 0.0

        @constraint.constraint
        @staticmethod
        def rmin_bound(rmin: float, nelx: int, nely: int) -> None:
            assert 0 < rmin <= max(nelx, nely), f"Params.rmin: {rmin} ∉ (0, max(nelx, nely)]"

    params = Params()
    violations = [
        (v.cause, v.constraint.criticality, v.constraint.categories)
        for v in constraint.check_field_constraints(params).violations
    ]
    assert violations == [
        ("Params.nelx: 0 ∉ [1, ∞]", constraint.Criticality.Error, THEORY),
        ("Params.nelx: 0 ∉ [10, 1000]", constraint.Criticality.Warning, IMPL),
        ("Params.volfrac: 0.0 ∉ [0.1, 0.9]", constraint.Criticality.Warning, IMPL),
        (
            "Params.rmin: 0.0 ∉ (0, max(nelx, nely)]\nassert 0 < 0.0",
            constraint.Criticality.Error,
            constraint.UNCATEGORIZED,
        ),
    ]


def test_field_constraints_with_class_vars() -> None:
    @dataclass
    class Params:
        static_x: ClassVar[Annotated[float, bounded(lower=1.0).category(THEORY)]] = 0.0
        x: Annotated[float, bounded(lower=1.0).category(IMPL)] = 0.0

    params = Params()

    violations = [
        (v.cause, v.constraint.criticality, v.constraint.categories)
        for v in constraint.check_field_constraints(params).violations
    ]
    assert violations == [
        ("Params.static_x: 0.0 ∉ [1.0, ∞]", constraint.Criticality.Error, THEORY),
        ("Params.x: 0.0 ∉ [1.0, ∞]", constraint.Criticality.Error, IMPL),
    ]
