# Problem constraints

The [`Problem`](#engibench.core.Problem) class provides a [`check_constraints`](#engibench.core.Problem.check_constraints) method to validate input parameters.

So that it works, problems have to declare the constraints for their parameters in their `Conditions` class member (which itself is a [dataclass](https://docs.python.org/3/library/dataclasses.html)).

Constraints can have the following categories:

```{eval-rst}
.. autodata:: engibench.constraint.THEORY
```

```{eval-rst}
.. autodata:: engibench.constraint.IMPL
```


A constraint can have more than one category. The `|` operator can be used to combine categories.

On top of categories, constraints have a criticality level ([`Error`](engibench.constraint.Criticality.Error) by default)

```{eval-rst}
.. autoclass:: engibench.constraint.Criticality
   :members:
   :undoc-members:
```

There are 2 ways to declare a constraint:

## Simple constraint, only constraining a single parameter

Use [typing.Annotated](https://docs.python.org/3/library/typing.html#typing.Annotated),
where the annotation is one or multiple [`Constraint`](#engibench.constraint.Constraint) objects.

Predefined constraints are:

```{eval-rst}
.. automethod:: engibench.constraint.bounded
```

```{eval-rst}
.. automethod:: engibench.constraint.less_than
```

```{eval-rst}
.. automethod:: engibench.constraint.greater_than
```

Example:
```py
    @dataclass
    class Conditions:
        """Conditions."""

        volfrac: Annotated[
            float,
            bounded(lower=0.0, upper=1.0).category(THEORY),
            bounded(lower=0.1, upper=0.9).warning().category(IMPL),
        ] = 0.35
```

Here, we declare a [`THEORY`](engibench.constraint.THEORY)/[`Error`](engibench.constraint.Criticality.Error) constraint and a [`IMPL`](engibench.constraint.IMPL)/[`Warning`](engibench.constraint.Criticality.Warning) constraint for the `volfrac` parameter.

## Constraint which also may affect more than one parameter
Add a static method, decorated with the [`@constraint`](#engibench.constraint.constraint) decorator.

Example:
```py
    @dataclass
    class Config(Conditions):
        """Structured representation of conditions."""

        rmin: float = 2.0
        nelx: int = 100
        nely: int = 50

        @constraint
        @staticmethod
        def rmin_bound(rmin: float, nelx: int, nely: int) -> None:
            """Constraint for rmin ∈ (0.0, max{ nelx, nely }]."""
            assert 0 < rmin <= max(nelx, nely), f"Params.rmin: {rmin} ∉ (0, max(nelx, nely)]"
```

This declares a constraint for the 3 parameters (`rmin`, `nelx`, `nely`) with custom logic. This constraint does not have any category.
If we would want to add a category, `@constraint` could be replaced by `@constraint(category=ERROR)` for example.


## Example: Beams2D

Here, we provide a concrete example for the <project:/problems/beams2d.md> problem that illustrates when
to use certain constraints over others.

1. The case for a [`THEORY`](engibench.constraint.THEORY)/[`Error`](engibench.constraint.Criticality.Error) constraint:
  We know that for a topology optimization problem like <project:/problems/beams2d.md>, the volume fraction `volfrac` must be defined between 0 and 1 because the design
  space must be somewhere between empty void (0) and completely filled with solid material (1). Therefore, this **must** be our theoretical bound and
  we declare `bounded(lower=0.0, upper=1.0).category(THEORY)` as an argument when defining `volfrac`. Violating this constraint will throw an error,
  because it is physically impossible to have a volume fraction outside of this range.

2. The case for a [`IMPL`](engibench.constraint.IMPL)/[`Warning`](engibench.constraint.Criticality.Warning) constraint:
  Through experimentation, we have found that values of `volfrac` below 0.1 lead to solver instability and high compliance values, as there is not
  enough material available for the optimizer to provide adequate support against the applied force. We have also found that values above 0.9 do not
  provide any structurally meaningful solutions, since the design space can be almost completely filled with solid material. Therefore, we
  **recommend** users to stay within these practical implementation bounds, declaring `bounded(lower=0.1, upper=0.9).warning().category(IMPL)` as
  another argument of `volfrac`. Violating this constraint will produce a warning summarizing the above explanation, but allow the user to continue.

## Writing custom constraints

Using custom logic using the `@constraint` form is straight forward - as shown in the above example with the `rmin_bound` constraint.
But also custom logic inside `Annotated[]` is supported. Every [`Constraint`](#engibench.constraint.Constraint)
objects object inside `Annotated[]` is considered a constraint.
The simplest case is an unparametrized constraint:

```py
from engibench.constraint import Constraint

non_zero = Constraint(lambda value: value != 0)
```

Parameterizable constraints are defined as functions returning [`Constraint`](#engibench.constraint.Constraint) objects:

```py
from engibench.constraint import Constraint

def not_equal_to(value: int, /) -> Constraint:
    """Create a constraint which checks that the specified parameter is not equal to `value`."""

    def check(actual_value: int) -> None:
        assert actual_value != value, f"value == {value}"

    return Constraint(check)
```

# API

```{eval-rst}
.. autofunction:: engibench.constraint.constraint
```


```{eval-rst}
.. autoclass:: engibench.constraint.Constraint
   :members:
```
