"""Core API for Problem and other base classes."""

from __future__ import annotations

import dataclasses
from enum import auto
from enum import Enum
from typing import Any, Generic, TYPE_CHECKING, TypeVar

from datasets import Dataset
from datasets import load_dataset
import numpy as np
import numpy.typing as npt

from engibench import constraint

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gymnasium import spaces

DesignType = TypeVar("DesignType")


@dataclasses.dataclass
class OptiStep:
    """Optimization step."""

    obj_values: npt.NDArray
    step: int


class ObjectiveDirection(Enum):
    """Direction of the objective function."""

    MINIMIZE = auto()
    MAXIMIZE = auto()


class Problem(Generic[DesignType]):
    r"""Main class for defining an engineering design problem.

    This class assumes there is:

    - an underlying simulator that is called to evaluate the performance of a design (see `simulate` method);
    - a dataset containing representations of designs and their performances (see `design_space`, `dataset_id` attributes);

    The main API methods that users should use are:

    - :meth:`simulate` - to simulate a design and return the performance given some conditions.
    - :meth:`optimize` - to optimize a design starting from a given point, e.g., using adjoint solver included inside the simulator.
    - :meth:`reset` - to reset the simulator and numpy random to a given seed.
    - :meth:`render` - to render a design in a human-readable format.
    - :meth:`random_design` - to generate a valid random design.

    There are some attributes that help understanding the problem:

    - :attr:`objectives` - a dictionary with the names of the objectives and their types (minimize or maximize).
    - :attr:`conditions` - the conditions for the design problem.
    - :attr:`design_space` - the space of designs (outputs of algorithms).
    - :attr:`dataset_id` - a string identifier for the problem -- useful to pull datasets.
    - :attr:`dataset` - the dataset with designs and performances.
    - :attr:`container_id` - a string identifier for the singularity container.

    Having all these defined in the code allows to easily extract the columns we want from the dataset to train ML models.

    Note:
        This class is generic and should be subclassed to define the specific problem.

    Note:
        This class is parameterized with `DesignType` is the type of the design that is optimized (e.g. a Numpy array representing the design).

    Note:
        Some simulators also ask for simulator related configurations. These configurations are generally defined in the
        problem implementation, do not appear in the `problem.conditions`, but sometimes appear in the dataset (for
        advanced usage). You can override them by using the `config` argument in the `simulate` or `optimize` method.
    """

    # Must be defined in subclasses
    version: int
    """Version of the problem"""
    objectives: tuple[tuple[str, ObjectiveDirection], ...]
    """Objective names and types (minimize or maximize)"""
    conditions: tuple[tuple[str, Any], ...]
    """Conditions for the design problem"""
    design_space: spaces.Space[DesignType]
    """Design space (algorithm output)"""
    dataset_id: str
    """String identifier for the problem (useful to pull datasets)"""
    design_constraints: tuple[constraint.Constraint, ...] = ()
    """Additional constraints for designs"""
    _dataset: Dataset | None = None
    """Dataset with designs and performances"""
    container_id: str | None
    """String identifier for the singularity container"""
    Config: type | None = None
    """Dataclass declaring types, defaults (optional) and constraints"""

    # This handles the RNG properly
    np_random: np.random.Generator

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the problem.

        Args:
            **kwargs: Keyword arguments.
        """
        self.reset(**kwargs)

    @property
    def dataset(self) -> Dataset:
        """Pulls the dataset if it is not already loaded."""
        if self._dataset is None:
            self._dataset = load_dataset(self.dataset_id)
        return self._dataset

    @property
    def conditions_dict(self) -> dict[str, Any]:
        """Returns the conditions as a dictionary."""
        return dict(self.conditions)

    @property
    def conditions_keys(self) -> list[str]:
        """Returns the condition names as a list."""
        return [name for name, _ in self.conditions]

    @property
    def objectives_keys(self) -> list[str]:
        """Returns the objective names as a list."""
        return [name for name, _ in self.objectives]

    def simulate(self, design: DesignType, config: dict[str, Any] | None = None) -> npt.NDArray:
        r"""Launch a simulation on the given design and return the performance.

        Args:
            design (DesignType): The design to simulate.
            config (dict): A dictionary with configuration (e.g., boundary conditions, filenames) for the optimization.
            **kwargs: Additional keyword arguments.

        Returns:
            np.array: The performance of the design -- each entry corresponds to an objective value.
        """
        raise NotImplementedError

    def optimize(
        self, starting_point: DesignType, config: dict[str, Any] | None = None
    ) -> tuple[DesignType, Sequence[OptiStep]]:
        r"""Some simulators have built-in optimization. This function optimizes the design starting from `starting_point`.

        This is optional and will probably be implemented only for some problems.

        Args:
            starting_point (DesignType): The starting point for the optimization.
            config (dict): A dictionary with configuration (e.g., boundary conditions, filenames) for the optimization.

        Returns:
            Tuple[DesignType, list[OptiStep]]: The optimized design and the optimization history.
        """
        raise NotImplementedError

    def reset(self, seed: int | None = None) -> None:
        r"""Reset the simulator and numpy random to a given seed.

        Args:
            seed (int, optional): The seed to reset to. If None, a random seed is used.
        """
        self.seed = seed
        self.np_random = np.random.default_rng(seed)

    def render(self, design: DesignType, *, open_window: bool = False) -> Any:
        r"""Render the design in a human-readable format.

        Args:
            design (DesignType): The design to render.
            open_window (bool): Whether to open a window to display the design.

        Returns:
            Any: The rendered design.
        """
        raise NotImplementedError

    def random_design(self) -> tuple[DesignType, int]:
        r"""Generate a random design.

        Returns:
            DesignType: The random design.
            idx: The index of the design in the dataset.
        """
        raise NotImplementedError

    def check_constraints(self, design: DesignType, config: dict[str, Any]) -> constraint.Violations:
        """Check if config and design violate any constraints declared in `Config` and `design_space`.

        Return a :class:`constraint.Violations` object containing all violations.
        """
        if self.Config is not None:
            checked_config = self.Config(**config)
            violations = constraint.check_field_constraints(checked_config)
        else:
            violations = constraint.Violations([], 0)

        @constraint.constraint
        def design_constraint(design: DesignType) -> None:
            assert self.design_space.contains(design), "design ∉ design_space"

        violations.n_constraints += 1 + len(self.design_constraints)
        for c in (design_constraint, *self.design_constraints):
            design_violation = c.check_value(design)
            if design_violation is not None:
                violations.violations.append(design_violation)

        return violations
