"""Core API for Problem and other base classes."""

from __future__ import annotations

import dataclasses
from enum import auto
from enum import Enum
from typing import Any, Generic, TypeVar

from datasets import Dataset
from datasets import load_dataset
from gymnasium import spaces
import numpy as np
import numpy.typing as npt

SimulatorInputType = TypeVar("SimulatorInputType")
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


class Problem(Generic[SimulatorInputType, DesignType]):
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

    Some methods are used internally:

    - :meth:`__design_to_simulator_input` - to convert a design (output of an algorithm) to an input of the simulation.
    - :meth:`__simulator_output_to_design` - to convert a simulation output to a design (output of an algorithm).

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
        This class is parameterized with two types: `SimulatorInputType` and `DesignType`. `SimulatorInputType` is the type
        of the input to the simulator (e.g. a file containing a mesh), while `DesignType` is the type of the design that is
        optimized (e.g. a Numpy array representing the design).

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
    conditions: frozenset[tuple[str, Any]]
    """Conditions for the design problem"""
    design_space: spaces.Space[DesignType]
    """Design space (algorithm output)"""
    dataset_id: str
    """String identifier for the problem (useful to pull datasets)"""
    _dataset: Dataset | None
    """Dataset with designs and performances"""
    container_id: str | None
    """String identifier for the singularity container"""

    # This handles the RNG properly
    np_random: np.random.Generator | None = None
    __np_random_seed: int | None = None

    @property
    def dataset(self) -> Dataset:
        """Pulls the dataset if it is not already loaded."""
        if self._dataset is None:
            self._dataset = load_dataset(self.dataset_id, download_mode="force_redownload")
        return self._dataset

    def simulate(self, design: DesignType, config: dict[str, Any], **kwargs) -> npt.NDArray:
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
        self, starting_point: DesignType, config: dict[str, Any] = {}, **kwargs
    ) -> tuple[DesignType, list[OptiStep]]:
        r"""Some simulators have built-in optimization. This function optimizes the design starting from `starting_point`.

        This is optional and will probably be implemented only for some problems.

        Args:
            starting_point (DesignType): The starting point for the optimization.
            config (dict): A dictionary with configuration (e.g., boundary conditions, filenames) for the optimization.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[DesignType, list[OptiStep]]: The optimized design and the optimization history.
        """
        raise NotImplementedError

    def reset(self, seed: int | None = None, **kwargs) -> None:  # noqa: ARG002
        r"""Reset the simulator and numpy random to a given seed.

        Args:
            seed (int, optional): The seed to reset to. If None, a random seed is used.
            **kwargs: Additional keyword arguments.
        """
        if seed is not None:
            self.__np_random_seed = seed
        self.seed = seed
        self.np_random = np.random.default_rng(seed)

    def render(self, design: DesignType, open_window: bool = False, **kwargs) -> Any:
        r"""Render the design in a human-readable format.

        Args:
            design (DesignType): The design to render.
            open_window (bool): Whether to open a window to display the design.
            **kwargs: Additional keyword arguments.

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

    def __design_to_simulator_input(self, design: DesignType, **kwargs) -> SimulatorInputType:
        r"""Convert a design to a simulator input.

        Args:
            design (DesignType): The design to convert.
            **kwargs: Additional keyword arguments.

        Returns:
            SimulatorInputType: The corresponding design as a simulator input.
        """
        raise NotImplementedError

    def __simulator_output_to_design(self, simulator_output: SimulatorInputType, **kwargs) -> DesignType:
        r"""Convert a simulator input to a design.

        Args:
            simulator_output (SimulatorInputType): The input to convert.
            **kwargs: Additional keyword arguments.

        Returns:
            DesignType: The corresponding design.
        """
        raise NotImplementedError
