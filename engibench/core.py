"""Core API for Problem and other base classes."""

from __future__ import annotations

from typing import Any, ClassVar, Generic, TypeVar

from datasets import Dataset
from gymnasium import spaces
import numpy as np

SimulatorInputType = TypeVar("SimulatorInputType")
DesignType = TypeVar("DesignType")


class Problem(Generic[SimulatorInputType, DesignType]):
    r"""Main class for defining an engineering design problem.

    This class assumes there is:
    - an underlying simulator that is called to evaluate the performance of a design (see `simulate` method);
    - a dataset containing representations of designs and their performances (see `design_space`, `str_id` attributes);

    The main API methods that users should use are:
    - :meth: `simulate` - to simulate a design and return the performance given some conditions.
    - :meth: `optimize` - to optimize a design starting from a given point.
    - :meth: `representation_to_design` - to convert a representation (output of an algorithm) to a design (input of the simulation).
    - :meth: `design_to_representation` - to convert a design (input of the simulation) to a representation (output of an algorithm).

    There are some attritbutes that help understanding the problem:
    - :attr: `input_space` - the inputs of simulator.
    - :attr: `objectives` - a dictionary with the names of the objectives and their types (minimize or maximize).
    - :attr: `design_space` - the space of designs (outputs of algorithms).
    - :attr: `str_id` - a string identifier for the problem -- useful to pull datasets and singularity containers.
    """

    # Must be defined in subclasses
    input_space: SimulatorInputType  # Simulator input (internal)
    possible_objectives: ClassVar[dict[str, str]]  # Objective names and types (minimize or maximize)
    design_space: spaces.Space[DesignType]  # Design space (algorithm output)
    str_id: str  # String identifier for the problem (useful to pull datasets and singularity containers)
    dataset: Dataset  # Dataset with designs and performances

    # This handles the RNG properly
    _np_random: np.random.Generator | None = None
    _np_random_seed: int | None = None

    def simulate(self, design: DesignType, conditions: dict[str, Any], **kwargs) -> dict[str, float]:
        r"""Launch a simulation on the given design and return the performance.

        Args:
            design (DesignType): The design to simulate.
            conditions (dict): A dictionary with additional conditions that might be needed for the simulation.

        Returns:
            dict: The performance of the design - each entry of the dict corresponds to a named objective value.
        """
        raise NotImplementedError

    def optimize(
        self, starting_point: DesignType, conditions: dict[str, Any], **kwargs
    ) -> tuple[DesignType, dict[str, float]]:
        r"""Some simulators have built-in optimization. This function optimizes the design starting from `starting_point`.

        This is optional and will probably be implemented only for some problems.

        Args:
            starting_point (DesignType): The starting point for the optimization.
            conditions (dict): A dictionary with additional conditions that might be needed for the optimization.

        Returns:
            Tuple[DesignType, dict]: The optimized design and its performance.
        """
        raise NotImplementedError

    def reset(self, seed: int | None = None) -> None:
        r"""Reset the simulator and numpy random to a given seed.

        Args:
            seed (int, optional): The seed to reset to. If None, a random seed is used.
        """
        if seed is not None:
            self._np_random_seed = seed
        self._np_random = np.random.default_rng(seed)

    def design_to_simulator_input(self, design: DesignType, **kwargs) -> SimulatorInputType:
        r"""Convert a design to a simulator input.

        Args:
            design (DesignType): The design to convert.

        Returns:
            SimulatorInputType: The corresponding design as a simulator input.
        """
        raise NotImplementedError

    def simulator_input_to_design(self, simulator_input: SimulatorInputType, **kwargs) -> DesignType:
        r"""Convert a simulator input to a design.

        Args:
            simulator_input (SimulatorInputType): The input to convert.

        Returns:
            DesignType: The corresponding design.
        """
        raise NotImplementedError
