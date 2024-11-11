"""Core API for Problem and other base classes."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from datasets import Dataset
from gymnasium import spaces
import numpy as np

SimulatorInputType = TypeVar("SimulatorInputType")
DesignType = TypeVar("DesignType")


class Problem(Generic[SimulatorInputType, DesignType]):
    r"""Main class for defining an engineering design problem.

    This class assumes there is:
    - an underlying simulator that is called to evaluate the performance of a design (see `simulate` method);
    - a dataset containing representations of designs and their performances (see `design_space`, `dataset_id` attributes);

    The main API methods that users should use are:
    - :meth: `simulate` - to simulate a design and return the performance given some conditions.
    - :meth: `optimize` - to optimize a design starting from a given point, e.g., using adjoint solver included inside the simulator.
    - :meth: `design_to_simulator_input` - to convert a design (output of an algorithm) to an input of the simulation.
    - :meth: `simulator_input_to_design` - to convert a simulation input to a design (output of an algorithm).

    There are some attributes that help understanding the problem:
    - :attr: `possible_objectives` - a dictionary with the names of the objectives and their types (minimize or maximize).
    - :attr: `design_space` - the space of designs (outputs of algorithms).
    - :attr: `dataset_id` - a string identifier for the problem -- useful to pull datasets.
    - :attr: `dataset` - the dataset with designs and performances.
    - :attr: `container_id` - a string identifier for the singularity container.
    - :attr: `input_space` - the inputs of simulator.
    """

    # Must be defined in subclasses
    possible_objectives: frozenset[tuple[str, str]]  # Objective names and types (minimize or maximize)
    design_space: spaces.Space[DesignType]  # Design space (algorithm output)
    dataset_id: str  # String identifier for the problem (useful to pull datasets)
    dataset: Dataset  # Dataset with designs and performances
    container_id: str  # String identifier for the singularity container
    input_space: SimulatorInputType  # Simulator input (internal)

    # This handles the RNG properly
    _np_random: np.random.Generator | None = None
    _np_random_seed: int | None = None

    def simulate(self, design: DesignType, config: dict[str, Any], **kwargs) -> dict[str, float]:
        r"""Launch a simulation on the given design and return the performance.

        Args:
            design (DesignType): The design to simulate.
            config (dict): A dictionary with configuration (e.g., boundary conditions, filenames) for the optimization.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: The performance of the design - each entry of the dict corresponds to a named objective value.
        """
        raise NotImplementedError

    def optimize(self, starting_point: DesignType, config: dict[str, Any], **kwargs) -> tuple[DesignType, dict[str, float]]:
        r"""Some simulators have built-in optimization. This function optimizes the design starting from `starting_point`.

        This is optional and will probably be implemented only for some problems.

        Args:
            starting_point (DesignType): The starting point for the optimization.
            config (dict): A dictionary with configuration (e.g., boundary conditions, filenames) for the optimization.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[DesignType, dict]: The optimized design and its performance.
        """
        raise NotImplementedError

    def reset(self, seed: int | None = None, **kwargs) -> None:  # noqa: ARG002
        r"""Reset the simulator and numpy random to a given seed.

        Args:
            seed (int, optional): The seed to reset to. If None, a random seed is used.
            **kwargs: Additional keyword arguments.
        """
        if seed is not None:
            self._np_random_seed = seed
        self.seed = seed
        self._np_random = np.random.default_rng(seed)

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
