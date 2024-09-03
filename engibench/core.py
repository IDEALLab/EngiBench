"""Core API for Problem and other base classes."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from gymnasium import spaces
import numpy as np

DesignType = TypeVar("DesignType")
RepresentationType = TypeVar("RepresentationType")


class Problem(Generic[DesignType, RepresentationType]):
    r"""Main class for defining an engineering design problem.

    This class assumes there is:
    - an underlying simulator that is called to evaluate the performance of a design (see `simulate` method);
    - a dataset containing representations of designs and their performances (see `representation_space` attribute);

    The main API methods that users should use are:
    - :meth: `simulate` - to simulate a design and return the performance given some conditions.
    - :meth: `representation_to_design` - to convert a representation (output of an algorithm) to a design (input of the simulation).
    - :meth: `design_to_representation` - to convert a design (input of the simulation) to a representation (output of an algorithm).

    There are some attritbutes that help understanding the problem:
    - :attr: `design_space` - the space of designs (inputs of simulator).
    - :attr: `representation_space` - the space of representations (outputs of algorithms).
    - :attr: `objectives` - a dictionary with the names of the objectives and their types (minimize or maximize).
    - :attr: `str_id` - a string identifier for the problem -- useful to pull datasets and singularity containers.
    """

    # Must be defined in subclasses
    design_space: spaces.Space[DesignType]  # Simulator input space
    objectives: dict[str, str]  # Objective names and types (minimize or maximize)
    representation_space: spaces.Space[RepresentationType]  # Algorithm output space
    str_id: str  # String identifier for the problem (useful to pull datasets and singularity containers)

    # This handles the RNG properly
    _np_random: np.random.Generator | None = None
    _np_random_seed: int | None = None

    def simulate(self, design: DesignType, conditions: dict[str, Any]) -> dict[str, float]:
        r"""Launch a simulation on the given design and return the performance.

        Args:
            design (DesignType): The design to simulate.
            conditions (dict): A dictionary with additional conditions that might be needed for the simulation.

        Returns:
            dict: The performance of the design - each entry of the dict corresponds to a named objective value.
        """
        raise NotImplementedError

    def optimize(self, starting_point: DesignType) -> tuple[DesignType, dict[str, float]]:
        r"""Some simulators have built-in optimization. This function optimizes the design starting from `starting_point`.

        (This is optional and will probably be implemented only for some problems.)

        Args:
            starting_point (DesignType): The starting point for the optimization.

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

    def representation_to_design(self, representation: RepresentationType) -> DesignType:
        r"""Convert a representation to a design.

        Args:
            representation (RepresentationType): The representation to convert.

        Returns:
            DesignType: The corresponding design.
        """
        raise NotImplementedError

    def design_to_representation(self, design: DesignType) -> RepresentationType:
        r"""Convert a design to a representation.

        Args:
            design (DesignType): The design to convert.

        Returns:
            RepresentationType: The corresponding representation.
        """
        raise NotImplementedError
