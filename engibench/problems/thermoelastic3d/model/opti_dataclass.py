"""This module contains the dataclass for updating an opti-step in the thermoelastic2d problem."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OptiStepUpdate:
    """Dataclass encapsulating all input parameters for an OptiStep update."""

    obj_values: np.ndarray
    """The objectives values of the current iteration"""
    iterr: int
    """The current iteration number"""
    x_curr: np.ndarray
    """The current design variables"""
    x_sensitivities: np.ndarray
    """The sensitivities of the design variables"""
    x_update: np.ndarray
    """The gradient update step taken by the optimizer"""
    extra_iter: bool
    """Whether this update is for the final iteration or not"""
