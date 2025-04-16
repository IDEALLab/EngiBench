"""Dataset Generation for Photonics2D Problem via SLURM.

This script generates a dataset for the Photonics2D problem using the SLURM API
"""

from itertools import product
import os
import pickle

import numpy as np

from engibench.problems.photonics2d import Photonics2D
from engibench.utils import slurm

lambda1 = np.linspace(start=0.5, stop=1.5, num=3)
lambda2 = np.linspace(start=0.5, stop=1.5, num=3)
blur_radius = range(0, 2)
num_elems_x = 120
num_elems_y = 120

# Cartesian product
combinations = list(product(lambda1, lambda2, blur_radius))
print(combinations)


def config_factory(lambda1: float, lambda2: float, blur_radius: int) -> dict:
    """Factory function to create configuration dictionaries."""
    return {
        "lambda1": lambda1,
        "lambda2": lambda2,
        "blur_radius": blur_radius,
        "num_elems_x": num_elems_x,
        "num_elems_y": num_elems_y,
    }


# Generate configurations
configs = [config_factory(l1, l2, br) for l1, l2, br in combinations]
print(configs)

# Make slurm Args
parameter_space = [slurm.Args(problem_args=config) for config in configs]

slurm.submit(
    problem=Photonics2D,
    parameter_space=parameter_space,
)

with open("results.pkl", "rb") as stream:
    results = pickle.load(stream)
os.remove("results.pkl")
