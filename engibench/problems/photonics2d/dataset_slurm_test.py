"""Dataset Generation for Photonics2D Problem via SLURM.

This script generates a dataset for the Photonics2D problem using the SLURM API
"""

from itertools import product
import pickle

import numpy as np

from engibench.problems.photonics2d import Photonics2D
from engibench.utils import slurm

rng = np.random.default_rng()
lambda1 = rng.uniform(low=0.5, high=1.25, size=10)
lambda2 = rng.uniform(low=0.75, high=1.5, size=10)
blur_radius = range(0, 6)
num_elems_x = 120
num_elems_y = 120

# Generate all combinations of parameters to run
combinations = list(product(lambda1, lambda2, blur_radius))


# Generate full problem configurations, including static parameters
def config_factory(lambda1: float, lambda2: float, blur_radius: int) -> dict:
    """Factory function to create configuration dictionaries."""
    return {
        "lambda1": lambda1,
        "lambda2": lambda2,
        "blur_radius": blur_radius,
        "num_elems_x": num_elems_x,
        "num_elems_y": num_elems_y,
    }


# Generate starting design for each problem based on each configuration
def design_factory(config: dict) -> dict:
    """Produces starting design for the problem."""
    problem = Photonics2D(**config)
    start_design, _ = problem.random_design(noise=0.001)  # Randomized design with noise
    return {"design": start_design}


# Generate configurations
configs = [config_factory(l1, l2, br) for l1, l2, br in combinations]

# Make slurm Args
parameter_space = [slurm.Args(problem_args=config, design_args=design_factory(config)) for config in configs]
print(f"Generating parameter space via SLURM with {len(parameter_space)} configurations.")

slurm.submit(
    problem=Photonics2D,
    parameter_space=parameter_space,
    config=slurm.SlurmConfig(log_dir="./logs/", runtime="00:05:00"),
)

# If interactive, load the results from file
with open("results.pkl", "rb") as stream:
    results = pickle.load(stream)
