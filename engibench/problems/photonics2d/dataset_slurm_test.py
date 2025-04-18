"""Dataset Generation for Photonics2D Problem via SLURM.

This script generates a dataset for the Photonics2D problem using the SLURM API
"""

from itertools import product
import os
import pickle
import time

import numpy as np

from engibench.problems.photonics2d import Photonics2D
from engibench.utils import slurm

# ============== Problem-specific elements ===================
# The following elements are specific to the problem and should be modified accordingly
target_problem = Photonics2D
# Specify the parameters you want to sweep over for optimization
rng = np.random.default_rng()
lambda1 = rng.uniform(low=0.5, high=1.25, size=2)
lambda2 = rng.uniform(low=0.75, high=1.5, size=2)
blur_radius = range(0, 2)
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
    problem = target_problem(config=config)
    start_design, _ = problem.random_design(noise=0.001)  # Randomized design with noise
    return {"design": start_design}


# Generate configurations
configs = [config_factory(l1, l2, br) for l1, l2, br in combinations]

# Any optimization-wide configurations can be set here
optimize_config = {"num_optimization_steps": 200}

# Timing information for `optimize` and `simulate` functions
# If you can estimate the time it takes to run `optimize` and `simulate`,
# you can set the runtimes here, and this will help with job scheduling.
# Try to be conservative with the time estimates, so SLURM doesn't kill it prematurely.
# The format is "HH:MM:SS"
runtime_optimize = "00:10:00"  # ~10 minutes for optimization
runtime_simulate = "00:01:00"  # ~1 minute for simulation
runtime_render = "00:02:00"  # ~2 minutes for rendering

# ============== End of problem-specific elements ===================


# ============== Problem-agnostic elements ===================
# The following elements are agnostic to the problem and can be reused across problems

# Make slurm Args
parameter_space = [
    slurm.Args(problem_args=config, design_args=design_factory(config), optimize_args=optimize_config) for config in configs
]
print(f"Generating parameter space via SLURM with {len(parameter_space)} configurations.")

# --------- Testing `optimize` via SLURM ---------
# First let's check if we can run `optimize``
start_time = time.time()
slurm.submit(
    job_type="optimize",
    problem=target_problem,
    parameter_space=parameter_space,
    config=slurm.SlurmConfig(log_dir="./opt_logs/", runtime=runtime_optimize),
)
end_time = time.time()
print(f"Elapsed time for `optimize`: {end_time - start_time:.2f} seconds")

# If interactive, load the results from file
with open("results.pkl", "rb") as stream:
    opt_results = pickle.load(stream)

# Since our slurm script currently save the results of every slurm job with the
# same filename (results.pkl), we need to rename opt results to avoid overwriting
# when we call simulate
os.rename("results.pkl", "results_opt.pkl")


# ---------- Testing `simulate` via SLURM ---------
# Now let's test slurm simulate
# For this, we will pull all of the final designs from the results, and then
# simulate them with the same parameters as the original problem
problem_args = []
final_designs = []
for _i, result in enumerate(opt_results):
    problem_args = result["problem_args"]
    final_design, obj_trajectory = result["results"]
    final_designs.append(final_design)
    problem_args.append(problem_args)

# Now assemble the slurm Args for the simulation
slurm_simulate_args = [
    slurm.Args(problem_args=problem_args[i], design_args={"design": final_designs[i]}) for i in range(len(problem_args))
]
start_time = time.time()
slurm.submit(
    job_type="simulate",
    problem=target_problem,
    parameter_space=slurm_simulate_args,
    config=slurm.SlurmConfig(log_dir="./sim_logs/", runtime=runtime_simulate),  # Shorter, since sim is faster
)
end_time = time.time()
print(f"Elapsed time for `simulate`: {end_time - start_time:.2f} seconds")

with open("results.pkl", "rb") as stream:
    sim_results = pickle.load(stream)
# Since our slurm script currently save the results of every slurm job with the
# same filename (results.pkl), we need to rename opt results to avoid overwriting
# when we call simulate
os.rename("results.pkl", "results_sim.pkl")


# ---------- Testing `render` via SLURM ---------
# Now let's test slurm render
# We can reuse slurm_simulate_args, since the render function takes similar args
start_time = time.time()
slurm.submit(
    job_type="render",
    problem=target_problem,
    parameter_space=slurm_simulate_args,
    config=slurm.SlurmConfig(log_dir="./sim_logs/", runtime=runtime_render),  # Shorter, since sim is faster
)
end_time = time.time()
print(f"Elapsed time for `render`: {end_time - start_time:.2f} seconds")

with open("results.pkl", "rb") as stream:
    render_results = pickle.load(stream)
# Since our slurm script currently save the results of every slurm job with the
# same filename (results.pkl), we need to rename opt results to avoid overwriting
# when we call simulate
os.rename("results.pkl", "results_render.pkl")
