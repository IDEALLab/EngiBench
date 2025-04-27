"""Dataset Generation for Photonics2D Problem via SLURM.

This script generates a dataset for the Photonics2D problem using the SLURM API
"""

from argparse import ArgumentParser
from itertools import product
import os
import pickle
import shutil
import time
from typing import Any, Callable

import numpy as np

from engibench.problems.photonics2d import Photonics2D
from engibench.utils import slurm

# ============== Problem-agnostic elements ===================
# The following elements are agnostic to the problem and can be reused across problems


def optimize_slurm(target_problem: Any, configs: list[dict], design_factory: Callable, optimize_config: dict) -> list[dict]:
    """Function to optimize designs via SLURM."""
    # Make slurm Args
    parameter_space = [
        slurm.Args(problem_args=config, design_args=design_factory(config), optimize_args=optimize_config)
        for config in configs
    ]
    print(f"Generating parameter space via SLURM with {len(parameter_space)} configurations.")

    # --------- Testing `optimize` via SLURM ---------
    # First let's check if we can run `optimize``
    print("Starting `optimize` via SLURM...")
    start_time = time.time()
    slurm.submit(
        job_type="optimize",
        problem=target_problem,
        parameter_space=parameter_space,
        config=slurm.SlurmConfig(log_dir="./opt_logs/", runtime=runtime_optimize),
    )
    end_time = time.time()
    print(f"Elapsed time for `optimize`: {end_time - start_time:.2f} seconds")

    # Since our slurm script currently save the results of every slurm job with the
    # same filename (results.pkl), we need to rename opt results to avoid overwriting
    # when we call simulate
    os.rename("results.pkl", "results_opt.pkl")
    with open("results_opt.pkl", "rb") as stream:
        return pickle.load(stream)


# At this point, we have the results of the optimization in `opt_results`.
# If we want to do further simulation or rendering, we can use the scripts below.
# To mirror conditions of each optimized design, we can pull out the SLURM args for each
# and then simulate them with the same parameters as the original problem
def make_sim_args_from_opt_results(opt_results: list[dict]) -> list[slurm.Args]:
    """Function to create simulation args from optimization results."""
    problem_args = []
    final_designs = []
    for _i, result in enumerate(opt_results):
        final_design, obj_trajectory = result["results"]
        final_designs.append(final_design)
        problem_args.append(result["problem_args"])
    # Now assemble the slurm Args for the simulation or rendering
    return [
        slurm.Args(
            problem_args=problem_args[i],
            design_args={"design": final_designs[i]},
            simulate_args={},  # No specific changes here, but you could if wanted.
        )
        for i in range(len(problem_args))
    ]


# If we just want to render the designs locally, we can do that here. If rendering takes a while
# then we might want to run it via SLURM, which is later in this script. Below is just for
# local execution of render.
def render_local(target_problem: Any, opt_results: list[dict], fig_path: str) -> None:
    """Function to render designs locally, without SLURM."""
    # Check if `figs` directory exists, if not create it
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    for _i, result in enumerate(opt_results):
        final_design, obj_trajectory = result["results"]
        problem = target_problem(result["problem_args"])
        fig = problem.render(design=final_design, config=result["simulate_args"])
        fig.savefig(fig_path + f"/final_design_{_i}.png")


# The below may be overkill, in the sense that if simulate and render are cheap functions,
# then it may not make sense to run them via SLURM. However, the below is just a test of the
# SLURM API in the case where the simulation or rendering is expensive.

# ---------- Testing `simulate` via SLURM ---------
# Now let's test slurm simulate


def simulate_slurm(target_problem: Any, slurm_simulate_args: list[slurm.Args], runtime_simulate: str) -> None:
    """Function to simulate designs via SLURM."""
    print("Starting `simulate` via SLURM...")
    start_time = time.time()
    slurm.submit(
        job_type="simulate",
        problem=target_problem,
        parameter_space=slurm_simulate_args,
        config=slurm.SlurmConfig(log_dir="./sim_logs/", runtime=runtime_simulate),  # Shorter, since sim is faster
    )
    end_time = time.time()
    print(f"Elapsed time for `simulate`: {end_time - start_time:.2f} seconds")

    # Since our slurm script currently save the results of every slurm job with the
    # same filename (results.pkl), we need to rename opt results to avoid overwriting
    # when we call simulate
    os.rename("results.pkl", "results_sim.pkl")
    with open("results_sim.pkl", "rb") as stream:
        return pickle.load(stream)


# ---------- Testing `render` via SLURM ---------
# Now let's test slurm render
# We can reuse slurm_simulate_args, since the render function takes similar args
def render_slurm(
    target_problem: Any, slurm_simulate_args: list[slurm.Args], runtime_render: str, fig_path: str = "./figs/"
) -> None:
    """Function to render designs via SLURM."""
    print("Starting `render` via SLURM...")
    start_time = time.time()
    slurm.submit(
        job_type="render",
        problem=target_problem,
        parameter_space=slurm_simulate_args,
        config=slurm.SlurmConfig(
            log_dir="./sim_logs/", runtime=runtime_render, mem_per_cpu="4G"
        ),  # Shorter, since sim is faster
    )
    end_time = time.time()
    print(f"Elapsed time for `render`: {end_time - start_time:.2f} seconds")

    # Since our slurm script currently save the results of every slurm job with the
    # same filename (results.pkl), we need to rename opt results to avoid overwriting
    # when we call simulate
    os.rename("results.pkl", "results_render.pkl")
    with open("results_render.pkl", "rb") as stream:
        render_results = pickle.load(stream)

    # Check if `figs` directory exists, if not create it
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    for i, result in enumerate(render_results):
        result["results"].savefig(fig_path + f"final_design_{i}.png")
    # Zip the figures for easy download
    shutil.make_archive("figures_all", "zip", fig_path)


if __name__ == "__main__":
    """Dataset Generation, Optimization, Simulation, and Rendering for Photonics2D Problem via SLURM.

    This script generates a dataset for the Photonics2D problem using the SLURM API, though it could
    be generalized to other problems as well. It includes functions for optimization, simulation,
    and rendering of designs.

    Command Line Arguments:
    -r, --render: Should we render the optimized designs?
    --figure_path: Where should we place the figures?
    -s, --simulate: Should we simulate the optimized designs?

    """
    # Fetch command line arguments for render and simulate to know whether to run those functions
    parser = ArgumentParser()
    parser.add_argument(
        "-r",
        "--render",
        action="store_true",
        dest="render_flag",
        default=False,
        help="Should we render the optimized designs?",
    )
    parser.add_argument("--figure_path", dest="fig_path", default="./figs", help="Where should we place the figures?")
    parser.add_argument(
        "-s",
        "--simulate",
        action="store_true",
        dest="simulate_flag",
        default=False,
        help="Should we simulate the optimized designs?",
    )
    args = parser.parse_args()

    # ============== Problem-specific elements ===================
    # The following elements are specific to the problem and should be modified accordingly
    target_problem = Photonics2D
    # Specify the parameters you want to sweep over for optimization
    rng = np.random.default_rng()
    lambda1 = rng.uniform(low=0.5, high=1.25, size=20)
    lambda2 = rng.uniform(low=0.75, high=1.5, size=20)
    blur_radius = range(5)
    num_elems_x = 120
    num_elems_y = 120

    # Generate all combinations of parameters to run
    combinations = list(product(lambda1, lambda2, blur_radius))

    # Generate full problem configurations, including static parameters
    # Note that currently this doesn't allow you to change the resolution of the problem
    # So in the re-write of the SLURM API we will need to add that functionality.
    def config_factory(lambda1: float, lambda2: float, blur_radius: int) -> dict:
        """Factory function to create configuration dictionaries."""
        return {
            "lambda1": lambda1,
            "lambda2": lambda2,
            "blur_radius": blur_radius,
        }

    # Generate starting design for each problem based on each configuration
    def design_factory(config: dict) -> dict:
        """Produces starting design for the problem."""
        problem = target_problem(config=config)
        start_design, _ = problem.random_design(noise=0.001)  # Randomized design with noise
        return {"design": start_design}

    # Call the config factory ro generate configurations
    configs = [config_factory(l1, l2, br) for l1, l2, br in combinations]

    # Any optimization configurations can be set here, if you want
    optimize_config = {"num_optimization_steps": 200}

    # Timing information for `optimize` and `simulate` functions for SLURM
    # If you can estimate the time it takes to run `optimize` and `simulate`,
    # you can set the runtimes here, and this will help with job scheduling.
    # Try to be conservative with the time estimates, so SLURM doesn't kill it prematurely.
    # The format is "HH:MM:SS"
    runtime_optimize = "00:12:00"  # ~10 minutes for optimization
    runtime_simulate = "00:02:00"  # ~1 minute for simulation
    runtime_render = "00:02:00"  # ~1 minutes for rendering

    # ============== End of problem-specific elements ===================

    # Now call optimize
    opt_results = optimize_slurm(target_problem, configs, design_factory, optimize_config)

    slurm_simulate_args = make_sim_args_from_opt_results(opt_results)
    if args.render_flag:
        fig_path = args.fig_path
        max_local_render = 50  # Set a threshold for local rendering on login node
        if len(opt_results) < max_local_render:
            # This might be fast, so we can run it locally
            render_local(target_problem, opt_results, fig_path)
        else:
            # This might take while, so we can run it via SLURM
            render_slurm(target_problem, slurm_simulate_args, runtime_render, fig_path)

    # Uncomment the below to run the simulation of the final designs via SLURM
    if args.simulate_flag:
        simulate_slurm(target_problem, slurm_simulate_args, runtime_simulate)
