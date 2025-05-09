"""Dataset Generator for the Photonics2D problem using the updated SLURM API."""

from argparse import ArgumentParser
from itertools import product
import os
import shutil
import time
from typing import Any

import numpy as np

from engibench.problems.photonics2d import Photonics2D
from engibench.utils import slurm


def compute_total_elapsed_time(return_values: list[dict]) -> float:
    """Computes the total elapsed time (in seconds) across all simulations."""
    total_elapsed_time = 0
    for result in return_values:
        # Retrieve all of the result elements
        optimized_design, opti_history, elapsed_time, problem_configuration, configuration_id = result

        # 1. Compute the total elapsed time across all simulations
        total_elapsed_time += elapsed_time
    return total_elapsed_time


def render_all_final_designs(return_values: list[dict], fig_path: str = "result_figures") -> None:
    """Renders all final designs and saves them in a zip file for easy download."""
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    for _i, result in enumerate(return_values):
        # Retrieve all of the result elements
        optimized_design, opti_history, elapsed_time, problem_configuration, configuration_id = result
        problem = Photonics2D(problem_configuration)
        fig = problem.render(design=optimized_design)
        fig.savefig(fig_path + f"/final_design_{_i}.png")
        # Now zip the figure directory
    zip_filename = "figures_all"
    shutil.make_archive(zip_filename, "zip", fig_path)
    print(f"Saved image archive in {zip_filename}.")


def post_process_optimize(return_values: list[dict]):
    """Post-Processing script that can operate on all of the returned results."""
    # In this case, the main purpose of this post-processing step could be two fold:
    # 1. Computing the total time for dataset generation and returning this.
    # 2. Computing all of the final rendered designs and storing this in a zip file
    #    so that users could get images of all the final designs.
    #
    total_elapsed_time = compute_total_elapsed_time(return_values)
    print(f"Total Elapsed time (in seconds): {total_elapsed_time}")

    render_all_final_designs(return_values, "result_figures")


def dict_cartesian_product(d):
    """Generates the cartesian product of a dictionary of lists."""
    keys = d.keys()
    values_product = product(*d.values())
    return [dict(zip(keys, values, strict=True)) for values in values_product]


def generate_configuration_args_for_optimize(params_to_sweep: dict[str, Any]) -> list[dict]:
    """Takes in a Problem and a lists of parameters to sweep, and generates the cartesian product of those params."""
    # Generate all combinations of each parameter across params_to_sweep
    combinations = dict_cartesian_product(params_to_sweep)

    # Now add in a configuration_id to each of the combinations, and assemble this into an args list
    # This is just a unique identifier for each of the combinations, in case you want to debug a given run
    args = []
    for i, config in enumerate(combinations):
        args.append(
            {
                "problem": Photonics2D(config=config),
                "problem_configuration": config,
                "configuration_id": i,
            }
        )
    return args


def optimize_slurm(problem: Photonics2D, problem_configuration: dict, configuration_id: int) -> dict:
    """Takes in the given problem, configuration, and designs, then runs the optimization.

    Any arguments should be things that you want to change across the different jobs, and anything
    that is the same/static across the runs should just be defined inside this function.

    Args:
        problem (Problem): The problem to run the optimization on.
        problem_configuration (dict): The specific configuration used to setup the problem being passed.
            This parameter is just being passed through for reporting, since problem should already be
            configured with these settings.
        configuration_id (int): A unique identifier for the job for later debugging or tracking.

    Returns:
        "optimized_design": The optimized design.
        "opti_history": The optimization history.
        "optimize_time": The time taken to run this optimization job. Useful for aggregating
            the time taken for dataset generation.
    """
    optimization_config = {"num_optimization_steps": 200}

    # Generate an initial design for the problem
    initial_design, _ = problem.random_design(noise=0.1, blur=1)  # Randomized design with noise

    print("Starting `optimize` via SLURM...")
    start_time = time.time()
    optimized_design, opti_history = problem.optimize(starting_point=initial_design, config=optimization_config)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for `optimize`: {elapsed_time:.2f} seconds")

    return {
        "optimized_design": optimized_design,
        "opti_history": opti_history,
        "optimize_time": elapsed_time,
        "problem_configuration": problem_configuration,
        "configuration_id": configuration_id,
    }


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
    params_to_sweep = {
        "lambda1": rng.uniform(low=0.5, high=1.25, size=20),
        "lambda2": rng.uniform(low=0.75, high=1.5, size=20),
        "blur_radius": range(5),
    }

    # For the updated SLURM API, we need to define four things:
    # 1. The job that we want SLURM to run -- this is essentially a factory for different
    #    problem runs and should take in the arguments that we want to vary.
    # 2. An `args` variable, which is all of the arguments that we want to pass to the job.
    #    SLURM will run `job` with each of the arguments in `args`.
    # 3. A function that does the post-processing of the results from the job. This gets passed
    #    via `reduce_job` and can also handle any JobErrors that occur.
    # 4. Lastly, any SLURM Configurations, including runtime, memory, and the group_size

    optimize_configurations = generate_configuration_args_for_optimize(params_to_sweep)
    slurm_config = slurm.SlurmConfig(
        name="Photonics2D_dataset_generation",
        runtime="00:12:00",  # Give 12 minutes for each optimization
        log_dir="./opt_logs/",
    )
    slurm.sbatch_map(
        f=optimize_slurm,
        args=optimize_configurations,
        slurm_args=slurm_config,
        group_size=5,  # Number of jobs to batch in sequence to reduce job array size
        reduce_job=post_process_optimize,
        out="results.pkl",
    )
    results = slurm.load_results()
    for result in results:
        if isinstance(result, slurm.JobError):
            raise result
