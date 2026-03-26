"""Dataset Generator for the Airfoil problem using the SLURM API."""

import time

import numpy as np

from engibench.problems.airfoil.v0 import Airfoil


def simulate_slurm(problem_configuration: dict, configuration_id: int, design: list) -> dict:
    """Takes in the given configuration and designs, then runs the simulation analysis.

    Any arguments should be things that you want to change across the different jobs, and anything
    that is the same/static across the runs should just be defined inside this function.

    Args:
        problem_configuration (dict): The specific configuration used to setup the problem being passed.
            For the airfoil problem this includes Mach number, Reynolds number, and angle of attack.
        configuration_id (int): A unique identifier for the job for later debugging or tracking.
        design (list): list of lists defining x and y coordinates of airfoil geometry.

    Returns:
        "performance_dict": Dictionary of aerodynamic performance (lift & drag).
        "simulate_time": The time taken to run this simulation job. Useful for aggregating
            the time taken for dataset generation.
        "problem_configuration": Problem configuration parameters
        "configuration_id": Identifier for specific simulation configurations
    """
    # Instantiate problem
    problem = Airfoil()

    # Set simulation ID
    sim_id = configuration_id + 1

    # Create unique simulation directory
    problem.reset(seed=sim_id, cleanup=False)

    # Create simulation design (coordinates + angle of attack)
    my_design = {"coords": np.array(design), "angle_of_attack": problem_configuration["alpha"]}

    print("Starting `simulate` via SLURM...")
    start_time = time.time()

    performance = problem.simulate(my_design, mpicores=1, config=problem_configuration)
    performance_dict = {"drag": performance[0], "lift": performance[1]}
    print("Finished `simulate` via SLURM.")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for `simulate`: {elapsed_time:.2f} seconds")

    return {
        "performance_dict": performance_dict,
        "simulate_time": elapsed_time,
        "problem_configuration": problem_configuration,
        "configuration_id": configuration_id,
    }


def optimize_slurm(problem_configuration: dict, configuration_id: int, design: list):
    """Takes starting point (design coordinate and angle of attack) and config (mach, reynolds, angle of attack), then runs the aerodynamic optimization.

    Any arguments should be things that you want to change across the different jobs, and anything
    that is the same/static across the runs should just be defined inside this function.

    Args:
        problem_configuration (dict): The specific configuration used to initialize the optimization being passed.
            For the airfoil problem this includes Mach number, Reynolds number, and angle of attack.
        configuration_id (int): A unique identifier for the job for later debugging or tracking.
        design (list): list of lists defining x and y coordinates of airfoil geometry.

    Returns:
        "performance_dict": Dictionary of aerodynamic performance (lift & drag).
        "optimization_time": The time taken to run this optimization job. Useful for aggregating
            the time taken for dataset generation.
        "optimized_configuration": Problem configuration parameters for optimized design (optimized coordinates and angle of attack)
        "configuration_id": Identifier for specific simulation configurations
    """
