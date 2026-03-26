"""Dataset Generator for the Airfoil problem using the SLURM API."""

import time

import numpy as np

from engibench.problems.airfoil.v0 import Airfoil


def simulate_slurm(problem_configuration: dict, configuration_id: int, design: list, *, field_output: bool = False) -> dict:
    """Takes in the given configuration and designs, then runs the simulation analysis.

    Any arguments should be things that you want to change across the different jobs, and anything
    that is the same/static across the runs should just be defined inside this function.

    Args:
        problem_configuration (dict): The specific configuration used to setup the problem being passed.
            For the airfoil problem this includes Mach number, Reynolds number, and angle of attack.
        configuration_id (int): A unique identifier for the job for later debugging or tracking.
        design (list): list of lists defining x and y coordinates of airfoil geometry.
        field_output (bool): If True, surface field variables (velocity components and pressure
            coefficient) are extracted from the simulation and included in the returned dict under
            the key ``"surface_fields"``.

    Returns:
        "performance_dict": Dictionary of aerodynamic performance (lift & drag).
        "simulate_time": The time taken to run this simulation job. Useful for aggregating
            the time taken for dataset generation.
        "problem_configuration": Problem configuration parameters
        "configuration_id": Identifier for specific simulation configurations
        "surface_fields": Array of shape ``(6, N)`` with rows
            ``[x, y, VelocityX, VelocityY, VelocityZ, CoefPressure]`` (only present when
            ``field_output=True``).
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

    if field_output:
        performance, surface_fields = problem.simulate_field(my_design, mpicores=1, config=problem_configuration)
        performance_dict = {"drag": performance[0], "lift": performance[1], "surface_fields": surface_fields}
    else:
        performance = problem.simulate(my_design, mpicores=1, config=problem_configuration)
        performance_dict = {"drag": performance[0], "lift": performance[1]}
    print("Finished `simulate` via SLURM.")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for `simulate`: {elapsed_time:.2f} seconds")

    result = {
        "performance_dict": performance_dict,
        "simulate_time": elapsed_time,
        "problem_configuration": problem_configuration,
        "configuration_id": configuration_id,
    }
    if field_output:
        result["surface_fields"] = surface_fields
    return result


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
