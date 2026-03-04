"""Dataset Generation for Airfoil Problem via SLURM.

This script generates a dataset for the Airfoil problem using the SLURM API
"""

from argparse import ArgumentParser

from datasets import load_dataset
import numpy as np
from scipy.stats import qmc

from engibench.problems.airfoil.simulation_jobs import simulate_slurm
from engibench.utils import slurm


def calculate_runtime(group_size, minutes_per_sim=5):
    """Calculate runtime based on group size and (rough) estimate of minutes per simulation."""
    total_minutes = group_size * minutes_per_sim
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}:00"


if __name__ == "__main__":
    """Dataset Generation, Simulation, and Rendering for Airfoil Problem via SLURM.

    This script generates a dataset for the Airfoil problem using the SLURM API, though it could
    be generalized to other problems as well. It includes functions for simulation of designs.

    Command Line Arguments:
    -n_designs, --num_designs: How many airfoil designs should we use?
    -n_flows, --num_flow_conditions: How many flow conditions should we use per design?
    -n_aoas, --num_angles_of_attack: How many angles of attack should we use per design & flow condition pairing?
    -group_size, --group_size: How many simulations should we group together on a single cpu?
    -n_slurm_array, --num_slurm_array: How many slurm jobs to spawn and submit via slurm arrays? Note this may be limited by the HPC system.
    """
    # Fetch command line arguments for render and simulate to know whether to run those functions
    parser = ArgumentParser()
    parser.add_argument(
        "-n_designs",
        "--num_designs",
        type=int,
        default=10,
        help="How many airfoil designs should we use?",
    )
    parser.add_argument(
        "-n_flows",
        "--num_flow_conditions",
        type=int,
        default=1,
        help="How many flow conditions (Mach Number and Reynolds Number) should we sample for each design?",
    )
    parser.add_argument(
        "-n_aoas",
        "--num_angles_of_attack",
        type=int,
        default=1,
        help="How many angles of attack should we sample for each design?",
    )
    parser.add_argument(
        "-group_size",
        "--group_size",
        type=int,
        default=2,
        help="How many simulations do you wish to batch within each individual slurm job?",
    )
    parser.add_argument(
        "-n_slurm_array",
        "--num_slurm_array",
        type=int,
        default=1000,
        help="What is the maximum size of the Slurm array (Will vary from HPC system to HPC system)?",
    )
    args = parser.parse_args()

    n_designs = args.num_designs
    n_flows = args.num_flow_conditions
    n_aoas = args.num_angles_of_attack
    group_size = args.group_size
    n_slurm_array = args.num_slurm_array

    # ============== Problem-specific elements ===================
    # The following elements are specific to the problem and should be modified accordingly

    # Define flow parameter and angle of attack ranges
    Ma_min, Ma_max = 0.5, 0.9  # Mach number range
    Re_min, Re_max = 1.0e6, 2.0e7  # Reynolds number range
    aoa_min, aoa_max = 0.0, 20.0  # Angle of attack range

    # Load airfoil designs from HF Database
    ds = load_dataset("IDEALLab/airfoil_v0")
    designs = (
        ds["train"]["initial_design"]
        + ds["train"]["optimal_design"]
        + ds["val"]["initial_design"]
        + ds["val"]["optimal_design"]
        + ds["test"]["initial_design"]
        + ds["test"]["optimal_design"]
    )

    # Use specified number of designs
    designs = designs[:n_designs]

    # Generate LHS samples
    rng = np.random.default_rng(seed=42)  # Optional seed for reproducibility
    sampler = qmc.LatinHypercube(d=2, seed=rng)
    samples = sampler.random(n=n_designs * n_flows)  # n samples needed

    # Scale to your flow domain
    bounds = np.array([[Ma_min, Ma_max], [Re_min, Re_max]])
    scaled_samples = qmc.scale(samples, bounds[:, 0], bounds[:, 1])
    mach_values = scaled_samples[:, 0]
    reynolds_values = scaled_samples[:, 1]

    # Generate all simulation configurations
    config_id = 0
    simulate_configs_designs = []
    for i, design in enumerate(designs):
        for j in range(n_flows):
            ma = mach_values[i * n_flows + j]
            re = reynolds_values[i * n_flows + j]
            for alpha in rng.uniform(low=aoa_min, high=aoa_max, size=n_aoas):
                problem_configuration = {"mach": ma, "reynolds": re, "alpha": alpha}
                config = {"problem_configuration": problem_configuration, "configuration_id": config_id}
                config["design"] = design["coords"]
                simulate_configs_designs.append(config)
                config_id += 1

    print(f"Generated {len(simulate_configs_designs)} configurations for simulation.")

    # Calculate total number of simulation jobs and number of sbatch maps needed
    n_simulations = len(simulate_configs_designs)
    n_sbatch_maps = np.ceil(n_simulations / (group_size * n_slurm_array))

    slurm_config = slurm.SlurmConfig(
        name="Airfoil_dataset_generation",
        runtime=calculate_runtime(group_size, minutes_per_sim=5),
        ntasks=1,
        cpus_per_task=1,
        log_dir="./sim_logs/",
    )
    print(calculate_runtime(group_size, minutes_per_sim=5))

    submitted_jobs = []
    for ibatch in range(int(n_sbatch_maps)):
        sim_batch_configs = simulate_configs_designs[
            ibatch * group_size * n_slurm_array : (ibatch + 1) * group_size * n_slurm_array
        ]
        print(len(sim_batch_configs))
        print(f"Submitting batch {ibatch + 1}/{int(n_sbatch_maps)}")

        job_array = slurm.sbatch_map(
            f=simulate_slurm,
            args=sim_batch_configs,
            slurm_args=slurm_config,
            group_size=group_size,  # Number of jobs to batch in sequence to reduce job array size
            work_dir="scratch",
        )

        # Save the job array reference
        submitted_jobs.append(job_array)

        # Wait for this job to complete by calling save()
        # This will submit a dependent job that waits for the array to finish
        print(f"Waiting for batch {ibatch + 1} to complete...")
        job_array.save(f"results_{ibatch}.pkl", slurm_args=slurm_config)
        print(f"Batch {ibatch + 1} completed!")
