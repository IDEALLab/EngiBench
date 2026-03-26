"""Dataset Generation for Airfoil Problem via SLURM.

This script generates a dataset for the Airfoil problem using the SLURM API
"""

from argparse import ArgumentParser
import sys

from datasets import concatenate_datasets
from datasets import load_dataset
import numpy as np
from scipy.stats import qmc

from engibench.problems.airfoil.simulation_jobs import simulate_slurm
from engibench.utils import slurm


def calculate_runtime(group_size, minutes_per_sim=6):
    """Calculate runtime based on group size and (rough) estimate of minutes per simulation."""
    total_minutes = group_size * minutes_per_sim
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}:00"


if __name__ == "__main__":
    """Dataset Generation and Simulation for Airfoil Problem via SLURM.

    This script generates a dataset for the Airfoil problem using the SLURM API, though it could
    be generalized to other problems as well. It includes functions for simulation of designs.

    Command Line Arguments:
    -n_designs, --num_designs: How many airfoil designs should we use?
    -n_flows, --num_flow_conditions: How many flow conditions should we use per design?
    -n_aoas, --num_angles_of_attack: How many angles of attack should we use per design & flow condition pairing?
    -group_size, --group_size: How many simulations should we group together on a single cpu?
    -n_slurm_array, --num_slurm_array: How many slurm jobs to spawn and submit via slurm arrays? Note this may be limited by the HPC system.
    -min_ma, --min_mach_number: Lower bound for mach number
    -max_ma, --max_mach_number: Upper bound for mach number
    -min_re, --min_reynolds_number: Lower bound for reynolds number
    -max_re, --max_reynolds_number: Upper bound for reynolds number
    -min_aoa, --min_angle_of_attack: Lower bound for angle of attack
    -max_aoa, --max_angle_of_attack: Upper bound for angle of attack
    """
    # Fetch command line arguments for render and simulate to know whether to run those functions
    parser = ArgumentParser()
    parser.add_argument(
        "-account",
        "--hpc_account",
        type=str,
        required=True,
        help="HPC account allocation to charge for job submission",
    )
    parser.add_argument(
        "-type",
        "--job_type",
        type=str,
        required=True,
        help="Engibench job type (simulate or optimization) for submission",
    )
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
        help="How many flow conditions (Mach Number, Reynolds Number, Angle of Attack) should we sample for each design?",
    )
    parser.add_argument(
        "-group_size",
        "--group_size",
        type=int,
        default=2,
        help="How many simulations do you wish to batch within each individual slurm job?",
    )
    parser.add_argument(
        "-minutes_per_sim",
        "--minutes_per_simulation",
        type=int,
        default=15,
        help="How long will each individual SLURM job take (in minutes) for either simulate or optimize? This is used to calculate the runtime for each job submission.",
    )
    parser.add_argument(
        "-n_slurm_array",
        "--num_slurm_array",
        type=int,
        default=1000,
        help="What is the maximum size of the Slurm array (Will vary from HPC system to HPC system)?",
    )
    parser.add_argument(
        "-min_ma",
        "--min_mach_number",
        type=float,
        default=0.5,
        help="Minimum sampling bound for Mach Number.",
    )
    parser.add_argument(
        "-max_ma",
        "--max_mach_number",
        type=float,
        default=0.9,
        help="Minimum sampling bound for Mach Number.",
    )
    parser.add_argument(
        "-min_re",
        "--min_reynolds_number",
        type=float,
        default=1.0e6,
        help="Minimum sampling bound for Reynolds Number.",
    )
    parser.add_argument(
        "-max_re",
        "--max_reynolds_number",
        type=float,
        default=2.0e7,
        help="Minimum sampling bound for Reynolds Number.",
    )
    parser.add_argument(
        "-min_aoa",
        "--min_angle_of_attack",
        type=float,
        default=0.0,
        help="Minimum sampling bound for angle of attack.",
    )
    parser.add_argument(
        "-max_aoa",
        "--max_angle_of_attack",
        type=float,
        default=20.0,
        help="Minimum sampling bound for angle of attack.",
    )
    args = parser.parse_args()

    # HPC account for job submission
    hpc_account = args.hpc_account

    # Job type (simulate or optimize)
    job_type = args.job_type
    if job_type not in ["simulate", "optimize"]:
        raise ValueError(f"Invalid job type: {job_type}. Must be 'simulate' or 'optimize'.")

    # Number of samples & flow conditions
    n_designs = args.num_designs
    n_conditions = args.num_flow_conditions

    # Slurm parameters
    group_size = args.group_size
    n_slurm_array = args.num_slurm_array
    minutes_per_sim = args.minutes_per_simulation

    # Flow parameter and angle of attack ranges
    min_ma = args.min_mach_number
    max_ma = args.max_mach_number
    min_re = args.min_reynolds_number
    max_re = args.max_reynolds_number
    min_aoa = args.min_angle_of_attack
    max_aoa = args.max_angle_of_attack

    # ============== Problem-specific elements ===================
    # The following elements are specific to the problem and should be modified accordingly

    # Define flow parameter and angle of attack ranges
    print(f"Mach number:      {min_ma:.2e} to {max_ma:.2e}")
    print(f"Reynolds number:  {min_re:.2e} to {max_re:.2e}")
    print(f"Angle of attack:  {min_aoa:.1f} to {max_aoa:.1f}")

    # --- Dataset Loading ---
    ds = load_dataset("IDEALLab/airfoil_v0")
    all_data = concatenate_datasets([ds[split] for split in ds])
    designs = all_data["initial_design"] + all_data["optimal_design"]
    if n_designs < len(designs):
        designs = designs[:n_designs]

    # --- Config Generation ---
    config_id = 0
    simulate_configs_designs = []
    for design in designs:
        # Generate LHS samples
        sampler = qmc.LatinHypercube(d=3)
        samples = sampler.random(n=n_conditions)  # n samples needed

        # Scale to your flow domain
        bounds = np.array([[min_ma, max_ma], [min_re, max_re], [min_aoa, max_aoa]])
        scaled_samples = qmc.scale(samples, bounds[:, 0], bounds[:, 1])
        mach_values = scaled_samples[:, 0]
        reynolds_values = scaled_samples[:, 1]
        aoa_values = scaled_samples[:, 2]

        for j in range(n_conditions):
            ma = mach_values[j]
            re = reynolds_values[j]
            alpha = aoa_values[j]

            problem_configuration = {"mach": ma, "reynolds": re, "alpha": alpha}
            config = {"problem_configuration": problem_configuration, "configuration_id": config_id}
            config["design"] = design["coords"]
            simulate_configs_designs.append(config)
            config_id += 1

    # Calculate total number of simulation jobs and number of sbatch maps needed
    n_simulations = len(simulate_configs_designs)
    n_sbatch_maps = np.ceil(n_simulations / (group_size * n_slurm_array))

    slurm_config = slurm.SlurmConfig(
        name="Airfoil_dataset_generation",
        runtime=calculate_runtime(group_size, minutes_per_sim=minutes_per_sim),
        account=hpc_account,
        ntasks=1,
        cpus_per_task=1,
        log_dir="./sim_logs/",
    )

    submitted_jobs = []
    for ibatch in range(int(n_sbatch_maps)):
        sim_batch_configs = simulate_configs_designs[
            ibatch * group_size * n_slurm_array : (ibatch + 1) * group_size * n_slurm_array
        ]
        print(f"Submitting batch {ibatch + 1}/{int(n_sbatch_maps)}")

        if job_type == "simulate":
            job_array = slurm.sbatch_map(
                f=simulate_slurm,
                args=sim_batch_configs,
                slurm_args=slurm_config,
                group_size=group_size,  # Number of jobs to batch in sequence to reduce job array size
                work_dir="scratch",
            )
        elif job_type == "optimize":
            sys.exit(
                'Optimize function not yet implemented for SLURM dataset generation script. Please use "simulate" for job_type argument.'
            )

        # Save the job array reference
        submitted_jobs.append(job_array)

        # Wait for this job to complete by calling save()
        # This will submit a dependent job that waits for the array to finish
        print(f"Waiting for batch {ibatch + 1} to complete...")
        job_array.save(f"results_{ibatch}.pkl", slurm_args=slurm_config)
        print(f"Batch {ibatch + 1} completed!")
