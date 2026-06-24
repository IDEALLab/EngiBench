"""Optimized Airfoil Dataset Generation via SLURM.

This script generates a dataset of optimized airfoil designs using the SLURM API.
For each starting design and flow condition, it runs the aerodynamic optimizer and
collects the resulting optimized geometry and performance.
"""

from argparse import ArgumentParser
import sys

from datasets import concatenate_datasets
from datasets import load_dataset
import numpy as np
from scipy.stats import qmc

from engibench.problems.airfoil.simulation_jobs import optimize_slurm
from engibench.utils import slurm


def calculate_runtime(group_size, minutes_per_opt=60):
    """Calculate runtime based on group size and (rough) estimate of minutes per optimization."""
    total_minutes = group_size * minutes_per_opt
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}:00"


if __name__ == "__main__":
    """Optimized Airfoil Dataset Generation via SLURM.

    For each starting design and sampled flow condition, runs the aerodynamic optimizer
    and saves the optimized geometry, angle of attack, and aerodynamic performance.

    Command Line Arguments:
    -account, --hpc_account: HPC account allocation to charge for job submission.
    -n_designs, --num_designs: How many starting airfoil designs should we use?
    -n_flows, --num_flow_conditions: How many flow conditions should we sample per design?
    -group_size, --group_size: How many optimization jobs to batch within each SLURM job.
    -minutes_per_opt, --minutes_per_optimization: Estimated minutes per optimization job.
    -n_slurm_array, --num_slurm_array: Maximum SLURM array size (varies by HPC system).
    -min_ma, --min_mach_number: Lower bound for Mach number.
    -max_ma, --max_mach_number: Upper bound for Mach number.
    -min_re, --min_reynolds_number: Lower bound for Reynolds number.
    -max_re, --max_reynolds_number: Upper bound for Reynolds number.
    -min_aoa, --min_angle_of_attack: Lower bound for angle of attack.
    -max_aoa, --max_angle_of_attack: Upper bound for angle of attack.
    -return_history, --return_history: Whether to include optimizer step history in results.
    --extra_args: Additional arguments forwarded verbatim to sbatch (e.g. --extra_args '--partition=gpu' --extra_args '--gres=gpu:1').
    """
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "-account",
        "--hpc_account",
        type=str,
        required=True,
        help="HPC account allocation to charge for job submission",
    )
    parser.add_argument(
        "-n_designs",
        "--num_designs",
        type=int,
        default=5,
        help="How many starting airfoil designs should we use?",
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
        default=1,
        help="How many optimization jobs do you wish to batch within each individual SLURM job?",
    )
    parser.add_argument(
        "-minutes_per_opt",
        "--minutes_per_optimization",
        type=int,
        default=60,
        help="How long will each individual optimization job take (in minutes)? Used to calculate the SLURM runtime.",
    )
    parser.add_argument(
        "-n_slurm_array",
        "--num_slurm_array",
        type=int,
        default=1000,
        help="What is the maximum size of the SLURM array (will vary from HPC system to HPC system)?",
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
        help="Maximum sampling bound for Mach Number.",
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
        help="Maximum sampling bound for Reynolds Number.",
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
        help="Maximum sampling bound for angle of attack.",
    )
    parser.add_argument(
        "-min_temp",
        "--min_temperature",
        type=float,
        default=220.0,
        help="Minimum sampling bound for freestream temperature [K].",
    )
    parser.add_argument(
        "-max_temp",
        "--max_temperature",
        type=float,
        default=310.0,
        help="Maximum sampling bound for freestream temperature [K].",
    )
    parser.add_argument(
        "-return_history",
        "--return_history",
        action="store_true",
        default=False,
        help="Include optimizer step history (optisteps_history) in results.",
    )
    parser.add_argument(
        "-coords_file",
        "--coords_file",
        type=str,
        default=None,
        help="Optional path to a local .npy of baseline coordinates to use as starting designs "
        "(shape (N, P, 2) or (N, 2, P)). If omitted, initial designs are loaded from Hugging Face.",
    )
    parser.add_argument(
        "--extra_args",
        action="append",
        default=[],
        metavar="ARG",
        help=(
            "Additional argument forwarded verbatim to sbatch. "
            "Repeat the flag for each extra argument. "
            "Example: --extra_args '--partition=gpu' --extra_args '--gres=gpu:1'"
        ),
    )
    # Rewrite "--extra_args -something" to "--extra_args=-something" so argparse
    # does not mistake the value for a new flag.
    argv = sys.argv[1:]
    rewritten = []
    i = 0
    while i < len(argv):
        if argv[i] == "--extra_args" and i + 1 < len(argv) and argv[i + 1].startswith("-"):
            rewritten.append(f"--extra_args={argv[i + 1]}")
            i += 2
        else:
            rewritten.append(argv[i])
            i += 1
    args = parser.parse_args(rewritten)

    # HPC account for job submission
    hpc_account = args.hpc_account

    # Number of samples & flow conditions
    n_designs = args.num_designs
    n_conditions = args.num_flow_conditions

    # SLURM parameters
    group_size = args.group_size
    n_slurm_array = args.num_slurm_array
    minutes_per_opt = args.minutes_per_optimization
    return_history = args.return_history
    extra_args = args.extra_args

    # Flow parameter and angle of attack ranges
    min_ma = args.min_mach_number
    max_ma = args.max_mach_number
    min_re = args.min_reynolds_number
    max_re = args.max_reynolds_number
    min_aoa = args.min_angle_of_attack
    max_aoa = args.max_angle_of_attack
    min_temp = args.min_temperature
    max_temp = args.max_temperature

    # ============== Problem-specific elements ===================

    print(f"Mach number:      {min_ma:.2e} to {max_ma:.2e}")
    print(f"Reynolds number:  {min_re:.2e} to {max_re:.2e}")
    print(f"Angle of attack:  {min_aoa:.1f} to {max_aoa:.1f}")
    print(f"Temperature [K]:  {min_temp:.1f} to {max_temp:.1f}")

    # --- Dataset Loading ---
    # Initial designs (starting points). Either a local .npy of coordinates
    # (shape (N, P, 2) or (N, 2, P)) via --coords_file, or the Hugging Face dataset.
    if args.coords_file is not None:
        raw = np.load(args.coords_file, allow_pickle=True)
        coords_list = []
        for entry in raw:
            arr = np.asarray(entry, dtype=float)
            # Normalize to (2, P): the per-design array is either (P, 2) or (2, P).
            if arr.shape[0] != 2:  # noqa: PLR2004
                arr = arr.T
            coords_list.append(arr)
        designs = [{"coords": c} for c in coords_list]
    else:
        ds = load_dataset("IDEALLab/airfoil_v0")
        all_data = concatenate_datasets([ds[split] for split in ds])
        designs = all_data["initial_design"]
    if n_designs < len(designs):
        designs = designs[:n_designs]

    # --- Config Generation ---
    config_id = 0
    optimize_configs = []
    for design in designs:
        sampler = qmc.LatinHypercube(d=4)
        samples = sampler.random(n=n_conditions)

        bounds = np.array([[min_ma, max_ma], [min_re, max_re], [min_aoa, max_aoa], [min_temp, max_temp]])
        scaled_samples = qmc.scale(samples, bounds[:, 0], bounds[:, 1])
        mach_values = scaled_samples[:, 0]
        reynolds_values = scaled_samples[:, 1]
        aoa_values = scaled_samples[:, 2]
        temperature_values = scaled_samples[:, 3]

        for j in range(n_conditions):
            problem_configuration = {
                "mach": mach_values[j],
                "reynolds": reynolds_values[j],
                "alpha": aoa_values[j],
                "temperature": temperature_values[j],
            }
            config = {
                "problem_configuration": problem_configuration,
                "configuration_id": config_id,
                "design": design["coords"],
                "return_history": return_history,
            }
            optimize_configs.append(config)
            config_id += 1

    # Calculate total number of optimization jobs and number of sbatch maps needed
    n_optimizations = len(optimize_configs)
    n_sbatch_maps = np.ceil(n_optimizations / (group_size * n_slurm_array))

    print(f"Total optimization jobs: {n_optimizations}")
    print(f"Submitting in {int(n_sbatch_maps)} batch(es) of up to {group_size * n_slurm_array} jobs each")

    slurm_config = slurm.SlurmConfig(
        name="Airfoil_optimize_dataset_generation",
        runtime=calculate_runtime(group_size, minutes_per_opt=minutes_per_opt),
        account=hpc_account,
        ntasks=1,
        cpus_per_task=1,
        log_dir="./opt_logs/",
        extra_args=extra_args,
    )

    submitted_jobs = []
    for ibatch in range(int(n_sbatch_maps)):
        opt_batch_configs = optimize_configs[
            ibatch * group_size * n_slurm_array : (ibatch + 1) * group_size * n_slurm_array
        ]
        print(f"Submitting batch {ibatch + 1}/{int(n_sbatch_maps)}")

        job_array = slurm.sbatch_map(
            f=optimize_slurm,
            args=opt_batch_configs,
            slurm_args=slurm_config,
            group_size=group_size,
            work_dir="scratch",
        )

        submitted_jobs.append(job_array)

        print(f"Waiting for batch {ibatch + 1} to complete...")
        job_array.save(f"opt_results_{ibatch}.pkl", slurm_args=slurm_config)
        print(f"Batch {ibatch + 1} completed!")
