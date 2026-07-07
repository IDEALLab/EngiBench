# Dataset Generation via Slurm

EngiBench ships ready-to-use datasets for most workflows. When you need to generate additional simulation or optimization samples, use a problem-provided dataset-generation entry point and submit it through your HPC cluster's Slurm scheduler. This page shows the end-to-end pattern with the Airfoil dataset-generation script; for the lower-level callback API, see the [Slurm utilities](../utils/slurm.md).

## When to use this workflow

Use this workflow when you need to create new samples rather than only loading the published Hugging Face dataset through `problem.dataset`. The exact entry point and command-line arguments are problem-specific, but the overall pattern is:

1. write a small shell script with the cluster resources and environment setup;
2. activate an environment with EngiBench installed;
3. call the problem's dataset-generation Python script; and
4. submit the shell script with `sbatch`.

## Airfoil example submission script

The Airfoil problem includes a dataset-generation entry point at [`engibench/problems/airfoil/dataset_slurm_airfoil.py`](source:engibench/problems/airfoil/dataset_slurm_airfoil.py). The script below submits a small simulation dataset-generation run. Save it as `dataset_slurm_airfoil.sh` and adjust paths, module names, and resource settings for your cluster.

```bash
#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH -n 1
#SBATCH -c 1

export OMP_NUM_THREADS=1

# Apptainer image cache. Using $HOME keeps the script portable across clusters.
export APPTAINER_HOME=$HOME/scratch/EngiBench
export APPTAINER_CACHEDIR=$APPTAINER_HOME/apptainer-cache

# Load the Apptainer module if your cluster requires it. For example, this is
# not required on ETH's Euler cluster, where Apptainer is available by default
# and no such module exists.
module load apptainer

# Activate a preconfigured Python environment with EngiBench installed.
# Adjust the path to your virtual environment. The convention used by `uv` and
# most Python tooling is `.venv`; this example assumes the virtual environment
# lives in the parent directory.
source ../.venv/bin/activate

# Run the dataset-generation Python file. The CLI exposes many parameters of
# the dataset generation, including the number of LHS samples and the Mach,
# Reynolds, and angle-of-attack ranges. Further customization, such as changing
# the sampling strategy or algorithm, requires editing the Python file.
python ../engibench/problems/airfoil/dataset_slurm_airfoil.py \
    -type simulate \
    -account "$SLURM_JOB_ACCOUNT" \
    -n_designs 5 \
    -n_flows 1 \
    -group_size 1 \
    -minutes_per_sim 5 \
    -n_slurm_array 1000 \
    -min_ma 0.25 \
    -max_ma 0.75 \
    -min_re 1.0e6 \
    -max_re 1.0e7 \
    -min_aoa 0.0 \
    -max_aoa 10.0 \
    --field_output
```

## Submit the job

Submit the script, passing your Slurm account on the command line:

```bash
sbatch -A <your-account> dataset_slurm_airfoil.sh
```

The account is exposed inside the job via `$SLURM_JOB_ACCOUNT` and forwarded to `dataset_slurm_airfoil.py` through the `-account` flag. The dataset-generation script uses that account for the worker array jobs it spawns internally.

## Cluster-specific settings

Before running a large dataset-generation job, check your cluster's Slurm policy and start with a small test run. In particular:

- adjust the `module load apptainer` line to match your cluster, or remove it if Apptainer is available by default;
- adjust the virtual environment path so the job activates the environment where EngiBench is installed;
- start with small values for `-n_designs`, `-n_flows`, `-group_size`, and `-minutes_per_sim`; and
- keep `-n_slurm_array` within the job-array limit recommended by your cluster.
