#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -A fuge-prj-jrl

# Set appropriate environment variables
export OMP_NUM_THREADS=1
export APPTAINER_HOME=/home/fvangess/scratch/EngiBench
export APPTAINER_CACHEDIR=$APPTAINER_HOME/apptainer-cache
module load apptainer
source ../engibench_env/bin/activate

echo "Cleaning up Workspace"
rm slurm-*.out
rm results_*.pkl
rm sim_logs/*
rm -rf scratch/*
rm -rf engibench_studies/*
echo "Running Dataset Generation"

python ../engibench/problems/airfoil/dataset_slurm_airfoil.py \
    -type simulate \
    -account fuge-prj-jrl \
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

# python ../engibench/problems/airfoil/dataset_slurm_airfoil.py \
#     -type simulate \
#     -account fuge-prj-jrl \
#     -n_designs 10 \
#     -n_flows 1 \
#     -group_size 2 \
#     -n_slurm_array 1000 \
#     -min_ma 0.25 \
#     -max_ma 0.75 \
#     -min_re 1.0e6 \
#     -max_re 1.0e7 \
#     -min_aoa 0.0 \
#     -max_aoa 10.0
