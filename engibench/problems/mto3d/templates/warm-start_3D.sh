#!/bin/bash
#The line above this is the "shebang" line.  It must be first line in script
#-----------------------------------------------------
#       Default OnDemand Job Template
#       For a basic Hello World sequential job
#-----------------------------------------------------
#
# Slurm sbatch parameters section:
#SBATCH --job-name="warm_3D"
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --array=6-10
#SBATCH -t 24:00:00
#SBATCH -A fuge-prj-eng
#SBATCH -p standard
#SBATCH --constraint="rhel8"
#SBATCH --output=./scratch/slurm-report/pred_mto3d-%a_%A.out

. ~/.bashrc

cd $cdr

# Optimization
singularity exec -H /root -B $cdr:/root $sif foamCleanTutorials
singularity exec -H /root -B $cdr:/root $sif wmake /root/src
singularity exec -H /root -B $cdr:/root $sif wclean /root/src
singularity exec -H /root -B $cdr:/root $sif blockMesh

singularity exec -H /root -B $cdr:/root $sif decomposePar
mpirun -n $ntasks singularity exec -H /root -B $cdr:/root $sif /root/src/EXEC -parallel
singularity exec -H /root -B $cdr:/root $sif reconstructPar
singularity exec -H /root -B $cdr:/root $sif foamToVTK

rm -rf ./processor*/