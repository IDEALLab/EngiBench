#!/bin/bash
source ~/.bashrc
source ~/.bashrc_mdolab

# Takes the number of mpi processes and study folder name as an argument
echo "Running airfoil optimization with $1 MPI processes"
cd /home/mdolabuser/mount/engibench && mpirun -np $1 python $2/airfoil_opt.py > $2/airfoil_opt.log
