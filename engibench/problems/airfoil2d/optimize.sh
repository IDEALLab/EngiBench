#!/bin/bash
source ~/.bashrc
source ~/.bashrc_mdolab

cd /home/mdolabuser/mount/engibench && mpirun -np 4 python engibench/problems/airfoil2d/airfoil_opt.py --gridFile $1 --ffdFile $2
