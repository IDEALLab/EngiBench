#!/bin/bash
source ~/.bashrc
source ~/.bashrc_mdolab

cd /home/mdolabuser/mount/engibench && pip install .
cd /home/mdolabuser/mount/engibench && python engibench/problems/airfoil2d/pre_process.py --input-fname $1 --output-fname $2
