#!/bin/bash
source ~/.bashrc
source ~/.bashrc_mdolab

# Takes the study folder name as an argument
cd /home/mdolabuser/mount/engibench && python $1/pre_process.py
