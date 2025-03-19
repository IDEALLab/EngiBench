#!/usr/bin/env python3

"""This script sets up and initializes the design problem for a finite element analysis using dolfin adjoint based on SIMP method.
It defines the resolution, reads the design variables, and writes out the initial design to a file.
"""

import os
import numpy as np
from fenics import *

# Define the base directory path for templates and read the design variables
base_path = "/home/fenics/shared"
des_var_path = os.path.join(base_path, "templates", "Des_var.txt")

# Read the design variable file and extract parameters
with open(des_var_path, "r") as file:
    lines = file.read().split("\t")

# Set the number of discretization points (NN) and the volume fraction (vol_f)
NN = int(lines[1])-1  # Resolution of the grid (arbitrary, affects performance)
vol_f = float(lines[0])  # Volume fraction for the control

# Discretization step size (based on NN)
step = 1.0 / float(NN)

# Generate mesh grid values for both x and y directions
x_values = np.linspace(0, 1, num=NN + 1)
y_values = np.linspace(0, 1, num=NN + 1)
os.remove(des_var_path)

# Set up the finite element function space
V = Constant(vol_f)  # Volume bound on the control
mesh = UnitSquareMesh(NN, NN)  # Create a unit square mesh with NN discretization
A = FunctionSpace(mesh, "CG", 1)  # Function space for the control variable

if __name__ == "__main__":
    # Initialize the design variable as a constant volume
    MM = V
    a = interpolate(MM, A)  # Initial guess for the design

    # Define the path to save the design files
    design_folder = os.path.join(base_path, "templates", "initialize_design")
    xdmf_file_path = os.path.join(design_folder, f"initial_v={vol_f}_resol={NN+1}_.xdmf")

    # Write the mesh and initial design to an XDMF file
    with XDMFFile(xdmf_file_path) as outfile:
        outfile.write(mesh)  # Write mesh to the file
        outfile.write_checkpoint(a, "u", 0, append=True)  # Write design variable to the file

    # Initialize an array to store the results (x, y, value)
    results = np.zeros(((NN + 1) ** 2, 1))

    # Populate the results array with the mesh coordinates and the corresponding volume value
    ind = 0
    for xs in x_values:
        for ys in y_values:
            results[ind, 0] = V  # Store the volume value
            ind += 1
    results=results.reshape(NN+1, NN+1)
    # Save the results array to a .npy file
    filename = os.path.join(design_folder, f"initial_v={vol_f}_resol={NN+1}_.npy")
    np.save(filename, results)
