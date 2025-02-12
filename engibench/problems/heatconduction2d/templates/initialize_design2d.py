#!/usr/bin/env python3
import glob, os, sys
import time as tm
import numpy as np
import time as tm
from math import floor
from fenics import *

#with open(r"templates/Des_var.txt", 'r') as fp:
base_path = "/home/fenics/shared"
des_var_path = os.path.join(base_path, "templates", "Des_var.txt")

with open(des_var_path, 'r') as fp:
    lines = fp.read()
    lines2=lines.split("\t")
NN = int(lines2[1]) #70 for experiments #discretization resolution: somewhat arbitrary. NOTE: Increasing the int coeff dramatically increases model training and testing #time!!!
step = 1.0/float(NN)
x_values = np.zeros((NN+1)) #horizontal dir (x(0))
y_values = np.zeros((NN+1)) #vertical dir (x(1))
x_values=np.linspace(0,1,num=NN+1)
y_values=np.linspace(0,1,num=NN+1)
vol_f = float(lines2[0])
###os.system('rm templates/Des_var.txt')
os.remove(des_var_path)
#Now set up

V = Constant(vol_f)  # volume bound on the control.   Default = 0.4
mesh = UnitSquareMesh(NN, NN)
A = FunctionSpace(mesh, "CG", 1)  # function space for control
if __name__ == "__main__":
    MM = V
    a = interpolate(MM, A)  # initial guess.
    #xdmf_filename = XDMFFile(MPI.comm_world, "Design/initial_v="+str(vol_f)+"_resol="+str(NN)+"_.xdmf")
    #xdmf_filename.write(a)

    #design_path = os.path.abspath("templates/Design/")
    #xdmf_filename = os.path.join(design_path, f"initial_v={vol_f}_resol={NN}_.xdmf")
    design_folder = os.path.join(base_path, "templates", "initialize_design")
    xdmf_file_path = os.path.join(design_folder, f"initial_v={vol_f}_resol={NN}_.xdmf")
    #with XDMFFile("templates/Design/initial_v="+str(vol_f)+"_resol="+str(NN)+"_.xdmf") as outfile:
    with XDMFFile(xdmf_file_path) as outfile:


        outfile.write(mesh)
        outfile.write_checkpoint(a, "u", 0, append=True)
    results = np.zeros(((NN+1)**2,3))
    ind = 0
    for xs in x_values:
        for ys in y_values:
            results[ind,0] = xs
            results[ind,1] = ys
            results[ind,2] =V
            ind = ind+1
    filename = os.path.join(design_folder, f"initial_v={vol_f}_resol={NN}_.npy")
    #filename = "templates/Design/initial_v="+str(vol_f)+"_resol="+str(NN)+"_.npy"
    np.save(filename,results)
