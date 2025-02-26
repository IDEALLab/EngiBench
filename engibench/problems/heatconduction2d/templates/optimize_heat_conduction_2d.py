#!/usr/bin/env python3

"""Topology optimization for heat conduction using the SIMP method with dolfin-adjoint.
The script reads initial design data, solves the heat conduction problem, and optimizes
material distribution to minimize thermal complaicen under a volume constraint.
"""

import os
import re
import numpy as np
from fenics import *
from fenics_adjoint import *

# TODO can we clean this up?


# Ensure IPOPT is available
try:
    from pyadjoint import ipopt  # noqa: F401
except ImportError:
    print("""This example depends on IPOPT and Python ipopt bindings. \
    When compiling IPOPT, make sure to link against HSL, as it \
    is a necessity for practical problems.""")
    raise
base_path = "/home/fenics/shared"
OPT_var_path = os.path.join(base_path, "templates", "OPT_var.txt")
with open(OPT_var_path, "r") as fp:
    lines = fp.read()
    lines2 = lines.split("\t")
NN = int(lines2[2])
step = 1.0 / float(NN)
x_values = np.zeros((NN + 1))  # horizontal dir (x(0))
y_values = np.zeros((NN + 1))  # vertical dir (x(1))
x_values = np.linspace(0, 1, num=NN + 1)
y_values = np.linspace(0, 1, num=NN + 1)
max_run_it = 1
vol_f = float(lines2[0])
width = float(lines2[1])
os.system("rm /home/fenics/shared/templates/OPT_var.txt")
filename = "/home/fenics/shared/templates/hr_data_OPT_v=" + str(vol_f) + "_w=" + str(width) + "_.npy"
image = np.load(filename)
os.remove(filename)
mesh1a = UnitSquareMesh(NN, NN)
x = mesh1a.coordinates().reshape((-1, 2))
h = 1.0 / NN
ii, jj = x[:, 0] / h, x[:, 1] / h
ii = np.array(ii, dtype=int)
jj = np.array(jj, dtype=int)
# Turn image into CG1 function
# Values are vertex ordered here
image_values = image[ii, jj]
V = FunctionSpace(mesh1a, "CG", 1)
init_guess = Function(V)
# Values will be dof ordered
d2v = dof_to_vertex_map(V)
image_values = image_values[d2v]
image_values = image_values.reshape(
    -1,
)
init_guess.vector()[:] = image_values

for run_it in range(max_run_it):
    # turn off redundant output in parallel
    parameters["std_out_all_processes"] = False
    V = Constant(vol_f)  # volume bound on the control.   Default = 0.4
    p = Constant(5)  # power used in the solid isotropic material.  Default = 5
    eps = Constant(1.0e-3)  # epsilon used in the solid isotropic material
    alpha = Constant(1.0e-8)  # regularisation coefficient in functional

    def k(a):
        return eps + (1 - eps) * a**p

    mesh = UnitSquareMesh(NN, NN)
    A = FunctionSpace(mesh, "CG", 1)  # function space for control
    P = FunctionSpace(mesh, "CG", 1)  # function space for solution

    lb_2 = 0.5 - width / 2  # lower bound on section of bottom face which is adiabatic
    ub_2 = 0.5 + width / 2  # Upper bound on section of bottom face which is adiabatic

    class WestNorth(SubDomain):
        def inside(self, x, on_boundary):
            return (
                x[0] == 0.0 or x[1] == 1.0 or x[0] == 1.0 or (x[1] == 0.0 and (x[0] < lb_2 or x[0] > ub_2))
            )  # modified from Fuge

    T_bc = 0.0
    bc = [DirichletBC(P, T_bc, WestNorth())]
    f_val = 1.0e-2  # Default = 1.0e-2
    f = interpolate(Constant(f_val), P)  # the volume source term for the PDE

    def forward(a):
        """Solve the forward problem for a given material distribution a(x)."""
        T = Function(P, name="Temperature")
        v = TestFunction(P)
        F = inner(grad(v), k(a) * grad(T)) * dx - f * v * dx
        solve(F == 0, T, bc, solver_parameters={"newton_solver": {"absolute_tolerance": 1.0e-7, "maximum_iterations": 20}})
        return T

    if __name__ == "__main__":
        if run_it == 0:
            MM = init_guess
        else:
            s_xmdf = "/home/fenics/shared/templates/RES_OPT/TEMP.xdmf"
            mesh1a = UnitSquareMesh(NN, NN)
            V1 = FunctionSpace(mesh1a, "CG", 1)
            sol = Function(V1)
            with XDMFFile(s_xmdf) as infile:
                # infile.read(mesh1)
                infile.read_checkpoint(sol, "u")
            MM = sol
        a = interpolate(MM, A)  # initial guess.
        T = forward(a)  # solve the forward problem once.
        controls = File("/home/fenics/shared/templates/RES_OPT/control_iterations" + str(run_it) + ".pvd")
        a_viz = Function(A, name="ControlVisualisation")
    J = assemble(f * T * dx + alpha * inner(grad(a), grad(a)) * dx)
    J_CONTROL = Control(J)
    m = Control(a)
    Jhat = ReducedFunctional(J, m)
    lb = 0.0
    ub = 1.0

    class VolumeConstraint(InequalityConstraint):
        def __init__(self, V):
            self.V = float(V)
            self.smass = assemble(TestFunction(A) * Constant(1) * dx)
            self.tmpvec = Function(A)

        def function(self, m):
            from pyadjoint.reduced_functional_numpy import set_local

            set_local(self.tmpvec, m)
            integral = self.smass.inner(self.tmpvec.vector())
            if MPI.rank(MPI.comm_world) == 0:
                # print("Current control integral: ", integral)
                return [self.V - integral]

        def jacobian(self, m):
            return [-self.smass]

        def output_workspace(self):
            return [0.0]

        def length(self):
            """Return the number of components in the constraint vector (here, one)."""
            return 1

    problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=VolumeConstraint(V))
    # Define filename for IPOPT log
    log_filename = f"/home/fenics/shared/templates/RES_OPT/solution_V={vol_f}_w={width}_it={run_it}.txt"

    parameters = {"acceptable_tol": 1.0e-3, "maximum_iterations": 100, "file_print_level": 5, "output_file": log_filename}
    solver = IPOPTSolver(problem, parameters=parameters)

    a_opt = solver.solve()
    # Read the log file and extract objective values
    # --- Extract Objective Values from the Log File ---
    objective_values = []

    # Open and read the log file
    with open(log_filename, "r") as f:
        for line in f:
            # Match lines that start with an iteration number followed by an objective value
            match = re.match(r"^\s*\d+\s+([-+]?\d*\.\d+e[-+]?\d+)", line)
            if match:
                objective_values.append(float(match.group(1)))  # Extract and convert to float

    # Convert to NumPy array
    objective_values = np.array(objective_values)
    mesh1 = UnitSquareMesh(NN, NN)
    V1 = FunctionSpace(mesh1, "CG", 1)
    sol1 = a_opt
    with XDMFFile("/home/fenics/shared/templates/RES_OPT/TEMP.xdmf") as outfile:
        outfile.write(mesh1)
        outfile.write_checkpoint(sol1, "u", 0, append=True)

    # -------------------------------------------------------------------------------------------
    if run_it == max_run_it - 1:  # if final run reached
        # Now store the RES_OPTults of this run (x,y,v,w,a)
        RES_OPTults = np.zeros(((NN + 1) ** 2, 1))
        ind = 0
        for xs in x_values:
            for ys in y_values:
                RES_OPTults[ind, 0] = a_opt(xs, ys)
                ind = ind + 1
        filename = "/home/fenics/shared/templates/RES_OPT/hr_data_v=" + str(vol_f) + "_w=" + str(width) + "_.npy"
        np.save(filename, RES_OPTults)
        xdmf_filename = XDMFFile(
            MPI.comm_world,
            "/home/fenics/shared/templates/RES_OPT/final_solution_v=" + str(vol_f) + "_w=" + str(width) + "_.xdmf",
        )
        xdmf_filename.write(a_opt)
        print("v=" + "{}".format(vol_f))
        print("w=" + "{}".format(width))
        filenameOUT = "/home/fenics/shared/templates/RES_OPT/OUTPUT=" + str(vol_f) + "_w=" + str(width) + "_.npz"
        np.savez(filenameOUT, design=RES_OPTults, OptiStep=objective_values)
        os.system("rm /home/fenics/shared/templates/RES_OPT/TEMP*")
