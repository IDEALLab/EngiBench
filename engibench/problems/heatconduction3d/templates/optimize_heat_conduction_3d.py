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


# Ensure IPOPT is available
try:
    from pyadjoint import ipopt
except ImportError:
    print("""This example depends on IPOPT and Python ipopt bindings. \
    When compiling IPOPT, make sure to link against HSL, as it \
    is a necessity for practical problems.""")
    raise
# Define base paths and read optimization variables
base_path = "/home/fenics/shared"
OPT_var_path = os.path.join(base_path, "templates", "OPT_var.txt")
# Open and read the optimization variable file
with open(OPT_var_path, "r") as file:
    data = file.read().split("\t")
# Extract parameters
NN = int(data[2]) - 1  # Grid size
vol_f = float(data[0])  # Volume fraction
width = float(data[1])  # Adiabatic boundary width

# Compute step size
step = 1.0 / float(NN)

# Generate x, y , and z coordinate values
x_values = np.linspace(0, 1, num=NN + 1)
y_values = np.linspace(0, 1, num=NN + 1)
z_values = np.linspace(0, 1, num=NN + 1)

# Remove simulation variable file after reading
os.remove(OPT_var_path)
# -------------------------------
# Load Initial Design Data
# -------------------------------
input_filename = f"templates/hr_data_OPT_v={vol_f}_w={width}_.npy"
input_path = os.path.join(base_path, input_filename)

# Load initial design image
image = np.load(input_path)
os.remove(input_path)  # Remove after loading

# -------------------------------
# Mesh and Function Space Setup
# -------------------------------

# Create computational mesh
mesh = UnitCubeMesh(NN, NN, NN)

# Map image data to mesh vertices
x = mesh.coordinates().reshape((-1, 3))
h = 1.0 / NN
ii, jj, kk = np.array(x[:, 0] / h, dtype=int), np.array(x[:, 1] / h, dtype=int), np.array(x[:, 2] / h, dtype=int)

# Extract image values corresponding to mesh vertices
image_values = image[ii, jj, kk]

# Define function space
V = FunctionSpace(mesh, "CG", 1)
# Initialize function for initial guess
init_guess = Function(V)

# Map values to function space degrees of freedom
d2v = dof_to_vertex_map(V)
init_guess.vector()[:] = image_values[d2v].reshape(
    -1,
)
# -------------------------------
# Define Material Properties and Boundary Conditions
# -------------------------------
# Define parameters for optimization
p = Constant(5)  # Power in material model
eps = Constant(1e-3)  # Regularization parameter
alpha = Constant(1e-8)  # Functional regularization coefficient


def k(a):
    """Material property function based on design variable 'a'."""
    return eps + (1 - eps) * a**p


# Define function spaces for control and solution
A = FunctionSpace(mesh, "CG", 1)  # Control variable space
P = FunctionSpace(mesh, "CG", 1)  # Temperature solution space

# Define adiabatic boundary region
lb_2, ub_2 = 0.5 - width / 2, 0.5 + width / 2


class BoundaryConditions(SubDomain):
    """Defines Dirichlet boundary conditions on specific edges."""

    def inside(self, x, on_boundary):
        return (
            (x[2] > 0 and x[0] == 0)
            or (x[2] > 0 and x[0] == 1)
            or (x[2] > 0 and x[1] == 0)
            or (x[2] > 0 and x[1] == 1)
            or (x[2] == 1)
            or (x[2] == 0 and (x[0] < lb_2 or x[0] > ub_2) and (x[1] < lb_2 or x[1] > ub_2))
        )


# Apply boundary condition: Temperature = 0
T_bc = 0.0
bc = [DirichletBC(P, T_bc, BoundaryConditions())]

# Define heat source term
f = interpolate(Constant(1.0e-2), P)  # Default source term

# -------------------------------
# Forward Heat Conduction Simulation
# -------------------------------
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"


def forward(a):
    """Solve the heat conduction PDE given a material distribution 'a'."""
    T = Function(P, name="Temperature")
    v = TestFunction(P)

    # Define variational form
    F = inner(grad(v), k(a) * grad(T)) * dx - f * v * dx

    # Solve PDE
    solve(
        F == 0,
        T,
        bc,
        solver_parameters={
            "newton_solver": {
                "absolute_tolerance": 1.0e-7,
                "maximum_iterations": 20,
                "linear_solver": "cg",
                "preconditioner": "petsc_amg",
            }
        },
    )

    return T


# -------------------------------
# Optimization Process
# -------------------------------

# Initialize control variable
a = interpolate(init_guess, A)

# Solve forward problem
T = forward(a)
controls = File("/home/fenics/shared/templates/RES_OPT/control_iterations.pvd")
a_viz = Function(A, name="ControlVisualisation")
# Define optimization objective function (cost function)
J = assemble(f * T * dx + alpha * inner(grad(a), grad(a)) * dx)

# Define control object for optimization
m = Control(a)
Jhat = ReducedFunctional(J, m)
J_CONTROL = Control(J)
# Define optimization bounds
lb, ub = 0.0, 1.0


class VolumeConstraint(InequalityConstraint):
    """Constraint to maintain volume fraction."""

    def __init__(self, V):
        self.V = float(V)
        self.smass = assemble(TestFunction(A) * Constant(1) * dx)
        self.tmpvec = Function(A)

    def function(self, m):
        """Compute volume constraint value."""
        from pyadjoint.reduced_functional_numpy import set_local

        set_local(self.tmpvec, m)
        integral = self.smass.inner(self.tmpvec.vector())
        return [self.V - integral] if MPI.rank(MPI.comm_world) == 0 else []

    def jacobian(self, m):
        """Compute Jacobian of volume constraint."""
        return [-self.smass]

    def output_workspace(self):
        return [0.0]

    def length(self):
        """Return number of constraint components (1)."""
        return 1


# Define optimization problem
problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=VolumeConstraint(vol_f))
# Define filename for IPOPT log
log_filename = f"/home/fenics/shared/templates/RES_OPT/solution_V={vol_f}_w={width}.txt"
# Set optimization solver parameters
solver_params = {"acceptable_tol": 1.0e-100, "maximum_iterations": 100, "file_print_level": 5, "output_file": log_filename}
solver = IPOPTSolver(problem, parameters=solver_params)
# -------------------------------
# Store and Save Results
# -------------------------------

# Solve optimization problem
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
# Save optimized design
mesh_output = UnitCubeMesh(NN, NN, NN)
V_output = FunctionSpace(mesh_output, "CG", 1)
sol_output = a_opt
output_xdmf = XDMFFile("/home/fenics/shared/templates/RES_OPT/final_solution_v={}_w={}.xdmf".format(vol_f, width))
output_xdmf.write(a_opt)
# Now store the RES_OPTults of this run (x,y,v,w,a)
RES_OPTults = np.zeros(((NN + 1) ** 3, 1))
ind = 0
for xs in x_values:
    for ys in y_values:
        for zs in z_values:
            RES_OPTults[ind, 0] = a_opt(xs, ys, zs)
            ind = ind + 1
RES_OPTults = RES_OPTults.reshape(NN + 1, NN + 1, NN + 1)
output_npy = "/home/fenics/shared/templates/RES_OPT/hr_data_v_v={}_w={}.npy".format(vol_f, width)
np.save(output_npy, RES_OPTults)
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
