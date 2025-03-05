#!/usr/bin/env python3

"""This script evaluates the design using finite element analysis with dolfin-adjoint based on the SIMP method.
It sets up the computational domain, reads the design variables, solves the forward heat conduction problem,
and saves performance (thermal conductivity) metric.
"""

import os
import numpy as np
from fenics import *
from fenics_adjoint import *

# -------------------------------
# Initialization and Parameter Setup
# -------------------------------

# Define base path for shared resources
BASE_PATH = "/home/fenics/shared"
SIM_VAR_PATH = os.path.join(BASE_PATH, "templates", "sim_var.txt")

# Read simulation parameters from file
with open(SIM_VAR_PATH, "r") as file:
    data = file.read().split("\t")

# Extract parameters
NN = int(data[2])-1  # Grid size
vol_f = float(data[0])  # Volume fraction
width = float(data[1])  # Adiabatic boundary width

# Compute step size
step = 1.0 / float(NN)

# Generate x and y coordinate values
x_values = np.linspace(0, 1, num=NN + 1)
y_values = np.linspace(0, 1, num=NN + 1)

# Remove simulation variable file after reading
os.remove(SIM_VAR_PATH)

# -------------------------------
# Load Initial Design Data
# -------------------------------

# Construct filename for input data
input_filename = f"templates/hr_data_v={vol_f}_w={width}_.npy"
input_path = os.path.join(BASE_PATH, input_filename)

# Load initial design image
image = np.load(input_path)
os.remove(input_path)  # Remove after loading

# -------------------------------
# Mesh and Function Space Setup
# -------------------------------

# Create computational mesh
mesh = UnitSquareMesh(NN, NN)

# Map image data to mesh vertices
x = mesh.coordinates().reshape((-1, 2))
h = 1.0 / NN
ii, jj = np.array(x[:, 0] / h, dtype=int), np.array(x[:, 1] / h, dtype=int)

# Extract image values corresponding to mesh vertices
image_values = image[ii, jj]

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
        return x[0] == 0.0 or x[1] == 1.0 or x[0] == 1.0 or (x[1] == 0.0 and (x[0] < lb_2 or x[0] > ub_2))


# Apply boundary condition: Temperature = 0
T_bc = 0.0
bc = [DirichletBC(P, T_bc, BoundaryConditions())]

# Define heat source term
f = interpolate(Constant(1.0e-2), P)  # Default source term

# -------------------------------
# Forward Heat Conduction Simulation
# -------------------------------


def forward(a):
    """Solve the heat conduction PDE given a material distribution 'a'."""
    T = Function(P, name="Temperature")
    v = TestFunction(P)

    # Define variational form
    F = inner(grad(v), k(a) * grad(T)) * dx - f * v * dx

    # Solve PDE
    solve(F == 0, T, bc, solver_parameters={"newton_solver": {"absolute_tolerance": 1.0e-7, "maximum_iterations": 20}})

    return T


# -------------------------------
# Optimization Process
# -------------------------------

# Initialize control variable
a = interpolate(init_guess, A)

# Solve forward problem
T = forward(a)

# Define optimization objective function (cost function)
J = assemble(f * T * dx + alpha * inner(grad(a), grad(a)) * dx)

# Define control object for optimization
m = Control(a)
Jhat = ReducedFunctional(J, m)
J_CONTROL = Control(J)
# Define optimization bounds
lb, ub = 0.0, 1.0


# Volume Constraint
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

# Set optimization solver parameters
solver_params = {"acceptable_tol": 1.0e-3, "maximum_iterations": 0}
solver = IPOPTSolver(problem, parameters=solver_params)

# Solve optimization problem
a_opt = solver.solve()

# -------------------------------
# Store and Save Results
# -------------------------------

# Save optimized design
mesh_output = UnitSquareMesh(NN, NN)
V_output = FunctionSpace(mesh_output, "CG", 1)
sol_output = a_opt

# Save optimized control to XDMF file
output_xdmf = XDMFFile("/home/fenics/shared/templates/RES_SIM/SIM_solution_v={}_w={}.xdmf".format(vol_f, width))
output_xdmf.write(a_opt)

# Save discrete results as numpy array
results = np.zeros(((NN + 1) ** 2, 5))

ind = 0
for xs in x_values:
    for ys in y_values:
        results[ind, 0] = xs
        results[ind, 1] = ys
        results[ind, 2] = vol_f
        results[ind, 3] = width
        results[ind, 4] = a_opt(xs, ys)
        ind += 1

# Save results as numpy file
output_npy = "/home/fenics/shared/templates/RES_SIM/SIM_hr_data_v={}_w={}.npy".format(vol_f, width)
np.save(output_npy, results)

# Save performance metric
with open("/home/fenics/shared/templates/RES_SIM/Performance.txt", "w") as f:
    f.write("%.14f" % J_CONTROL.tape_value())

# Clean up temporary files
os.system("rm /home/fenics/shared/templates/RES_SIM/TEMP*")

print(f"Optimization complete: v={vol_f}, w={width}")
