#!/usr/bin/env python3

"""This script evaluates the design using finite element analysis with dolfin-adjoint based on the SIMP method.

It sets up the computational domain, reads the design variables, solves the forward heat conduction problem,
and saves performance (thermal conductivity) metric.
"""

import glob
import os

from fenics import dof_to_vertex_map
from fenics import dx
from fenics import FunctionSpace
from fenics import grad
from fenics import inner
from fenics import MPI
from fenics import parameters
from fenics import SubDomain
from fenics import TestFunction
from fenics import XDMFFile
from fenics_adjoint import assemble
from fenics_adjoint import Constant
from fenics_adjoint import Control
from fenics_adjoint import DirichletBC
from fenics_adjoint import Function
from fenics_adjoint import InequalityConstraint
from fenics_adjoint import interpolate
from fenics_adjoint import IPOPTSolver
from fenics_adjoint import MinimizationProblem
from fenics_adjoint import ReducedFunctional
from fenics_adjoint import solve
from fenics_adjoint import UnitCubeMesh
import numpy as np
from pyadjoint.reduced_functional_numpy import set_local

from engibench.utils.cli import cast_argv
from engibench.utils.cli import np_array_from_stdin

# Extract parameters
# NN: Grid size
# vol_f: Volume fraction
# width: Adiabatic boundary width
NN, vol_f, width, output_path = cast_argv(int, float, float, str)
# Load Initial Design Data
image = np_array_from_stdin()

output_dir = os.path.dirname(output_path)

# Compute step size
step = 1.0 / float(NN)

# Generate x, y , and z coordinate values
x_values = np.linspace(0, 1, num=NN + 1)
y_values = np.linspace(0, 1, num=NN + 1)
z_values = np.linspace(0, 1, num=NN + 1)

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

    def inside(self, x, _on_boundary):
        """True if in the interior of the domain."""
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
    # ruff: noqa: N806
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

    # ruff: noqa: N803
    def __init__(self, V):
        self.V = float(V)
        self.smass = assemble(TestFunction(A) * Constant(1) * dx)
        self.tmpvec = Function(A)

    def function(self, m):
        """Compute volume constraint value."""
        set_local(self.tmpvec, m)
        integral = self.smass.inner(self.tmpvec.vector())
        return [self.V - integral] if MPI.rank(MPI.comm_world) == 0 else []

    def jacobian(self, _m):
        """Compute Jacobian of volume constraint."""
        return [-self.smass]

    def output_workspace(self):
        """Return an object like the output of c(m) for calculations."""
        return [0.0]

    def length(self):
        """Return number of constraint components (1)."""
        return 1


# Define optimization problem
problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=VolumeConstraint(vol_f))

# Set optimization solver parameters
solver_params = {"acceptable_tol": 1.0e-100, "maximum_iterations": 0}
solver = IPOPTSolver(problem, parameters=solver_params)

# Solve optimization problem
a_opt = solver.solve()

# -------------------------------
# Store and Save Results
# -------------------------------

# Save optimized design
mesh_output = UnitCubeMesh(NN, NN, NN)
V_output = FunctionSpace(mesh_output, "CG", 1)
sol_output = a_opt

# Save optimized control to XDMF file
output_xdmf = XDMFFile(os.path.join(output_dir, f"RES_SIM/SIM_solution_v={vol_f}_w={width}.xdmf"))
output_xdmf.write(a_opt)

# Save discrete results as numpy array
results = np.zeros(((NN + 1) ** 3, 6))

ind = 0
for xs in x_values:
    for ys in y_values:
        for zs in z_values:
            results[ind, 0] = xs
            results[ind, 1] = ys
            results[ind, 2] = zs
            results[ind, 3] = vol_f
            results[ind, 4] = width
            results[ind, 5] = a_opt(xs, ys, zs)
            ind += 1

# Save results as numpy file
output_npy = os.path.join(output_dir, f"RES_SIM/SIM_hr_data_v={vol_f}_w={width}.npy")
np.save(output_npy, results)

# Save performance metric
with open(output_path, "w") as f:
    f.write(f"{J_CONTROL.tape_value():.14f}")

# Clean up temporary files
for f in glob.glob("/home/fenics/shared/templates/RES_SIM/TEMP*"):
    os.remove(f)

print(f"Optimization complete: v={vol_f}, w={width}")
