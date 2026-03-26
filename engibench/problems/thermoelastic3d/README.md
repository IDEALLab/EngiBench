# ThermoElastic3D

**Lead**: Gabriel Apaza @gapaza

This is 3D topology optimization problem for minimizing weakly coupled thermo-elastic compliance subject to boundary conditions and a volume fraction constraint.


## Design space
This multi-physics topology optimization problem is governed by linear elasticity and steady-state heat conduction with a one-way coupling from the thermal domain to the elastic domain.
The problem is defined over a cube 3D domain, where load elements and support elements are placed along the boundary to define a unique elastic condition.
Similarly, heatsink elements are placed along the boundary to define a unique thermal condition.
The design space is then defined by a 3D array representing density values (parameterized by DesignSpace = [0,1]^{nelx x nely x nelz}, where nelx, nely, and nelz denote the x, y, and z dimensions respectively).

## Objectives
The objective of this problem is to minimize total compliance C under a volume fraction constraint V by placing a thermally conductive material.
Total compliance is defined as the sum of thermal compliance and structural compliance.

## Simulator
The simulation code is based on a Python adaptation of the popular 88-line topology optimization code, modified to handle the thermal domain in addition to thermal-elastic coupling.
Optimization is conducted by reformulating the integer optimization problem as a continuous one (leveraging a SIMP approach), where a density filtering approach is used to prevent checkerboard-like artifacts.
The optimization process itself operates by calculating the sensitivities of the design variables with respect to total compliance (done efficiently using the Adjoint method), calculating the sensitivities of the design variables with respect to the constraint value, and then updating the design variables by solving a convex-linear subproblem and taking a small step (using the method of moving asymptotes).
The optimization loop terminates when either an upper bound of the number of iterations has been reached or if the magnitude of the gradient update is below some threshold.

## Conditions
Problem conditions are defined by creating a python dict with the following info:
- `fixed_elements`: Encodes a binary NxNxN matrix of the structurally fixed elements in the domain.
- `force_elements_x`: Encodes a binary NxNxN matrix specifying elements that have a structural load in the x-direction.
- `force_elements_y`: Encodes a binary NxNxN matrix specifying elements that have a structural load in the y-direction.
- `force_elements_z`: Encodes a binary NxNxN matrix specifying elements that have a structural load in the z-direction.
- `heatsink_elements`: Encodes a binary NxNxN matrix specifying elements that have a heat sink.
- `volfrac`: Encodes the target volume fraction for the volume fraction constraint.
- `rmin`: Encodes the filter size used in the optimization routine.
- `weight`: Allows one to control which objective is optimized for. 1.0 Is pure structural optimization, while 0.0 is pure thermal optimization.
