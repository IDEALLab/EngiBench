# Beams3D

This is a 3D structural topology optimization problem for minimizing compliance
subject to boundary conditions and a volume fraction constraint.


## Design space
This structural topology optimization problem is governed by linear elasticity.
The problem is defined over a cubic 3D domain, where load nodes and support
nodes are placed along the boundary to define a unique elastic condition. The
design is a density array of shape `(nely, nelx, nelz)`, with axis order
`[y, x, z]` and values in `[0, 1]`. The names `nelx`, `nely`, and `nelz` still
refer to the physical x, y, and z element counts.

The default v0 problem uses a `16 x 16 x 16` element grid. Matching datasets are
also available at `32 x 32 x 32` and `64 x 64 x 64` resolution.

## Objectives
The public objective is `c`, the structural compliance of the design under the
applied load. Lower compliance corresponds to a stiffer structure. The target
volume fraction is handled as an optimization constraint rather than a dataset
objective, following the Beams2D convention.

## Simulator
The simulation code is based on density-based structural topology optimization.
Optimization is conducted by reformulating the discrete material-placement
problem as a continuous one using SIMP-style material interpolation. A
density/sensitivity filter is used to reduce checkerboard-like artifacts, and
the design variables are updated using the method of moving asymptotes.

The FEM solver uses Hex8 structural elements, sparse assembly, and an
AMG-preconditioned conjugate-gradient linear solve. Iteration-invariant
structural FEM data are cached once per optimization run, and the 3D cone filter
is applied in matrix-free form instead of explicitly assembling the full filter
matrix.

Each active `force_elements_z` node receives a fixed nondimensional nodal load
of magnitude `0.5` in the positive z-direction. Reported compliance values scale
with this constant.

Rendered designs use a simple Matplotlib 3D voxel plot. Pass `open_window=True`
to display it in a window.

## Conditions
Problem conditions are defined by creating a Python dict with the following
fields:

- `volfrac`: Encodes the target volume fraction for the volume fraction
  constraint.
- `rmin`: Encodes the filter size used in the optimization routine.
- `forcedist_x`: Fractional x-position of the vertical load on the top face.
- `forcedist_y`: Fractional y-position of the vertical load on the top face.

Boundary-condition masks are indexed `[x, y, z]`, which differs from the design
array's `[y, x, z]` axis order.

Additional config and solver parameters are:

- `fixed_elements`: A binary node mask of shape
  `(nelx + 1, nely + 1, nelz + 1)` marking structurally fixed nodes.
- `force_elements_z`: A binary node mask of shape
  `(nelx + 1, nely + 1, nelz + 1)` marking nodes with the exposed vertical
  z-direction load. If omitted, it is generated from `forcedist_x` and
  `forcedist_y`. `force_elements_x` and `force_elements_y` are not public
  Beams3D config fields.
- `penal`: Encodes the SIMP penalization parameter.
- `nelx`: Number of elements in the x-direction.
- `nely`: Number of elements in the y-direction.
- `nelz`: Number of elements in the z-direction.
- `max_iter`: Maximum number of optimization iterations.

Following the Beams2D convention, custom grids should be created by passing
`nelx`, `nely`, and `nelz` to the `Beams3D(config={...})` constructor. That
keeps the instance design space, default masks, and dataset id consistent.

## Dataset
Beams3D datasets are available on the Hugging Face Datasets Hub:

- `16 x 16 x 16`: https://huggingface.co/datasets/IDEALLab/beams_3d_16_v0
- `32 x 32 x 32`: https://huggingface.co/datasets/IDEALLab/beams_3d_32_v0
- `64 x 64 x 64`: https://huggingface.co/datasets/IDEALLab/beams_3d_64_v0

Dataset access is implemented only for these cubic resolutions. Non-cubic
configurations may still be used for local FEM smoke tests and optimization,
but `dataset` and `random_design()` require `nelx == nely == nelz` with a
resolution of `16`, `32`, or `64`.

Relevant datapoint fields in the Hugging Face datasets include:

- `optimal_design`: An optimized design for the set of boundary conditions. Hugging Face rows may store this as a flat vector; `random_design()` reshapes it to `(nely, nelx, nelz)`.
- `volfrac`: The target volume fraction.
- `rmin`: The filter size used in the optimization routine.
- `forcedist_x`: The fractional x-position of the vertical load on the top face.
- `forcedist_y`: The fractional y-position of the vertical load on the top face.
- `c`: The structural compliance of the optimized design.
- `optimization_history`: Objective values recorded during optimization.

The simulator still uses explicit node masks for boundary conditions. Dataset
examples therefore reconstruct `force_elements_z` from `forcedist_x` and
`forcedist_y`, and use the problem's default fixed-support mask.

## Relation to Thermoelastic3D
Beams3D intentionally follows the structure of Thermoelastic3D while removing
the thermal domain. This makes it easier to review, maintain, and compare with
the existing 3D problem implementation.

### File layout

| File | Beams3D | Thermoelastic3D | Main difference |
| --- | --- | --- | --- |
| `README.md` | yes | yes | Same documentation role. |
| `__init__.py` | yes | yes | Exports the problem class. |
| `v0.py` | yes | yes | Beams3D exposes structural-only conditions and the Beams2D-style `c` objective; Thermoelastic3D exposes structural, thermal, and weighting conditions. |
| `model/__init__.py` | yes | yes | Package marker. |
| `model/fem_matrix_builder.py` | yes | yes | Beams3D builds only structural Hex8 stiffness; Thermoelastic3D builds structural, thermal, and coupling matrices. |
| `model/fem_model.py` | yes | yes | Beams3D runs structural-only optimization; Thermoelastic3D runs coupled thermoelastic optimization. |
| `model/fem_plotting.py` | yes | yes | Beams3D plots density and mechanical BCs; Thermoelastic3D also plots heatsink information. |
| `model/fem_setup.py` | compatibility exports | full coupled setup | Beams3D keeps a compatibility export and moves structural assembly to `fem_structural_setup.py`. |
| `model/fem_structural_setup.py` | yes | no | Beams3D-specific structural-only cached setup and solve path. |
| `model/linear_solver.py` | yes | yes | Same AMG-preconditioned sparse-solver role. |
| `model/mma_subroutine.py` | yes | yes | Same MMA wrapper role with Beams3D-specific defaults. |
| `model/opti_dataclass.py` | no | yes | Thermoelastic3D records richer gradient/update data; Beams3D currently uses EngiBench `OptiStep` directly. |

### Physics and data
Thermoelastic3D is a coupled multi-physics problem with structural loads,
supports, heatsink elements, thermal compliance, structural compliance, and a
`weight` parameter controlling the structural/thermal objective mix. Beams3D is
structural-only: it has fixed supports, a vertical structural load, structural
compliance, and a volume-fraction constraint.

Thermoelastic3D currently documents one `16 x 16 x 16` dataset. Beams3D follows
the same default resolution in the built-in problem class and also documents
`32 x 32 x 32` and `64 x 64 x 64` dataset variants.

### Optimization and FEM
Both problems use density-based topology optimization, SIMP-style interpolation,
3D FEM, sensitivity filtering, sparse linear solves, and MMA updates. Beams3D
removes all thermal assembly and thermal adjoint work, caches structural FEM
bookkeeping once per run, and applies the sensitivity filter in matrix-free form
to avoid materializing the full 3D filter matrix.

## Citation
This problem follows density-based structural topology optimization methods for
3D compliance minimization. If you use this problem in your experiments, cite:

```
@article{bendsoe1999material,
  title={Material interpolation schemes in topology optimization},
  author={Bends{\o}e, Martin P. and Sigmund, Ole},
  journal={Archive of Applied Mechanics},
  volume={69},
  number={9--10},
  pages={635--654},
  year={1999},
  publisher={Springer},
  doi={10.1007/s004190050248}
}

@article{liu2014efficient,
  title={An efficient 3D topology optimization code written in Matlab},
  author={Liu, Kai and Tovar, Andr{\'e}s},
  journal={Structural and Multidisciplinary Optimization},
  volume={50},
  number={6},
  pages={1175--1196},
  year={2014},
  publisher={Springer},
  doi={10.1007/s00158-014-1107-x}
}

@article{svanberg1987method,
  title={The method of moving asymptotes -- a new method for structural optimization},
  author={Svanberg, Krister},
  journal={International Journal for Numerical Methods in Engineering},
  volume={24},
  number={2},
  pages={359--373},
  year={1987},
  publisher={Wiley},
  doi={10.1002/nme.1620240207}
}
```
