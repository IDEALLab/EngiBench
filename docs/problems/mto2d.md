# MTO2D

```{problem:table}
```

## Release status

The native dataset is public at
[`IDEALLab/mto_2d_v0`](https://huggingface.co/datasets/IDEALLab/mto_2d_v0).
The source-built Linux/AMD64 solver is hosted at
`ghcr.io/arthurdrake1/engibench-mto2d`; `MTO2D.container_id` pins the accepted
manifest by immutable digest.

## Motivation

Multiphysics topology optimization (MTO) couples fluid flow, heat transfer, and
design sensitivities in a single optimization loop, making it one of the most
expensive problem families in density-based topology optimization. MTO2D
distributes fluid and solid material inside a two-dimensional heat sink so that
coolant entering at the top carries heat out at the bottom. Each cold-start
optimization solves the incompressible Navier–Stokes and energy-balance
equations with adjoint sensitivities for 200 iterations, which is why this
problem is a strong benchmark for surrogate and generative approaches that try
to warm-start or replace the solver.

The problem, solver, and dataset follow the VQGAN-for-MTO study by Drake et
al., which generated thousands of optimized heat sinks over a range of inlet
velocities, power-dissipation limits, and fluid volume fractions.

## Design space

A design is a `float32` array with shape `(400, 200)` and values in `[0, 1]`,
storing the non-redundant left half of the symmetric design domain:

- `gamma = 0` means solid material.
- `gamma = 1` means fluid.
- `render()` mirrors the half-domain horizontally into a symmetric
  `(400, 400)` image.

The underlying OpenFOAM field has 86,400 cells: the first 80,000 encode the
`(400, 200)` design in two solver-specific mesh blocks, and the remaining
6,400 are fixed fluid cells (inlet and outlet channels) copied unchanged from
the case template. The `design_io` module owns this mapping so that simulation
never silently resizes a design.

Datasets store `optimal_design` as a flat `list<float32>` of length 80,000 in
C order, following the Beams3D Hub convention; `random_design()` reshapes rows
back to the native `(400, 200)` representation.

The earlier
[`IDEALLab/MTO-2D`](https://huggingface.co/datasets/IDEALLab/MTO-2D) NumPy
dataset stores each design as a `(256, 256)` image of the whole left
half-domain, produced by a lossy anisotropic bicubic resize of the native
field. MTO2D v0 deliberately accepts only the native representation so
simulation never silently resizes a topology.

## Objectives

Objective arrays always use this order:

0. `mean_temperature`: mean temperature over the domain, to minimize.
1. `power_dissipation`: normalized fluid power dissipation, to minimize.

The native optimization formulation minimizes mean temperature only; power
dissipation and fluid volume are solver constraints. Power dissipation is also
reported as a second EngiBench objective so that designs can be compared in
Pareto studies. Because the power constraint depends on the simulation output,
it cannot be checked from inputs alone: `simulate_verbose()` reports its
relative residual as `power_dissipation / max_power_dissipation - 1` together
with the solver-reported volume residual.

## Conditions

```{problem:conditions}
:defaults:
```

- `inlet_velocity`: signed inlet velocity, nominally from `-0.095` to
  `-0.025` m/s, corresponding to Reynolds numbers from 50 to 190.
- `max_power_dissipation`: dimensionless normalized power-dissipation limit,
  nominally from `50` to `75`. The solver normalizes by
  `D_normalization = 1.57572e-7`; the paper calls the rounded `J1 ≈ 1.58e-7`
  reference scale.
- `volfrac`: maximum all-cell fluid fraction, nominally from `0.25`
  to `0.70`.

## Simulator

The solver is an adjoint-based OpenFOAM 5 application: steady incompressible
Navier–Stokes and energy-conservation equations with a Brinkman penalization
for solid regions, RAMP material interpolation, a distance-based density
filter with Heaviside projection, and the method of moving asymptotes (MMA)
for design updates. It runs in an external containerized runtime and supports
MPI decomposition.

`simulate()` performs one frozen final-physics evaluation: the design is
written into a pristine copy of the case, design updates are disabled, and the
final continuation parameters (`qu = 0.01`, `alphaMax = 5.0252e6`,
`Heaviside = 59.8`, movement limit `0`) are applied so the reported objectives
belong to the input design exactly.

`optimize()` runs the full adjoint/MMA loop. The default
`optimization_schedule="legacy"` reproduces the source optimizer's exact
continuation timing used to generate the published dataset; an alternative
`"strict"` schedule exposes configurable interpolation profiles and endpoints.
Cold starts run 200 iterations by default; warm starts from a good initial
design commonly use 20. The retained legacy optimization history for the case
with `inlet_velocity = -0.074`, `max_power_dissipation = 63.1`, and
`volfrac = 0.61` reaches `mean_temperature = 9.45825` and
`power_dissipation = 62.2588`. Those historical values describe the
pre-update optimizer field, not the committed frozen-simulation reference.

`optimize_verbose()` is a deliberate MTO2D-specific extension rather than
part of the base `Problem` contract. Standard `optimize()` still returns
`(design, history)`; the extension additionally exposes constraint residuals,
active power bounds, elapsed times, and retained artifacts.

See the package `README.md` in `engibench/problems/mto2d/` for solver
backends, runtime-image preparation, and usage.

## Dataset

Rows contain the flat `optimal_design`, the three condition fields, and both
objective fields. The 4,249/283/1,134 train/validation/test splits follow the
source paper's 75/5/20 policy.

## Citation

If you use this problem or dataset, please cite the source study:

```bibtex
@article{drake2026quantize,
  title={To Quantize or Not to Quantize: Effects on Generative Models for Topology Optimization Problems},
  author={Drake, Arthur and Chen, Qiuyi and Wang, Jun and Nejat, Ardalan and Guest, James K. and Fuge, Mark},
  journal={Journal of Mechanical Design},
  year={2026},
  volume={148},
  number={10},
  pages={101704},
  doi={10.1115/1.4071440}
}

@article{yu2020three,
  title={Three-dimensional topology optimization of thermal-fluid-structural problems for cooling system design},
  author={Yu, M. and Ruan, S. and Gu, J. and Ren, M. and Li, Z. and Wang, X. and Shen, C.},
  journal={Structural and Multidisciplinary Optimization},
  volume={62},
  number={6},
  pages={3347--3366},
  year={2020},
  publisher={Springer}
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
