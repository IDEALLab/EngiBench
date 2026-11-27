# MTO3D

``` {problem:table}
:lead: Arthur Drake @arthurdrake1
```

## Motivation

MTO3D is the three-dimensional counterpart of [MTO2D](mto2d.md): multiphysics topology
optimization of a liquid-cooled heat sink, coupling steady incompressible flow, heat transfer and
adjoint sensitivities in one optimization loop. A cold-start optimization runs up to 1,920 MMA
iterations on a 264,000-cell mesh and takes several hours on 16 cores, which makes the problem a
demanding benchmark for surrogate, generative and warm-start methods.

The problem, solver, and dataset follow the 3D heat-sink study of Chen (2024), cited below, which
optimized 5,000 heat sinks over a range of inlet velocities, power-dissipation bounds, and fluid volume
fractions.

## Problem setup

The model is one symmetric half of a liquid-cooled heat sink. The full plate is 10 mm wide (x), 5 mm
long (z) and 1.2 mm thick (y); its 1 mm design layer, meshed as three stacked design bodies, lies
between two 0.1 mm solid shells. Coolant (water) enters through a 2 mm by 1 mm channel at the top
(`z = 5` to `6.5 mm`), flows along `-z` through the channels placed by the optimizer, and leaves
through an identical channel at the bottom (`z = -1.5` to `0 mm`). The solid is aluminium.

The inlet has a fixed velocity and reference temperature (`T = 0`). The outlet has zero pressure and
no normal temperature gradient. The heated wall (`force` patch, the shell face at `y = 1.2 mm`)
carries a uniform heat flux, imposed as a fixed temperature gradient. The opposite face and the
remaining walls are no-slip and adiabatic, and the plane `x = 0` is a symmetry boundary. For every
material layout, the solver computes steady incompressible flow and heat transfer.

## Design space

A design is a `float32` array with shape `(20, 100, 100)` and values in `[0, 1]`, in `(y, z, x)`
order:

- `xh = 0` means solid material and `xh = 1` means fluid;
- y (20 cells) increases between the solid shells, z (100 cells) decreases so that index 0 is the
  inlet side, and x (100 cells) increases towards the symmetry plane at the last index;
- `render()` mirrors the half-domain across the symmetry plane into `(20, 100, 200)` and draws it
  in 3D: its fluid cells, colored by depth into the page, with the heated wall facing the viewer.

```{figure} ../_static/img/problems/mto3d.png
:alt: 3D view of the fluid channels of an MTO3D design, colored by depth from the heated wall
:name: mto3d-example-design
:width: 700px
:align: center
```

The example above is the `render()` output for public dataset row `train[2888]` (`volfrac = 0.29`),
selected for its clean channel network. Its central trunk lies next to the heated wall, and its side
arcs run through the middle of the thickness. Like most designs in the dataset, it is mostly solid
next to both faces, with its channels in the middle layers.

The OpenFOAM field has 264,000 equal-volume cells: 200,000 design cells, 40,000 fixed solid shell
cells and 24,000 fixed fluid inlet/outlet cells, copied unchanged from the case template. The
`design_io` module owns this mapping; it was proven against the mesh's own cell zones and cell
centres, so simulation never reshapes or resizes a design.

Datasets store `optimal_design` as a flat `list<float32>` of length 200,000 in C order, as for MTO2D;
`random_design()` reshapes rows back to the native `(20, 100, 100)` representation.

## Objectives

EngiBench minimizes both reported objectives, in this order:

0. `mean_temperature`: mean temperature of the heated wall in the solver's normalization (sum of
   face temperatures divided by the legacy constant 128, `numberOfPassiveElements`).
1. `power_dissipation`: fluid power dissipation through inlet and outlet, divided by `1e-9`.

The native MMA formulation minimizes mean temperature subject to the fluid-volume and
power-dissipation constraints. EngiBench also exposes the measured power dissipation as a second
objective so designs can be compared on a Pareto front. The condition `max_power_dissipation` is the
input bound, not the measured objective. Because that constraint depends on simulation output,
`simulate_verbose()` reports its relative residual as `power_dissipation / max_power_dissipation - 1`.

## Conditions

```{problem:conditions}
:defaults:
```

- `inlet_velocity`: signed inlet z-velocity in m/s, from `-0.170` to `-0.100` in the dataset. The
  published campaigns effectively used this value rounded to 3 significant figures, which the
  runner reproduces by default.
- `max_power_dissipation`: bound on the normalized power dissipation, from `5` to `25`.
- `volfrac`: maximum all-cell fluid fraction, from `0.25` to `0.55`.

## Simulator

The solver is an adjoint-based OpenFOAM 5 application (Yu et al. 2020) with Brinkman
penalization, RAMP interpolation, a PDE density filter with Heaviside projection, and MMA. It runs
in an external Linux/AMD64 container on 16 MPI ranks by default.

`simulate()` performs a frozen final-physics evaluation: design updates are disabled, the flow is
solved from rest until both initial residuals reach `1e-6`, and the energy equation is re-assembled
and re-solved for 100 iterations until its second-order (`linearUpwind`) convection correction,
which OpenFOAM applies explicitly from the current temperature, has converged. A single energy solve
from the uniform initial field would give the first-order upwind temperature, up to 16% low. The
design is verified to be unchanged. `optimize()` runs the native continuation (20 levels, adaptive
stopping, at most 1,920 iterations) or, with `mode="warm"`, final physics from the first iteration.
Because of adaptive stopping, a cold optimization can return fewer history entries than `max_iter`.
For the exact historical initialization, start a cold optimization from
`uniform_starting_design(volfrac)`.

`optimize()` returns `(design, history)` exactly as the base `Problem` contract specifies. Each
history entry includes the constraint residuals and the elapsed solver time. Retained artifacts remain
available through `problem.last_solver_run`.

See the package `README.md` in `engibench/problems/mto3d/` for runner settings, the runtime image,
and usage.

## Dataset

The dataset is hosted at
[`IDEALLab/mto_3d_v0`](https://huggingface.co/datasets/IDEALLab/mto_3d_v0). It holds 4,977 unique
optimized designs over the condition ranges above, split 3,733/249/995 into train/val/test (75/5/20,
as for MTO2D). It follows the format of MTO2D's dataset: the same six columns, in the same order and
with the same types. `optimal_design` is a flat `float32` list of 200,000 values in
C order; the three conditions are `float64` and both objectives `float32`.

The objectives are what `simulate()` returns for each stored design with that row's own conditions,
stored as `float32`. The evaluation behind them is bitwise reproducible across toolchains and CPU types
(AMD EPYC 7763, Intel Xeon 8592+). The labels the original optimizer logged describe the design one MMA
update earlier, with the temperature carried over from earlier iterations. They differ from the new
labels by a median of 0.006% for mean temperature and 0.003% for power dissipation (at most 4.9% and
2.6%).

Known issues:

- 367 designs exceed their power-dissipation bound, 158 of them by more than 5% (at most 184%).
  86 exceed their fluid-volume bound, by at most 5.8e-5. All are kept. The 4,610 designs that satisfy
  both constraints are those with `power_dissipation <= max_power_dissipation` and
  `MTO3D.design_volume_residual(design, volfrac) <= 0`.
- The original 5,000-case collection held 22 pairs of byte-identical designs. One design of each pair
  was dropped.
- One design (case 3905 of the original collection) is excluded: its flow never meets the convergence
  test from rest (the pressure residual stalls at about 2e-6), so it has no valid label.
- The campaigns used the inlet velocity rounded to 3 significant figures. `inlet_velocity` holds the
  nominal value, and `simulate()` rounds it the same way by default.
- 1,201 optimizations stopped before 1,920 iterations, by the optimizer's own convergence criterion.

```{figure} ../_static/img/problems/mto3d_pareto.png
:alt: Pareto front of mean temperature and fluid power dissipation for the MTO3D dataset
:name: mto3d-pareto-front
:width: 700px
:align: center

The global dataset front contains 81 Pareto-optimal designs across all three splits and operating
conditions; all 81 satisfy both of their constraints. Both axes are minimized and in the solver's
normalizations.
```

## Citation

If you use this problem or dataset, please cite the source study:

```bibtex
@phdthesis{chen2024analyzing,
  title={Analyzing Inverse Design Problems from a Topological Perspective},
  author={Chen, Qiuyi},
  school={University of Maryland, College Park},
  year={2024},
  doi={10.13016/k440-nqcc}
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
