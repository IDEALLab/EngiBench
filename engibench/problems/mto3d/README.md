# MTO3D

MTO3D is a three-dimensional thermofluid topology-optimization problem for a liquid-cooled heat
sink. It couples steady incompressible flow (Brinkman penalization) and heat transfer with adjoint
sensitivities and MMA design updates. EngiBench reports mean temperature and fluid power
dissipation as objectives; both are minimized.

The implementation is split into:

- `v0.py`: EngiBench API and user-facing demo;
- `model/design_io.py`: native NumPy/OpenFOAM `xh` field conversion; and
- `model/runner.py`: isolated case preparation and container execution.

## Design and data

A design is a `float32` `(20, 100, 100)` array in `[0, 1]` holding the non-redundant half-domain in
`(y, z, x)` order: y increases between the solid shells, z decreases (index 0 is the inlet side),
and x increases towards the symmetry plane at the last index. `xh=0` is solid and `xh=1` is fluid.
`render()` mirrors the array to `(20, 100, 200)` and draws it in 3D, coloring fluid cells by depth from
the heated wall.

The native OpenFOAM field has 264,000 equal-volume cells: 200,000 design cells in three bodies of
three blocks each, 40,000 fixed solid (shells) and 24,000 fixed fluid (inlet/outlet) cells copied
unchanged from the image's final-time template. `design_io` owns this mapping, so simulation never
reshapes, resizes or interpolates a design.

The dataset `IDEALLab/mto_3d_v0` holds 4,977 unique optimized designs, split 3,733/249/995 for
train/val/test, in the format of MTO2D's dataset: the same six columns in the same order and with the
same types (`optimal_design` as a flat `float32` list in C order, the three conditions as `float64` and
both objectives as `float32`). The objectives are the `simulate()` results for each stored design, with
that row's own conditions. Designs that violate their power-dissipation bound are kept. One stored
design (case 3905) is left out because its flow never meets the convergence test. The docs page lists
these and the other known issues.

## Simulation

`simulate()` evaluates a fixed design: the runner exports a pristine case from the image, writes the
design and conditions, selects final physics (`qU=0.001`, `alphaMax=2.5e6`), disables design updates
(`updateDesign no`) and runs 100 solver iterations with the design fixed. The first iteration solves
the flow from rest until both initial residuals reach 1e-6. Every iteration re-assembles and
re-solves the energy equation (tolerance 1e-10): OpenFOAM applies its second-order convection term
(`linearUpwind`) as an explicit correction computed from the current temperature, so a single solve
would return the first-order upwind temperature (mean temperature up to 16% low) and depend on the
initial field. The correction converges geometrically, and the result does not depend on the initial
temperature. `simulate()` returns `[mean_temperature, power_dissipation]` of the last iteration at
full precision. A flow solve that does not converge within `frozen_ns_max_iter` SIMPLE passes, or a
mean temperature still changing over the final iterations, raises an error, and the runner verifies
that the design came back unchanged. The solver stops as soon as an iteration uses all its passes
without converging the flow, so a non-converging design fails after one capped flow solve rather than
100.

By default the inlet velocity is rounded to 3 significant figures, which is what the published
campaigns effectively used (`decomposePar` wrote it at `writePrecision 3`). Pass
`inlet_significant_figures=None` for the nominal value.

## Optimization

`optimize()` runs the adjoint/MMA loop and returns `(design, history)`. `mode="cold"` uses the
legacy 20-level continuation, which stops by itself after at most 1,920 iterations; starting from
`uniform_starting_design(volfrac)` it reproduces the published initialization exactly (the runner
then supplies no `0/xh`). `mode="warm"` applies final physics from the first iteration. It is not an
exact restart of a finished run, because a stored design carries no optimizer state: the first warm
step re-filters and re-projects the design. History row
`k` describes the design evaluated in iteration `k`; the returned design is the subsequent MMA
update. Full optimization takes hours on 16 ranks (about 5 h for case 0 on EPYC 7763) and is outside
the shared smoke suite. It is not bitwise reproducible against the published campaigns. For case 0 it
matches the legacy trajectory for 4 iterations and then drifts slowly. The final design differs in
about 1% of its cells, and its `simulate()` objectives are within 0.09% (mean temperature) and 0.25%
(power dissipation) of the published design's. History objectives come from one energy solve per
iteration with the temperature carried over, not from the converged evaluation `simulate()` performs.

## Runtime

The runtime is Linux/AMD64, 16 MPI ranks by default (`simple (2 2 4)`, the decomposition of every
published case). On machines with fewer cores the 16 ranks are oversubscribed: slower, but with the
same results, which depend only on the rank count. Measured times:

- 16-core EPYC 7763 node: about 45 s per `simulate()` and about 10 s per `optimize()` iteration.
- 8-core Apple-silicon laptop (Docker Desktop with Rosetta): about 20 min per `simulate()` and
  about 6 min per `optimize()` iteration, so full optimizations need a Linux machine with 16 or more
  cores.

The image recipe, its pinned dependencies, the EngiBench solver patches and the numerical release
gate live in a separate image repository, `IDEALLab/engibench-mto3d-image`. The image derives from the
pinned MTO2D runtime. `MTO3D.container_id` pins the published image
`ghcr.io/ideallab/engibench-mto3d` by manifest digest.

With Apptainer, EngiBench runs the SIF named after the image in the working directory
(`Apptainer.sif_filename`) and pulls it only when that file is absent. On clusters whose compute
nodes have no internet access, place the SIF there beforehand.

## References

- Yu et al., "Three-dimensional topology optimization of thermal-fluid-structural problems for
  cooling system design," *Structural and Multidisciplinary Optimization* 62, 3347-3366 (2020).
- Svanberg, "The method of moving asymptotes," *International Journal for Numerical Methods in
  Engineering* 24(2), 359-373 (1987).
