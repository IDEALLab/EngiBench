# MTO2D

MTO2D is a two-dimensional thermofluid topology-optimization problem for a
fluid-cooled heat sink. It couples steady flow and heat transfer with adjoint
sensitivities and MMA design updates. EngiBench reports mean temperature and
fluid power dissipation as objectives, and both are minimized.

The implementation is split into:

- `v0.py`: EngiBench API and user-facing demo;
- `model/design_io.py`: native NumPy/OpenFOAM field conversion; and
- `model/runner.py`: isolated case preparation and container execution.

The solver runs in a container built outside EngiBench; see Runtime below.

## Design and data

A design is a `float32` `(400, 200)` array in `[0, 1]` containing the
non-redundant half-domain. `gamma=0` is solid, `gamma=1` is fluid, and
`render()` mirrors the array to `(400, 400)`.

The public [IDEALLab/mto_2d_v0](https://huggingface.co/datasets/IDEALLab/mto_2d_v0)
dataset has `train`, `val`, and `test` splits and six columns:

- `optimal_design`: flattened native design with 80,000 `float32` values;
- `inlet_velocity`, `max_power_dissipation`, and `volfrac`; and
- `mean_temperature` and `power_dissipation`.

`MTO2D().dataset` and the demo load that Hub dataset by default. For offline
use, load a saved Hugging Face `DatasetDict` with `datasets.load_from_disk()`
and pass it as `MTO2D(dataset=dataset)`.

## Simulation

`simulate()` evaluates a fixed design once. The runner exports a pristine case
from the image, writes the design and conditions, applies the final physical
parameters (`q=0.01`, `alphaMax=5.0252e6`, `Heaviside=59.8`), disables design
updates, runs OpenFOAM, and returns:

```text
[mean_temperature, power_dissipation]
```

`simulate_verbose()` additionally reports volume and power residuals, elapsed
solver time, and retained artifacts when requested.

```python
from engibench.problems.mto2d import MTO2D

problem = MTO2D()
design, _ = problem.random_design()
objectives = problem.simulate(design)
```

Running the module directly (`python engibench/problems/mto2d/v0.py`) samples
one dataset design, renders it, and re-evaluates it in the published
container.

Solver settings are ordinary `Config` fields, passed as keyword arguments like
every other EngiBench problem (`MTO2D(max_iter=20, mode="warm")`) or per call
(`problem.simulate(design, config={"volfrac": 0.4})`).

Host settings -- which image to run, where to work, how long to allow, what to
retain -- are constructor arguments of `MTO2DRunner`, not part of the benchmark
configuration:

```python
from engibench.problems.mto2d import MTO2D
from engibench.problems.mto2d.model.runner import MTO2DRunner

problem = MTO2D(runner=MTO2DRunner(timeout=3600.0, retain_artifacts=True))
```

`$ENGIBENCH_MTO2D_IMAGE` overrides the pinned `container_id` for the default
runner; see `MTO2D.resolved_container_image()`.

## Optimization

Note that the dataset satisfies the power-dissipation constraint only to within
MMA's convergence tolerance: 3,149 of the 5,666 published rows report
`power_dissipation` slightly above their own `max_power_dissipation`, by at most
0.98%. `power_constraint_residual` is therefore expected to be small and
positive for roughly half of all dataset designs, and is not a defect.

`optimize()` runs the adjoint/MMA loop and returns `(design, history)`, as in
the base `Problem` contract. `optimization_schedule="legacy"` reproduces the
original 200-step cold schedule; `"strict"` supports configurable cold or warm
continuation.

Per-iteration residuals and timings are on `problem.last_solver_run` after the
call, and `problem.active_power_bounds(...)` reproduces the bound MMA actually
saw at each iteration:

```python
design, history = problem.optimize(starting_design)
run = problem.last_solver_run
residuals = run.power_dissipation / problem.active_power_bounds() - 1.0
```

Full optimization is intentionally absent from the shared EngiBench smoke
suite because it takes hours. It remains available through the API for
explicit runs.

## Runtime

The runtime is Linux/AMD64. EngiBench pins its immutable GHCR digest in
`MTO2D.container_id`; Docker can emulate it on ARM hosts at lower speed.
Each run uses a unique working directory and the image-contained case, so no
host OpenFOAM installation is required.

The image recipe, its pinned dependencies and its numerical release gate live
in [`IDEALLab/engibench-mto2d-image`][image-repo]. EngiBench does not build the
image, the same way it does not build `mdolab/public` for `Airfoil` or
`quay.io/dolfinadjoint/pyadjoint` for the heat-conduction problems.

[image-repo]: https://github.com/IDEALLab/engibench-mto2d-image

## References

- Drake et al., “To Quantize or Not to Quantize: Effects on Generative Models
  for Topology Optimization Problems,” *Journal of Mechanical Design* 148(10),
  101704 (2026), [doi:10.1115/1.4071440](https://doi.org/10.1115/1.4071440).
- Yu et al., “Three-dimensional topology optimization of
  thermal-fluid-structural problems for cooling system design,” *Structural
  and Multidisciplinary Optimization* 62, 3347–3366 (2020).
- Svanberg, “The method of moving asymptotes,” *International Journal for
  Numerical Methods in Engineering* 24(2), 359–373 (1987).
