# MTO2D

MTO2D is a two-dimensional thermofluid topology-optimization problem for a
fluid-cooled heat sink. It couples steady flow and heat transfer with adjoint
sensitivities and MMA design updates.

The implementation is split into:

- `v0.py`: EngiBench API and user-facing demo;
- `model/design_io.py`: native NumPy/OpenFOAM field conversion; and
- `model/runner.py`: isolated container and command execution.

The pinned source-image recipe and release reference live outside the Python
package in `docker/mto2d/` at the repository root.

## Design and data

A design is a `float32` `(400, 200)` array in `[0, 1]` containing the
non-redundant half-domain. `gamma=0` is solid, `gamma=1` is fluid, and
`render()` mirrors the array to `(400, 400)`.

The public [IDEALLab/mto_2d_v0](https://huggingface.co/datasets/IDEALLab/mto_2d_v0)
dataset has `train`, `val`, and `test` splits and six columns:

- `optimal_design`: flattened native design with 80,000 `float32` values;
- `inlet_velocity`, `max_power_dissipation`, and `volfrac`; and
- `mean_temperature` and `power_dissipation`.

`MTO2D().dataset` and the demo load that Hub dataset by default. A saved
Hugging Face `DatasetDict` can be selected explicitly with `--dataset`.

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

The command-line demo uses the published dataset and container:

```bash
python engibench/problems/mto2d/v0.py --simulate --no-show
```

Set `ENGIBENCH_MTO2D_IMAGE` to override the image. Advanced solver settings
can be supplied as JSON through `--solver-config` or
`ENGIBENCH_MTO2D_SOLVER_CONFIG`.

## Optimization

`optimize()` runs the adjoint/MMA loop and returns `(design, history)`.
`optimization_schedule="legacy"` reproduces the original 200-step cold
schedule; `"strict"` supports configurable cold or warm continuation.
`optimize_verbose()` also returns residual and timing histories.

Full optimization is intentionally absent from the shared EngiBench smoke
suite because it takes hours. It remains available through the API for
explicit runs.

## Runtime

The runtime is Linux/AMD64. EngiBench pins its immutable GHCR digest in
`MTO2D.container_id`; Docker can emulate it on ARM hosts at lower speed.
Each run uses a unique working directory and the image-contained case, so no
host OpenFOAM installation is required.

The maintainer build is source-pinned and its one-step reference must match
the committed scalar histories and final gamma bytes exactly. Build and
release instructions are in `docker/mto2d/README.md` at the repository root.

## References

- Drake et al., “To Quantize or Not to Quantize: Effects on Generative Models
  for Topology Optimization Problems,” *Journal of Mechanical Design* 148(10),
  101704 (2026), [doi:10.1115/1.4071440](https://doi.org/10.1115/1.4071440).
- Yu et al., “Three-dimensional topology optimization of
  thermal-fluid-structural problems for cooling system design,” *Structural
  and Multidisciplinary Optimization* 62, 3347–3366 (2020).
- Svanberg, “The method of moving asymptotes,” *International Journal for
  Numerical Methods in Engineering* 24(2), 359–373 (1987).
