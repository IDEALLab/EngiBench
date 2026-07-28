# MTO2D

<!-- start docs -->

> **Draft implementation.** The Python API, isolated case runner, dataset
> generator, and legacy-data converter are implemented for review. The solver
> case, a redistributable runtime image, and the provisional
> `IDEALLab/mto_2d_v0` dataset are not bundled or published yet. Therefore,
> `simulate()` and `optimize()` require an externally supplied solver case and
> runtime.

MTO2D is a two-dimensional multiphysics topology-optimization problem for a
fluid-cooled heat sink. It couples flow and heat transfer, and uses an adjoint
solver with the method of moving asymptotes (MMA) to update the material
distribution.

The public implementation is intentionally small:

- `v0.py` defines the EngiBench problem and API.
- `model/design_io.py` converts between NumPy designs and OpenFOAM `gamma`
  fields.
- `model/runner.py` prepares and runs an isolated solver case.
- `model/dataset.py`, `model/generate_dataset.py`, and
  `model/reformat_hf_dataset.py` contain the heavier dataset workflows.

## Design space

An EngiBench design is a `float32` array with shape `(400, 200)` and values in
`[0, 1]`. It stores only the non-redundant left half of the design domain:

- `gamma = 0` means solid.
- `gamma = 1` means fluid.
- `render()` mirrors the half-domain horizontally to make a symmetric
  `(400, 400)` image.

The OpenFOAM field has 86,400 cells. The first 80,000 cells encode the
`(400, 200)` design in two solver-specific blocks, while the remaining 6,400
cells are fixed, non-design cells copied unchanged from the case template.
`design_io.py` owns this mapping so that simulation never silently resizes a
design.

### Relation to the published `256 x 256` data

Each design in the published
[`IDEALLab/MTO-2D`](https://huggingface.co/datasets/IDEALLab/MTO-2D) NumPy
dataset is a `(256, 256)` image of the **whole left half-domain**. It is not a
full symmetric design and not one quadrant. The native `(400, 200)` half was
anisotropically and lossily resized to that square representation.

The converter reconstructs a native-shaped array with the corresponding
non-antialiased bicubic transform and records `design_is_exact = False`.
Resizing cannot recover the original cell values, so the stored objective
values belong to the published legacy design's source case, not to an exactly
reproducible native field. Use a solver-native `(400, 200)` design and
`design_is_exact = True` when exact simulation is required.

## Objectives

Objective arrays always use this order:

0. `mean_temperature` — minimize the mean temperature.
1. `power_dissipation` — minimize normalized fluid power dissipation.

The native optimization formulation minimizes mean temperature. Power
dissipation and fluid volume are solver constraints; power dissipation is also
reported as the second EngiBench objective so that designs can be compared in
Pareto studies.

The output-dependent power constraint cannot be checked from inputs alone.
`simulate_verbose()` reports its relative residual as
`power_dissipation / max_power_dissipation - 1`. It also returns the
solver-reported volume residual. Ordinary `check_constraints()` validates the
design bounds and input conditions only.

## Conditions

The public conditions are:

- `inlet_velocity`: signed inlet velocity, nominally from `-0.095` to
  `-0.025` m/s.
- `max_power_dissipation`: power-dissipation limit as a multiple of the
  published reference scale `J1`, nominally from `50` to `75`.
- `volume_fraction`: maximum all-cell fluid fraction, nominally from `0.25` to
  `0.70`.

The converter preserves source values rather than clipping them. In
particular, a small number of rows in the published dataset have power limits
slightly below the nominal range.

## How `simulate()` works

In simple terms, `simulate()` asks: “How well does this fixed heat-sink layout
perform under these conditions?”

The runner:

1. checks the native design's shape, values, and finiteness;
2. copies a pristine external OpenFOAM case into a unique temporary directory;
3. writes the design and the three physical conditions;
4. sets the final physical parameters and freezes MMA movement;
5. runs `blockMesh`, `decomposePar`, the parallel solver, and
   `reconstructPar`;
6. reads the final temperature, power, volume-residual, and timing values; and
7. removes temporary artifacts unless retention was requested.

The freeze uses one solver step with `qu = 0.019`,
`alphaMax = 5.0252e6`, `Heaviside = 59.8`, and movement limit `0`.
`simulate()` returns only
`[mean_temperature, power_dissipation]`. `simulate_verbose()` returns those
same objectives plus constraint residuals, elapsed time, status, and an
artifact path when retained.

```python
import numpy as np

from engibench.problems.mto2d import MTO2D

problem = MTO2D(
    config={
        "case_template": "/path/to/mto2d-case",
        "backend": "local",
        "mpi_cores": 8,
        "inlet_velocity": -0.074,
        "max_power_dissipation": 63.1,
        "volume_fraction": 0.61,
    }
)
design = np.load("native_half_400x200.npy").astype(np.float32)

objectives = problem.simulate(design)
detailed = problem.simulate_verbose(design)
```

## How `optimize()` works

In simple terms, `optimize()` asks: “Starting from this layout, where should
the solver put fluid and solid to lower temperature while respecting the power
and volume limits?”

At each iteration, the solver computes the flow and temperature fields, uses
adjoint sensitivities to estimate how changing every design cell affects the
problem, and lets MMA choose a bounded design update. A cold run continues from
soft initial physics toward the final parameters; a warm run can start at or
near those final parameters.

```python
starting_design = problem.uniform_starting_design(
    problem.conditions.volume_fraction
)
optimized_design, history = problem.optimize(
    starting_design,
    config={"mode": "cold", "max_iter": 200},
)

# Re-evaluate the returned update when exact final objectives are needed.
final_objectives = problem.simulate(optimized_design)
```

Every `OptiStep.obj_values` follows the two-objective order documented above.
One legacy-solver detail matters: iteration row `k` describes the field
evaluated **before** its MMA update, while the returned final `gamma` is the
subsequent update. Consequently, the last history row is not an exact
evaluation of the returned field. Run the frozen `simulate()` call on the
result when exact final objectives are needed.

## Solver setup and execution backends

The case template must contain at least `app/` and `src_TF/`, including an
ASCII 86,400-cell `gamma` template. Configure it with
`case_template="/path/to/case"` or the
`ENGIBENCH_MTO2D_CASE_TEMPLATE` environment variable.

The retained solver currently depends on a Linux/HPC stack with OpenFOAM 5,
PETSc, MPI, GroovyBC, `libMMA_yu`, and custom adjoint boundary-condition
libraries. Three execution backends are available:

- `local` runs the OpenFOAM commands directly in the current environment.
- `container` uses EngiBench's container abstraction and a user-supplied
  `container_image` (or `ENGIBENCH_MTO2D_IMAGE`).
- `command` calls a user-supplied `driver_command`; the prepared case path is
  available as `MTO2D_CASE_DIR`. This is useful for site-specific schedulers,
  modules, or a Docker/Podman/Apptainer wrapper.

Each call works in a unique directory, so concurrent runs do not share solver
state. Set `retain_artifacts=True` to keep a successful run. By default,
`retain_on_failure=True` keeps a failed case and `run.log`; the raised
`SolverRunError` gives the retained path. The container backend currently
cannot enforce the API timeout; use the `command` backend when a hard timeout
is required.

### Reference case

The retained migration case is the initial reproduction target:

| Quantity | Reference value |
| --- | ---: |
| Inlet velocity | `-0.074` |
| Maximum power dissipation | `63.1` |
| Volume fraction | `0.61` |
| Mean temperature | `9.45825` |
| Power dissipation | `62.2588` |
| Volume residual | `-0.000671484` |
| Elapsed wall time | `13,713 s` |

These numbers came from the original retained case. Reproduction through a
fresh, published runtime image remains a release requirement, not a claim of
the current draft.

## Generate a 10,000-case gridded dataset

The default Cartesian grid has shape `20 x 20 x 25`, with axes spanning the
nominal inlet-velocity, power-limit, and volume-fraction ranges. This produces
exactly 10,000 condition tuples. Preview it without running a solver:

```bash
python -m engibench.problems.mto2d.model.generate_dataset generate \
  --output-dir ./mto2d_shards \
  --dry-run
```

Pass solver settings in a JSON object:

```json
{
  "case_template": "/path/to/mto2d-case",
  "backend": "local",
  "mpi_cores": 16,
  "mode": "cold",
  "max_iter": 200,
  "timeout": 43200,
  "retain_on_failure": true
}
```

Start with a short slice:

```bash
python -m engibench.problems.mto2d.model.generate_dataset generate \
  --output-dir ./mto2d_shards \
  --solver-config ./solver.json \
  --start-index 0 \
  --stop-index 2
```

Each case is written atomically as `case_NNNNN.npz`. Existing valid shards are
reused, making interrupted campaigns resumable. By default, each optimization
is followed by a frozen evaluation of the returned field. Use
`--no-evaluate-final` only if the pre-update last history row is acceptable.
Use `--force` to replace completed shards.

A full campaign is cluster-scale work. The retained 200-step reference run
took 13,713 seconds, and runtime varies with hardware, MPI setup, topology, and
convergence. Do not treat 10,000 sequential local jobs as a practical default.
The generator integrates with EngiBench's SLURM helper:

```bash
python -m engibench.problems.mto2d.model.generate_dataset generate \
  --output-dir ./mto2d_shards \
  --solver-config ./solver.json \
  --slurm \
  --account MY_ACCOUNT \
  --runtime 12:00:00 \
  --ntasks 16 \
  --group-size 1
```

Workers return small shard paths rather than transferring the large designs
through the scheduler. Site-specific `sbatch` options can be repeated with
`--sbatch-extra-arg`. Assemble the completed contiguous shard set into
deterministic 75/5/20 `train`, `val`, and `test` splits:

```bash
python -m engibench.problems.mto2d.model.generate_dataset assemble \
  --shard-dir ./mto2d_shards \
  --output-dir ./mto2d_dataset \
  --expected-count 10000 \
  --seed 1
```

The script saves a local Hugging Face `DatasetDict`; it does not upload data.

## Reformat the published MTO-2D data

The existing source contains 5,666 rows in five raw NumPy files. The formatter
memory-maps the approximately 1.49 GB legacy design array, reconstructs one
native-shaped row at a time, and streams the result into Arrow:

```bash
python -m engibench.problems.mto2d.model.reformat_hf_dataset \
  --raw-dir /path/to/five-npy-files \
  --output-dir ./mto2d_dataset \
  --seed 1
```

Omit `--raw-dir` to download the pinned source revision through the Hugging
Face cache:

```bash
python -m engibench.problems.mto2d.model.reformat_hf_dataset \
  --output-dir ./mto2d_dataset
```

Nothing is downloaded merely by importing MTO2D, and the converter never
uploads data. The default 75/5/20 split contains 4,249 training, 283
validation, and 1,134 test rows.

Both generated and converted datasets use the same schema:

- `optimal_design` and the three condition fields;
- `mean_temperature` and `power_dissipation`;
- absolute and relative power-constraint residuals;
- the volume-constraint residual when available;
- source IDs, row indices, provenance, revision, and timing fields; and
- `design_is_exact`, which distinguishes native solver output from lossy
  legacy reconstruction, plus `objectives_evaluated_on_design`, which is false
  for reconstructed legacy rows and for generated rows created with
  `--no-evaluate-final`.

The problem's `dataset_id` is provisionally `IDEALLab/mto_2d_v0`. Until that
repository is published, load a local conversion explicitly:

```python
from datasets import load_from_disk

from engibench.problems.mto2d import MTO2D

dataset = load_from_disk("./mto2d_dataset")
problem = MTO2D(dataset=dataset)
design, row_index = problem.random_design("train")
```

## Current limitations and release blockers

- No reproducible OCI solver image is published, and `container_id` is
  intentionally unset.
- The solver case and compiled executable are not included in EngiBench.
- The new `IDEALLab/mto_2d_v0` dataset has not been published.
- The legacy `256 x 256` conversion is useful for training and analysis but
  cannot reproduce native solver cell values exactly.
- Redistribution rights for the adapted solver, `libMMA_yu`, and bundled
  libraries still need to be established. The migrated solver source has no
  obvious top-level license, so this draft does **not** claim that it can be
  redistributed.
- A fresh Linux environment must reproduce the reference case before this
  problem is considered release-ready.

## References

- *To Quantize or Not to Quantize: Effects on Generative Models for Topology
  Optimization Problems*,
  [doi:10.1115/1.4071440](https://doi.org/10.1115/1.4071440).
- [IDEALLab/MTO-2D raw dataset](https://huggingface.co/datasets/IDEALLab/MTO-2D).
- [IDEALLab/VQGAN-TO utilities](https://github.com/IDEALLab/VQGAN-TO).
- [IDEALLab/EngiBench](https://github.com/IDEALLab/EngiBench).

<!-- end docs -->
