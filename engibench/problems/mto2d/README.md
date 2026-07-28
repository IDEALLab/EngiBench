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
- `model/retrieve_native_gammas.py` retrieves and validates authorized
  solver-native source fields without modifying the remote case tree.

## Design space

An EngiBench design is a `float32` array with shape `(400, 200)` and values in
`[0, 1]`. It stores only the non-redundant left half of the design domain:

- `gamma = 0` means solid.
- `gamma = 1` means fluid.
- `render()` mirrors the half-domain horizontally to make a symmetric
  `(400, 400)` image.

Following the
[`beams_3d_16_v0`](https://huggingface.co/datasets/IDEALLab/beams_3d_16_v0)
Hub convention, datasets store `optimal_design` as a flat
`list<float32>` of length 80,000 in C order. `MTO2D.random_design()` reshapes
that sequence back to the native `(400, 200)` API representation.

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

The legacy `gamma_npy.py` helper can look contradictory because it mirrors the
solver-native half and returns a `(400, 400)` tensor. That tensor is a full
visualization/conversion representation; it is not the representation stored
by the published `(256, 256)` NumPy dataset and is not accepted directly by the
simulator.

The converter reconstructs a native-shaped array with the corresponding
non-antialiased bicubic transform and records `design_is_exact = False`.
Resizing cannot recover the original cell values, so the stored objective
values belong to the published legacy design's source case, not to an exactly
reproducible native field. Use a solver-native `(400, 200)` design and
`design_is_exact = True` when exact simulation is required.

There is also a solver-fidelity difference. The published cold-start designs
were optimized with a lower RAMP parameter, while `simulate()` deliberately
uses the strict final Brinkman/RAMP parameters. Appendix E of the accompanying
paper reports that all test designs violate the strict constraint before
warm-starting. The converted v0 data therefore preserves the published
objectives as source labels and sets `objectives_evaluated_on_design = False`;
those labels must not be presented as strict evaluations of the reconstructed
arrays.

At the pinned source revision, 3,149 of 5,666 published power labels are
slightly above their listed bound, but the largest excess is only `0.4997`
normalized units (`0.976%`). This is consistent with a legacy feasibility
tolerance, not with the much larger strict re-simulation gap. Keep the exact
residual and state the tolerance used when deriving any feasible/infeasible
flag.

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
- `max_power_dissipation`: dimensionless normalized power-dissipation limit,
  nominally from `50` to `75`. The retained solver uses the exact
  `D_normalization = 1.57572e-7`; the paper calls the rounded
  `J1 ≈ 1.58e-7` reference scale.
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
4. sets the final physical parameters and disables design updates;
5. runs `blockMesh` and the solver, using decomposition and MPI when requested;
6. reads the final temperature, power, volume-residual, and timing values; and
7. removes temporary artifacts unless retention was requested.

The freeze uses one solver step with `qu = 0.019`,
`alphaMax = 5.0252e6`, `Heaviside = 59.8`, movement limit `0`, and
`updateDesign = false`. The patched solver records the physical objectives but
bypasses its sensitivity/MMA update, so the written `gamma` remains finite and
equal to the input design within OpenFOAM's output precision. The runner
validates that invariant and reconstructs parallel output before returning.
Because the objectives were already recorded before MMA in the legacy solver,
this bypass fixes the invalid output field without changing the measured
temperature or power.
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
problem, and lets MMA choose a bounded design update.

`optimization_schedule="legacy"` is the default cold schedule. It reproduces
the source optimizer's iteration timing instead of approximating it with one
interpolation curve: `qu` stays at `0.005`, begins increasing at iteration 61,
and caps at `0.01`; `alphaMax` follows the original linear-then-geometric
formula to `5.0252266e6`; and the Heaviside parameter advances from `0.1` to
`59.8` immediately before the same projections as the source solver. The full
published path is 200 iterations. A shorter run is an exact prefix for smoke
testing, not a converged substitute.

`optimization_schedule="strict"` uses the configurable
`continuation_profile` and endpoint fields. Cold mode continues from soft
initial physics toward `qu=0.019`, `alphaMax=5.0252e6`, and
`Heaviside=59.8`; warm mode starts at those endpoints. The canonical
`simulate()` call is always a strict frozen evaluation and does not depend on
the optimization schedule.

Cold mode also preserves the legacy power-bound continuation: by default the
active bound starts at `90` and decreases by `0.2` per iteration until it
reaches `max_power_dissipation`. A short cold run can therefore exercise the
solver without yet enforcing the requested final bound. `optimize_verbose()`
warns in that situation and returns both `active_power_bounds` and residuals
against those bounds. Its `power_constraint_residuals` always use the requested
final bound.

Warm mode defaults to the requested power bound from its first iteration.
This is important because the legacy warm-start scripts reset their iteration
counter and otherwise restart the loose bound near `90`. Set
`power_bound_start=90` explicitly only when historical warm-start parity is
required. Legacy is deliberately a cold-only selector; strict repair must pass
both `mode="warm"` and `optimization_schedule="strict"`.

```python
starting_design = problem.uniform_starting_design(problem.conditions.volume_fraction)
optimized_design, history = problem.optimize(
    starting_design,
    config={"mode": "cold", "max_iter": 200},
)

# Re-evaluate the returned update when exact final objectives are needed.
final_evaluation = problem.simulate_verbose(optimized_design)

# A positive value means the returned design still needs strict warm repair.
if final_evaluation.power_constraint_residual > 0:
    optimized_design, repair_history = problem.optimize(
        optimized_design,
        config={
            "mode": "warm",
            "optimization_schedule": "strict",
            "max_iter": 20,
        },
    )
    final_evaluation = problem.simulate_verbose(optimized_design)
```

`optimize_verbose()` is a deliberate MTO2D-specific extension, not a method on
EngiBench's base `Problem` API. It preserves the standard `optimize()` return
contract while additionally exposing constraint and elapsed-time histories.

Every `OptiStep.obj_values` follows the two-objective order documented above.
One legacy-solver detail matters: iteration row `k` describes the field
evaluated **before** its MMA update, while the returned final `gamma` is the
subsequent update. Consequently, the last history row is not an exact
evaluation of the returned field. Run the frozen `simulate()` call on the
result when exact final objectives are needed.

For dataset production, the frozen result is the canonical EngiBench
measurement. Preserve any historical values in provenance/source fields,
retain and flag a strictly infeasible design, or continue warm repair and
re-evaluate it. Do not rescale the measured power or silently attach a
pre-update history value to the returned design.

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

The source-tree demonstration accepts a solver JSON with `--solver-config`.
For `v0.py --simulate` only, it also checks
`ENGIBENCH_MTO2D_SOLVER_CONFIG` and then the migration workspace's private
`../.artifacts/mto2d-docker.json`. This convenience does not change the
defaults of the programmatic `MTO2D()` API, and render-only CLI calls do not
load machine-specific runtime configuration.

### Local Docker parity image

EngiBench's HeatConduction2D, HeatConduction3D, and Airfoil problems use
externally published OCI images through the same container abstraction.
MTO2D follows the Airfoil pattern: it mounts one isolated case directory and
runs MPI inside the container.

Until a licensed, source-built image can be published, the scripts in
[`model/runtime`](model/runtime/README.md) can convert the exact retained
`MTO_GEN.sif` into a private `linux/amd64` Docker image:

```bash
./engibench/problems/mto2d/model/runtime/convert_sif.sh \
  /path/to/MTO_GEN.sif \
  engibench-mto2d:sif-parity
```

This conversion is a numerical-parity tool, not a reproducible release
artifact. The helper verifies the original SIF's size and SHA-256, extracts
the root filesystem inside Docker, and restores the OpenFOAM/OpenMPI/PETSc
environment. On ARM hosts, set
`DOCKER_DEFAULT_PLATFORM=linux/amd64`; execution uses emulation.

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
  "optimization_schedule": "legacy",
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
deterministic 80/15/5 `train`, `val`, and `test` splits:

```bash
python -m engibench.problems.mto2d.model.generate_dataset assemble \
  --shard-dir ./mto2d_shards \
  --output-dir dataset_output/mto_2d_v0 \
  --expected-count 10000 \
  --seed 1
```

For the full 10,000-case grid this produces 8,000 training, 1,500 validation,
and 500 test rows. The script saves a local Hugging Face `DatasetDict`; it does
not upload data.

## Retrieve exact native source fields

The public `IDEALLab/MTO-2D` repository contains lossy `256 x 256` arrays, not
the original OpenFOAM fields. An authorized Zaratan user can retrieve the
exact final fields selected by the pinned `index_5666.npy` with a resumable,
manifest-driven workflow. The source is read-only: the command requests only
`2Dheatsink_<id>/app/200/gamma` and never uses rsync deletion or source-removal
options.

First prepare the 5,666-file manifest:

```bash
python -m engibench.problems.mto2d.model.retrieve_native_gammas prepare \
  --ids-npy /path/to/index_5666.npy \
  --output-dir /path/to/source-gammas
```

Then run the transfer in an interactive terminal so SSH password/MFA prompts
stay outside source files and command output. On macOS, `caffeinate` prevents
sleep during the transfer:

```bash
caffeinate -i python \
  -m engibench.problems.mto2d.model.retrieve_native_gammas fetch \
  --output-dir /path/to/source-gammas \
  --bwlimit-kib 1024
```

The command pins `login-1`, disables SSH multiplexing, uses compression and
keepalives, preserves partial files, and reads paths from `gamma-files.txt`.
Rerun the same command after a disconnect; rsync skips completed files and
reuses partial destinations.

After transfer, parse and hash every field:

```bash
python -m engibench.problems.mto2d.model.retrieve_native_gammas validate \
  --output-dir /path/to/source-gammas
```

Validation requires exactly 86,400 finite values in `[0, 1]` and verifies that
the final 6,400 fixed cells are all fluid. It writes:

- `gamma-validation.jsonl`, with byte size and SHA-256 per source row;
- `gamma-validation-summary.json`, with valid/missing/invalid counts; and
- `gamma-retry-files.txt`, containing only missing or invalid paths.

Retry that smaller list with remote checksums when necessary:

```bash
caffeinate -i python \
  -m engibench.problems.mto2d.model.retrieve_native_gammas fetch \
  --output-dir /path/to/source-gammas \
  --retry-only --checksum --bwlimit-kib 1024
```

Do not publish these higher-resolution fields until their redistribution
status is confirmed. Retrieval also does not repair the historical label
timing mismatch: source objectives were recorded before the final MMA update,
while `app/200/gamma` was written afterward.

## Reformat the published MTO-2D data

The existing source contains 5,666 rows in five raw NumPy files. The formatter
memory-maps the approximately 1.49 GB legacy design array, reconstructs one
native-shaped row at a time, flattens it in C order, and streams the result
into Arrow:

```bash
python -m engibench.problems.mto2d.model.reformat_hf_dataset \
  --raw-dir /path/to/five-npy-files \
  --output-dir dataset_output/mto_2d_v0 \
  --seed 1
```

Omit `--raw-dir` to download the pinned source revision through the Hugging
Face cache:

```bash
python -m engibench.problems.mto2d.model.reformat_hf_dataset \
  --output-dir dataset_output/mto_2d_v0
```

Nothing is downloaded merely by importing MTO2D, and the converter never
downloads the source when `--raw-dir` is supplied. The published-data
converter reproduces the paper and released VQGAN-TO code: it uses
`int(0.75 * n)` training rows, `int(0.05 * n)` validation rows, assigns the
remainder to test, and reproduces seeded PyTorch `random_split` membership and
order (default seed 1). For 5,666 rows, that is 4,249 training, 283 validation,
and 1,134 test rows. This legacy policy is intentionally distinct from the
80/15/5 policy used above for future native solver-generated datasets.

For the pinned source revision, conversion verifies the SHA-256 digest of all
five NumPy files by default. It then validates the Arrow schema, split sizes,
flat design lengths and bounds, finite condition/objective values, unique
source IDs, provenance flags, and retained reference row. Finally, it saves,
reloads, and validates the `DatasetDict` again. The output directory also
contains `conversion_manifest.json` with the source revision, split policy,
native and stored shapes, and transform description. `--no-verify-hashes`
exists for deliberately converting a different source, but should not be used
for the pinned publication.

Both generated and converted datasets use the same schema:

- `optimal_design`: a flat C-order sequence of 80,000 `float32` values;
- `inlet_velocity`, `max_power_dissipation`, and `volume_fraction` as
  `float64`;
- `mean_temperature` and `power_dissipation` as `float32`;
- absolute and relative power-constraint residuals;
- the volume-constraint residual when available;
- source IDs, row indices, provenance, revision, and timing fields; and
- `design_is_exact`, which distinguishes native solver output from lossy
  legacy reconstruction, plus `objectives_evaluated_on_design`, which is false
  for reconstructed legacy rows and for generated rows created with
  `--no-evaluate-final`.

### Validate and load the local result

Conversion already performs validation. It can also be rerun independently
before publication:

```python
from datasets import load_from_disk

from engibench.problems.mto2d.model.dataset import validate_legacy_dataset

dataset = load_from_disk("dataset_output/mto_2d_v0")
print(validate_legacy_dataset(dataset))
# {'train': 4249, 'val': 283, 'test': 1134}
```

The problem's `dataset_id` is provisionally `IDEALLab/mto_2d_v0`. Until that
repository is published, inject the local conversion explicitly:

```python
from datasets import load_from_disk

from engibench.problems.mto2d import MTO2D

dataset = load_from_disk("dataset_output/mto_2d_v0")
problem = MTO2D(dataset=dataset)
design, row_index = problem.random_design("train")
```

### Push to Hugging Face

Authenticate with the Hugging Face CLI in your shell; do not place access
tokens in source files or command history. Conversion can validate and publish
in one command:

```bash
python -m engibench.problems.mto2d.model.reformat_hf_dataset \
  --raw-dir /path/to/five-npy-files \
  --output-dir dataset_output/mto_2d_v0 \
  --push-to-hub IDEALLab/mto_2d_v0 \
  --max-shard-size 500MB
```

To publish a previously converted and validated directory without downloading
or converting again:

```bash
python -m engibench.problems.mto2d.model.reformat_hf_dataset \
  --output-dir dataset_output/mto_2d_v0 \
  --push-to-hub IDEALLab/mto_2d_v0 \
  --max-shard-size 500MB
```

This path revalidates the saved dataset and uploads its Parquet shards,
dataset card, and `conversion_manifest.json`. The card keeps the MIT license,
citation, and lossy-reconstruction warning beside the data. Use `--private`
when a review upload should not be public.

## Dataset-backed `v0.py` demo

Run the problem module against the local `DatasetDict` to sample a real
converted design, print its source conditions and objectives, and render the
symmetric heat sink:

```bash
python ./engibench/problems/mto2d/v0.py \
  --dataset dataset_output/mto_2d_v0
```

Use `--no-show` in a headless shell. This default demo does **not** launch
OpenFOAM.

Solver-backed evaluation is deliberately opt-in:

```bash
python ./engibench/problems/mto2d/v0.py \
  --dataset dataset_output/mto_2d_v0 \
  --simulate \
  --solver-config ./solver.json \
  --no-show
```

`--simulate` is a real frozen OpenFOAM evaluation, not a lookup or surrogate.
It therefore still requires the external Linux/OpenFOAM runtime and solver
case described above. The command evaluates the sampled reconstruction under
that row's three physical conditions.

Most importantly, the two objective values stored in the converted dataset
belong to the original solver-native source topology. The published
`256 x 256` design was resized lossily, so those values are **not** reference
values for the reconstructed `(400, 200)` array. A successful `--simulate`
result is a fresh evaluation of the reconstruction and is expected to differ
from the stored legacy objectives.

There are two additional fidelity differences. The source optimizer logged
each objective before its MMA/Heaviside update, but wrote `app/200/gamma`
after that update. The published label and published final topology therefore
did not describe exactly the same field even before resizing. In addition,
the source cold optimizer capped `qu` at `0.01`, while canonical
`simulate()` uses the stricter final value `0.019`. The larger value increases
the flow resistance assigned to intermediate-density cells and can raise the
simulated power substantially.

For example, converted source case `9130` has stored power `57.4882`.
Evaluating its reconstructed design at `qu=0.01` gives `58.5875`, while the
canonical `qu=0.019` evaluation gives `67.636`. Thus the physics change
accounts for about 89% of that strict-versus-stored gap. The published Hugging
Face data cannot remove the remainder because it contains neither native
field. When the exact post-update `app/200/gamma` is retrieved separately, its
legacy frozen evaluation is `57.0208`; it still does not equal the pre-update
`57.4882` source label. The pre-update field was not written and therefore
cannot be recovered by retrieving the final gamma.

For a diagnostic comparison with the old material interpolation, override
the final evaluation parameters explicitly:

```python
legacy_like = problem.simulate_verbose(
    design,
    config={
        "qu_final": 0.01,
        "alpha_max_final": 5_025_200.0,
        "heaviside_final": 59.8,
    },
)
```

This is not the canonical v0 simulation and cannot recreate a legacy label
exactly. To obtain a design feasible under strict v0 physics, run a strict
warm optimization, freeze-simulate the returned field, and store those new
objectives rather than reusing the legacy labels.

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

The package remains on its feature branch until those artifacts exist.
Registration must not be made green with MTO2D-specific shared-test skips.
Once the exact dataset is published, a legally redistributable OCI image is
available by immutable digest, and a canonical strict reference is recorded,
set `container_id` and let the ordinary EngiBench built-in tests exercise the
problem.

## References

- *To Quantize or Not to Quantize: Effects on Generative Models for Topology
  Optimization Problems*,
  [doi:10.1115/1.4071440](https://doi.org/10.1115/1.4071440).
- [IDEALLab/MTO-2D raw dataset](https://huggingface.co/datasets/IDEALLab/MTO-2D).
- [IDEALLab/VQGAN-TO utilities](https://github.com/IDEALLab/VQGAN-TO).
- [IDEALLab/EngiBench](https://github.com/IDEALLab/EngiBench).

<!-- end docs -->
