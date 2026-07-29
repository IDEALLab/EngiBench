# MTO2D

> **Draft implementation.** The Python API, isolated case runner, dataset
> generator, and legacy-data converter are implemented for review. A source
> image recipe now builds and embeds a pristine exportable solver case, but
> the redistributable runtime image is not published yet. The
> `IDEALLab/mto_2d_v0` dataset is public; solver calls still require a locally
> built image (or an explicit compatible runtime and case).

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

The published cold-start designs used the final RAMP parameter `q=0.01`, which
is also the default for `simulate()`. The converted v0 data nevertheless
preserves the published objectives as source labels and sets
`objectives_evaluated_on_design = False`: the public arrays are lossy, and the
source labels describe the field before its final MMA update.

At the pinned source revision, 3,149 of 5,666 published power labels are
slightly above their listed bound, but the largest excess is only `0.4997`
normalized units (`0.976%`). This is consistent with a legacy feasibility
tolerance, not with the much larger `q=0.019` experiment. Keep the exact
residual and state the tolerance used when deriving any feasible/infeasible flag.

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
- `volfrac`: maximum all-cell fluid fraction, nominally from `0.25` to
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

The freeze uses one solver step with `qu = 0.01`,
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
same objectives plus constraint residuals, elapsed time, and an artifact path
when retained. Solver failures raise `SolverRunError` rather than returning a
result.

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
        "volfrac": 0.61,
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
initial physics toward the configured endpoints; warm mode starts at those
endpoints. The default endpoint is the source-matched `qu=0.01`,
`alphaMax=5.0252e6`, and `Heaviside=59.8`. `simulate()` freezes those endpoint
values and does not depend on the optimization schedule.

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
starting_design = problem.uniform_starting_design(problem.conditions.volfrac)
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
This extension should be called out explicitly in the upstream pull request
because it establishes an optimize-side counterpart to the already supported
`SimulationResult` subclass pattern.

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

An explicitly supplied case template must contain at least `app/` and
`src_TF/`, including an ASCII 86,400-cell `gamma` template. Configure it with
`case_template="/path/to/case"` or the
`ENGIBENCH_MTO2D_CASE_TEMPLATE` environment variable. When the container
backend has no host template, the runner instead calls
`mto2d-export-case` in the image to materialize a pristine writable template
inside the isolated run directory.

The retained solver currently depends on a Linux/HPC stack with OpenFOAM 5,
PETSc, MPI, GroovyBC, `libMMA_yu`, and custom adjoint boundary-condition
libraries. Three execution backends are available:

- `local` runs the OpenFOAM commands directly in the current environment.
- `container` is the installed default. It uses EngiBench's container
  abstraction and `MTO2D.container_id`; `container_image` or
  `ENGIBENCH_MTO2D_IMAGE` can override that image.
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

The generic EngiBench shared test exercises a different deterministic input:
`MTO2D(seed=1).random_design()` selects train index 2010 (source case 6799)
and evaluates it under the class-default conditions. The local parity image
produced the provisional source-matched `q=0.01` reference
`[13.8912, 63.8033]`. This value is committed so the release path is ready,
but it must be confirmed once against the source-built published image before
the artifact gate is enabled.

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

After validation succeeds, build a local exact-design dataset with the four
small metadata arrays from the pinned Hugging Face snapshot:

```bash
python -m engibench.problems.mto2d.model.reformat_native_gamma_dataset \
  --gamma-dir /path/to/source-gammas \
  --raw-dir /path/to/pinned/MTO-2D/snapshot \
  --output-dir dataset_output/mto_2d_exact_source_v0 \
  --cache-dir dataset_output/mto_2d_exact_source_v0_cache
```

The converter rechecks every field checksum, converts OpenFOAM storage order
to the native `(400, 200)` orientation, and preserves the paper-compatible
4,249/283/1,134 split. Rows set `design_is_exact=true` and
`objectives_evaluated_on_design=false`. Their volume residual is computed from
the exact 86,400-cell field; their temperature, power, and power residuals
remain historical pre-update labels. The generated card and manifest
explicitly block publication until redistribution rights are confirmed.

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
- `inlet_velocity`, `max_power_dissipation`, and `volfrac` as
  `float64`;
- `mean_temperature` and `power_dissipation` as `float32`;
- absolute and relative power-constraint residuals;
- the volume-constraint residual when available;
- source IDs, row indices, provenance, revision, and timing fields; and
- `design_is_exact`, which distinguishes native solver output from lossy
  legacy reconstruction, plus `objectives_evaluated_on_design`, which is false
  for reconstructed legacy rows and for generated rows created with
  `--no-evaluate-final`.

### Validate and load the canonical exact result

The publication helper performs a complete, non-networked validation by
default:

```bash
python -m engibench.problems.mto2d.model.publish_native_dataset \
  --dataset-dir dataset_output/mto_2d_exact_source_v0
```

It verifies all 5,666 native designs, schema and bounds, exact split
membership, pinned source hashes, gamma-validation evidence, provenance, and
the historical-label semantics. To use the richer local audit copy instead of
the public six-column projection, inject it explicitly:

```python
from datasets import load_from_disk

from engibench.problems.mto2d import MTO2D

dataset = load_from_disk("dataset_output/mto_2d_exact_source_v0")
problem = MTO2D(dataset=dataset)
design, row_index = problem.random_design("train")
```

### Push to Hugging Face

Authenticate with the Hugging Face CLI in your shell; do not place access
tokens in source files or command history. After confirming redistribution
rights, publish the already validated exact-native dataset with:

```bash
python -m engibench.problems.mto2d.model.publish_native_dataset \
  --dataset-dir dataset_output/mto_2d_exact_source_v0 \
  --repo-id IDEALLab/mto_2d_v0 \
  --license mit \
  --confirm-redistribution-rights \
  --push
```

The two independent confirmation flags prevent an accidental upload. The
example verifies that the existing human-edited card declares the intended
EngiBench `mit` license. The publisher validates the rich local audit data,
projects the six EngiBench columns, and replaces only `data/*.parquet` in the
existing repository. It does not upload or rewrite the README, conversion
manifest, gamma-validation JSONL, or other local evidence. The objective
columns remain historical pre-update labels, not fresh frozen evaluations of
the exact stored fields.

If rights for the retrieved native fields remain unresolved, the public
MIT-licensed lossy conversion can instead be regenerated and published under
a distinct fallback ID:

```bash
python -m engibench.problems.mto2d.model.reformat_hf_dataset \
  --output-dir dataset_output/mto_2d_lossy_v0 \
  --push-to-hub IDEALLab/mto_2d_lossy_v0 \
  --max-shard-size 500MB
```

## Dataset-backed `v0.py` demo

Run the problem module against the local `DatasetDict` to sample a real
converted design, print its source conditions and objectives, and render the
symmetric heat sink:

```bash
python ./engibench/problems/mto2d/v0.py \
  --dataset dataset_output/mto_2d_exact_source_v0
```

Use `--no-show` in a headless shell. This default demo does **not** launch
OpenFOAM.

Solver-backed evaluation is deliberately opt-in:

```bash
python ./engibench/problems/mto2d/v0.py \
  --dataset dataset_output/mto_2d_exact_source_v0 \
  --simulate \
  --solver-config ./solver.json \
  --no-show
```

`--simulate` is a real frozen OpenFOAM evaluation, not a lookup or surrogate.
It therefore still requires the external Linux/OpenFOAM runtime and solver
case described above. The command evaluates the sampled exact design under
that row's three physical conditions. When no `--dataset` is given in a source
checkout, this exact-source dataset is preferred automatically.

Most importantly, the two objective values stored in the converted dataset
describe the pre-update source field, while each exact `app/200/gamma`
topology is post-update. A successful `--simulate` result is a fresh evaluation
of the exact stored topology and can therefore differ slightly from its
historical objective labels.

There is an additional timing difference. The source optimizer logged
each objective before its MMA/Heaviside update, but wrote `app/200/gamma`
after that update. The published label and published final topology therefore
did not describe exactly the same field even before resizing. Default
`simulate()` now uses the source-matched final value `q=0.01`.

For example, converted source case `9130` has stored power `57.4882`.
Evaluating its lossy reconstruction at `q=0.01` gives `58.5875`. Evaluating
the exact post-update `app/200/gamma` gives `57.0208`; it still does not equal
the pre-update `57.4882` source label. The pre-update field was not written and
cannot be recovered from the final gamma.

The former stricter `q=0.019` experiment remains available explicitly:

```python
strict_experiment = problem.simulate_verbose(
    design,
    config={
        "qu_final": 0.019,
        "alpha_max_final": 5_025_200.0,
        "heaviside_final": 59.8,
    },
)
```

This is not the default v0 simulation. Any dataset objective intended to
describe the stored topology should come from a fresh frozen evaluation rather
than reusing the pre-update labels.

## Current limitations and release blockers

- No reproducible OCI solver image is published. `container_id` reserves the
  intended GHCR tag, which must be replaced with its immutable digest after
  publication.
- The wheel does not include a solver case or compiled executable. The
  source-built image will export its pristine prepared template on demand.
- The legacy `256 x 256` conversion is useful for training and analysis but
  cannot reproduce native solver cell values exactly.
- Redistribution rights for the adapted solver, `libMMA_yu`, and bundled
  libraries still need to be established. The migrated solver source has no
  obvious top-level license, so this draft does **not** claim that it can be
  redistributed.
- A fresh Linux environment must reproduce the reference case before this
  problem is considered release-ready.

The package remains on its feature branch until the remaining artifacts exist.
Once a legally redistributable OCI image is available by immutable digest and
a canonical source-matched reference is recorded, set `container_id` and
change the MTO2D
`ProblemTestPolicy.artifacts_available` flag in
`tests/test_problem_implementations.py` to `True`. The ordinary shared suite
will then exercise dataset loading and frozen simulation. Optimization is
entirely excluded from that shared suite because even a shortened external
solve is unsuitable for its lightweight contract; focused opt-in integration
tests remain the appropriate coverage. The initial OCI release is
`linux/amd64`, so the shared simulation is also skipped on Linux ARM until a
native multi-architecture image is published; explicitly configured amd64
emulation remains available for local use.

## References

- *To Quantize or Not to Quantize: Effects on Generative Models for Topology
  Optimization Problems*,
  [doi:10.1115/1.4071440](https://doi.org/10.1115/1.4071440).
- [IDEALLab/MTO-2D raw dataset](https://huggingface.co/datasets/IDEALLab/MTO-2D).
- [IDEALLab/VQGAN-TO utilities](https://github.com/IDEALLab/VQGAN-TO).
- [IDEALLab/EngiBench](https://github.com/IDEALLab/EngiBench).
