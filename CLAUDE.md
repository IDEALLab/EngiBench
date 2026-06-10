# CLAUDE.md

Project-specific guidance for Claude Code in this repository. The global `~/.claude/CLAUDE.md`
covers the HPC environment, Slurm allocation, and scratch layout — that still applies.
See [AGENTS.md](AGENTS.md) for the canonical dev-setup / test / docs / commit-convention reference.

## What this fork is for

EngiBench is a benchmark framework wrapping engineering simulators behind a common `Problem` API.
**This is a single-contributor fork focused almost entirely on the `airfoil` problem**, specifically:
parallel airfoil simulation via the Slurm API, dataset generation on HPC, and — current work —
**extending airfoil output to emit full surface fields** (velocity components + pressure coefficient),
not just lift/drag.

Branch context: work happens on feature branches like `airfoil-field-output`; `main` is the upstream-tracking branch.

## Orientation: the airfoil pipeline

Everything lives under [engibench/problems/airfoil/](engibench/problems/airfoil/):

- [v0.py](engibench/problems/airfoil/v0.py) — the `Airfoil(Problem)` class. The methods that matter:
  - `simulate` → returns just objective values `[drag, lift]` (drag first, then lift); delegates to
    `simulate_verbose(...).objective_values`.
  - `simulate_verbose` → **the field-output path**: runs the same sim and returns an
    `AirfoilSimulationResult` (a `SimulationResult` subclass defined in this module) with
    `.objective_values = [drag, lift]` and `.surface_fields` of shape `(6, N)`, rows
    `[CoordinateX, CoordinateY, VelocityX, VelocityY, VelocityZ, CoefPressure]`. (The old standalone
    `simulate_field` method was removed in favor of this verbose path.)
  - `simulator_output_to_design(field_output=...)` → parses the MACH-Aero `*_slices.dat` output;
    returns `(2, N)` coords or `(6, N)` coords+fields.
  - `optimize` → SLSQP shape optimization via pyoptsparse; reads `opt.hst` history.
- [utils.py](engibench/problems/airfoil/utils.py) — geometry + slice parsing. `reorder_coords` (2×N) and
  `reorder_coords_fields` (6×N) share the same segment-ordering pipeline; the field version reuses the
  reordered point indices to align field values. Keep these two in lockstep when editing ordering logic.
- [templates/](engibench/problems/airfoil/templates/) — Python scripts (`pre_process.py`, `airfoil_analysis.py`,
  `airfoil_opt.py`, `cli_interface.py`) that run **inside the MACH-Aero container**, driven by JSON args
  serialized from the dataclasses in `cli_interface.py`. Changing simulator inputs/outputs means editing
  both the host-side dataclass and the in-container script.
- `pyopt_history.py` / `fake_pyoptsparse/` — read optimizer history; the fake module lets histories load
  without a real pyoptsparse install.

### How a simulation actually runs

`simulate*` writes a per-run **study directory** (`engibench_studies/problems/airfoil/study_<seed>-pid<pid>/`)
under the base/current working directory, mounts it into the `mdolab/public:u22-gcc-ompi-stable` container,
and runs `mpirun -np <mpicores> python -m mpi4py .../airfoil_analysis.py '<json>'`. Results come back as
`output/outputs.npy` (lift/drag) and `output/*_slices.dat` (surface fields). `reset(cleanup=True)` wipes the
study dir.

**Container runtime** is auto-detected ([engibench/utils/container.py](engibench/utils/container.py)); on this
HPC cluster it resolves to **Apptainer/Singularity** (Docker isn't available on compute nodes). Override with
the `CONTAINER_RUNTIME` env var. Apptainer images cache to `$SCRATCH`.

## Slurm / HPC dataset generation

The Slurm API is in [engibench/utils/slurm/](engibench/utils/slurm/):
- `sbatch_map(f, args, slurm_args=SlurmConfig(...), group_size=, work_dir=)` pickles a callable + per-job
  kwargs and submits a job array; each array element runs `run_job.py`. Returns a `SubmittedJobArray` with
  `.save(out_path)` and `.reduce(f_reduce)` (both submit a dependent post-processing job; `.reduce` enforces
  a 10 MB pickle size limit so login nodes aren't abused).
- `group_size` batches multiple sims sequentially into one array element; array is capped at `%1000` concurrent.
- `MemorizeModule` is the trick that lets `__main__`-defined callables unpickle on the worker — don't break it,
  and keep a `__main__` guard on any script submitted this way (the loader hard-fails without one).

The end-to-end driver is [dataset_slurm_airfoil.py](engibench/problems/airfoil/dataset_slurm_airfoil.py)
(LHS-samples Mach/Reynolds/AoA per design, batches via `sbatch_map`), calling the job wrappers in
[simulation_jobs.py](engibench/problems/airfoil/simulation_jobs.py) (`simulate_slurm`, `optimize_slurm`).
The `--verbose` flag threads through to `simulate_slurm`'s `verbose` kwarg, which routes to `simulate_verbose`
(coefficients + `surface_fields`) vs `simulate` (coefficients only); verbose results store the field array
under the `surface_fields` key. `optimize_slurm` is stubbed — `optimize` is not yet wired for Slurm dataset generation.

Run it (must pass an allocation account):
```sh
python engibench/problems/airfoil/dataset_slurm_airfoil.py \
  -account fuge-prj-jrl -type simulate -n_designs 10 -n_flows 4 \
  -group_size 2 --verbose
```

## Working agreements

- Commits use conventional-commit format enforced by pre-commit (`feat`/`fix`/`docs`/`refactor`/…), scope in
  parens, lowercase imperative description ≤72 chars, no trailing period. See [AGENTS.md](AGENTS.md) for the full spec.
- Before committing: `ruff check`, `ruff format --check`, `mypy .`, `pytest`. `OMP_NUM_THREADS=1` is required
  on this cluster (already set in `~/.bashrc`).
- Airfoil tests: [tests/test_airfoil.py](tests/test_airfoil.py); the full-simulator path is exercised via
  [tests/test_problem_implementations.py](tests/test_problem_implementations.py) and needs the container.
  When adding field-output behavior, prefer unit-testing the slice-parsing/reorder functions in `utils.py`
  with fixture `*_slices.dat` data rather than spinning up the container.
- When touching simulator I/O, remember the boundary: host-side dataclass (`cli_interface.py`) **and** the
  in-container template script must agree, and `simulator_output_to_design` must match whatever the container writes.
