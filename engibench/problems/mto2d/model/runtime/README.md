# MTO2D container runtime

EngiBench already runs Docker-backed problems by assigning an OCI image to
`container_id` and calling `engibench.utils.container.run()`. MTO2D uses the
same mounted-work-directory pattern in `model/runner.py`.

This directory currently builds a **local compatibility image** from the
legacy `MTO_GEN.sif`. It exists to reproduce the historical solver before a
clean image is rebuilt from source. The converted image is Linux/amd64,
inherits an unreproducible CentOS 7 filesystem, and contains code whose
redistribution rights have not yet been established. Do not push it to a
registry.

## Build the local parity image

Docker must be running. On Apple Silicon, Docker Desktop must have amd64
emulation enabled.

```bash
./engibench/problems/mto2d/model/runtime/convert_sif.sh \
  /path/to/MTO_GEN.sif \
  engibench-mto2d:sif-parity
```

The converter verifies the exact retained image:

- size: `1,670,721,536` bytes;
- SHA-256:
  `d53c0b6f8ec566b0d165be485efefde814e9f2af7e1e39f1ebc30a9a86ca62a6`;
- SquashFS system partition offset: `45,056` bytes.

It extracts the filesystem inside a temporary Linux Docker volume, imports it
as a Linux/amd64 image, and restores the OpenFOAM, OpenMPI, and PETSc
environment that Singularity previously supplied.

## Prepare the retained warm-start case

The case and solver are deliberately not copied into EngiBench while their
license is unresolved:

```bash
./engibench/problems/mto2d/model/runtime/prepare_case.sh \
  /path/to/warm-ready-2d.zip \
  /tmp/mto2d-case
```

The helper applies the small frozen-evaluation patch in this directory and
rebuilds `EXEC` inside the local parity image. The patched solver honors
`updateDesign = false`, recording objectives without running sensitivity/MMA
or writing the invalid `nan` field produced by the original zero-movement
update. Recreate previously prepared cases so that their executable contains
this fix.

Pass a different local runtime image as an optional third argument:

```bash
./engibench/problems/mto2d/model/runtime/prepare_case.sh \
  /path/to/warm-ready-2d.zip \
  /tmp/mto2d-case \
  my-local-mto2d-runtime:tag
```

## Smoke test

```bash
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker run --rm \
  engibench-mto2d:sif-parity \
  bash -c 'blockMesh -help >/dev/null && test -f /opt/libMMA_yu.so'
```

To evaluate through EngiBench, use:

```json
{
  "backend": "container",
  "container_image": "engibench-mto2d:sif-parity",
  "case_template": "/tmp/mto2d-case",
  "solver_executable": "../src_TF/EXEC",
  "mpi_cores": 1
}
```

Pass that file with `--solver-config`, or set
`ENGIBENCH_MTO2D_SOLVER_CONFIG` to its path. In the migration source
workspace, the direct demonstration also auto-loads
`../.artifacts/mto2d-docker.json` when `--simulate` is present:

```bash
python ./engibench/problems/mto2d/v0.py --simulate
```

An explicit `--solver-config` takes precedence over the environment-selected
file, which takes precedence over the auto-detected local file. Render-only
invocations never auto-load a solver configuration.

Set `DOCKER_DEFAULT_PLATFORM=linux/amd64` when invoking EngiBench on an ARM
host.

## Local validation record

The converted image was exercised through EngiBench on an ARM64 Docker
Desktop host with four emulated amd64 MPI ranks:

- `v0.py --simulate` completed for converted dataset row 0 with objectives
  `[22.2645, 67.6276]`;
- `simulate_verbose()` completed for the exact retained `app/200/gamma` with
  objectives `[9.47532, 70.805]`; and
- a one-iteration strict warm `optimize_verbose()` used `D0 = D1 = 63.1`,
  returned a finite `(400, 200)` design bounded by `[0, 1]`, and that returned
  design frozen-evaluated at `[9.55802, 40.7588]`;
- a two-iteration cold smoke run traversed the expected physical continuation
  endpoints and reconstructed a finite output design; and
- after applying `frozen-evaluation.patch`, the completed reference-campaign
  design reproduced strict objectives `[9.40088, 70.0166]` and legacy-profile
  objectives `[9.33296, 62.0621]`. Both runs skipped sensitivity/MMA, wrote
  finite unchanged gamma fields, and passed the runner's frozen-field
  validation.

These are runtime-validation values, not a new canonical numerical reference.
The two-step cold run is intentionally too abrupt to assess optimization
quality; meaningful cold validation requires the full, smoother continuation.
The retained scalar history `[9.45825, 62.2588]` also uses the legacy cold
physics, while strict evaluation uses the final RAMP parameters.

## Publishable successor

The parity image must eventually be replaced by a pinned source build:

1. OpenFOAM-5.x commit
   `7f7d351b741bf6406366a043cac98de56d2d44dd`;
2. ThirdParty-5.x commit
   `a807587a7babd4d03b62794b26e5ef4105301416`;
3. OpenMPI 4.0.4 and PETSc 3.12.5 commit
   `30e86313c4ca9b4414d2a7d1611388a22af15427`;
4. swak4Foam commit
   `9d8f12af95f5b6496ce28849d013b729a6a73abf`;
5. the recovered modified MMA source from the preserved `MTO-Scripts` Git
   history;
6. the warm-start MTO2D solver; and
7. an amd64 reference test, followed by a separately validated arm64 build.

Only after licensing and numerical parity are resolved should that OCI image
be published and assigned to `MTO2D.container_id`.
