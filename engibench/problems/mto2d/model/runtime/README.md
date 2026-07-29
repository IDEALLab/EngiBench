# MTO2D container runtime

EngiBench already runs Docker-backed problems by assigning an OCI image to
`container_id` and calling `engibench.utils.container.run()`. MTO2D uses the
same mounted-work-directory pattern in `model/runner.py`.

This directory has two local image tracks:

1. `Dockerfile.source` builds the runtime and solver from pinned source inputs.
   It is the publishable-image candidate, but it remains local-only until its
   licensing and numerical-parity gates are complete.
2. `Dockerfile.extractor` converts the legacy `MTO_GEN.sif` into an opaque
   compatibility image. It exists only as a historical parity oracle.

Neither image may be pushed while the MTO2D solver, exact case, and modified
MMA redistribution rights remain unresolved.

## Build the source-image candidate

The source build consumes a caller-supplied warm-ready case archive and the
preserved `MTO-Scripts` Git checkout. EngiBench does not contain or silently
download either unresolved MTO2D input.

```bash
./engibench/problems/mto2d/model/runtime/build_source_image.sh \
  /path/to/warm-ready-2d.zip \
  /path/to/MTO-Scripts \
  engibench-mto2d:source-local
```

Set `MTO2D_BUILD_JOBS` to control compilation parallelism; it defaults to
eight. A native Linux/amd64 builder is strongly preferred: building the
amd64 toolchain under Apple Silicon emulation is reproducible but much slower.
BuildKit automatically resumes from the selected builder's local cache.
For CI or a different builder, import/export a persistent cache without
changing the compiled inputs:

```bash
export MTO2D_BUILDX_CACHE_FROM='type=registry,ref=ghcr.io/ideallab/engibench-mto2d:buildcache'
export MTO2D_BUILDX_CACHE_TO='type=registry,ref=ghcr.io/ideallab/engibench-mto2d:buildcache,mode=max'
```

The cache reference is only an acceleration input; release validation still
checks the resulting image and deterministic solver outputs. Do not use
`--no-cache`, prune the active builder, change compiler flags, or change MPI
rank count when reproducing an accepted image.

`APPROVED_IMAGE_LICENSES` in `source-pins.env` remains `NOASSERTION`
for local development. A release requires that versioned value to be replaced
by the exact SPDX expression approved by the licensing review; an environment
override cannot bypass that pin. The build:

- checks out OpenFOAM, its third-party tree, PETSc, and swak4Foam at the full commits in
  `source-pins.env`;
- verifies the OpenMPI 4.0.4 archive SHA-256 before compiling it;
- caches OpenMPI, PETSc, OpenFOAM, and swak4Foam as separate build layers so
  a later dependency failure does not discard earlier compilation work;
- builds the OpenFOAM libraries plus only `blockMesh`, `decomposePar`, and
  `reconstructPar`, rather than bundling unrelated solver applications;
- fixes each run to OpenFOAM's deterministic `simple` decomposition method,
  while rebuilding the historical PT-Scotch ABI from exact upstream Scotch
  6.0.8 sources (the legacy image stored them under a `6.0.3` directory name);
- rejects any warm-ready archive whose SHA-256 differs from the retained
  release input pinned in `source-pins.env`;
- extracts the modified MMA implementation from the pinned historical
  `MTO-Scripts` commit and records its hashes;
- applies both EngiBench solver patches and deletes every inherited `EXEC`
  before rebuilding the solver;
- removes histories, decomposed output, and editor backups before the case
  enters any OCI layer;
- records Git revisions, input hashes, and installed Debian package versions
  under `/opt/mto2d`, including checksums of the rebuilt solver and MMA
  library; and
- installs a pristine, built template at `/opt/mto2d/case-template`;
- installs the snapshot-pinned OpenSSH client required by OpenMPI singleton
  initialization;
- activates OpenFOAM from a static, build-validated environment instead of
  reparsing its interactive bash setup on every container launch; and
- stores a hash-verified prebuilt mesh outside the exported case template.
  The runner reuses it only when `blockMeshDict` matches exactly and otherwise
  falls back to `blockMesh`.

The base linux/amd64 image is digest-pinned, Git and archive inputs are
content-pinned, and Ubuntu packages resolve through the dated Canonical
snapshot in `source-pins.env`. The only bootstrap exception is the exact
focal-release CA/OpenSSL package set needed to establish TLS with the snapshot
service. Exact maintained versions from the dated snapshot replace that active
package set before any dependency source is fetched; the original bootstrap
bytes remain in the earlier OCI layer, as with an ordinary layered package
upgrade. The image stores the resulting `dpkg-manifest.txt`. These controls
make the source build repeatable; they do not promise byte-identical OCI
layers because upstream build tools may embed nondeterministic metadata. See
Canonical's
[Ubuntu snapshot-service documentation](https://documentation.ubuntu.com/server/how-to/software/snapshot-service/)
for the dated archive mechanism.

The image exposes the case-template protocol expected by the EngiBench runner:

```bash
mkdir -p /tmp/mto2d-export
docker run --rm --platform linux/amd64 \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src=/tmp/mto2d-export,dst=/work \
  engibench-mto2d:source-local \
  mto2d-export-case /work/case
```

`mto2d-export-case` requires an empty destination and copies a writable
template containing the source-built `src_TF/EXEC` and runtime capability
marker. Run the image's structural check with:

```bash
docker run --rm --platform linux/amd64 \
  engibench-mto2d:source-local \
  mto2d-source-smoke
```

That smoke test checks the serial/MPI command surface, shared-library
resolution, the built executable, and case export. It deliberately does not
claim numerical parity.

From the EngiBench checkout root, run the deterministic shared-suite reference
through the image with:

```bash
DOCKER_DEFAULT_PLATFORM=linux/amd64 \
python engibench/problems/mto2d/model/runtime/verify_source_reference.py \
  --image engibench-mto2d:source-local \
  --dataset dataset_output/mto_2d_exact_source_v0
```

The verifier pins train index `2010`, source case `6799`, the class-default
conditions, and the native design bytes. It requires the parsed float64
objectives to equal the committed `q=0.01` values `[13.8912, 63.8033]`
exactly, then checks every deterministic scalar-history hash and the complete
86,400-cell serialized gamma field against
`source-reference-golden.json`. That golden records the SHA-256 of the
original SIF, converted oracle image, and prepared oracle executable.

Before publication, compare the source-built candidate with the local parity
oracle:

```bash
DOCKER_DEFAULT_PLATFORM=linux/amd64 \
python engibench/problems/mto2d/model/runtime/verify_source_reference.py \
  --image engibench-mto2d:source-local \
  --dataset dataset_output/mto_2d_exact_source_v0 \
  --oracle-image engibench-mto2d:sif-parity \
  --oracle-case-template /path/to/prepared/warm-ready-2d
```

This oracle run compares the raw deterministic scalar histories and final
gamma bytes as well as their parsed values. The retained solver does not
serialize its final `T`, `U`, or `p` fields. The gate deliberately excludes
`Time.txt`, the adjoint scratch fields that frozen evaluation now bypasses,
and host/timestamp-bearing log lines. Repeat the structural and frozen
q=0.01 checks against the pulled immutable digest on fresh amd64 Linux, then
pin that digest and its package manifest. Short/full optimization comparisons
are optional manual validation only; optimization is not part of the shared
EngiBench suite.

### Recommended registry

Host the release image in GitHub Container Registry as
`ghcr.io/ideallab/engibench-mto2d`. This keeps the image attached to the
EngiBench organization/repository, supports anonymous pulls for public
packages, and exposes immutable OCI digests to `container_id`.

After rights and numerical parity are approved:

```bash
# CR_PAT is a classic GitHub token with write:packages, authorized for
# IDEALLab organization SSO when applicable.
printf '%s' "${CR_PAT:?set CR_PAT first}" \
  | docker login ghcr.io --username YOUR_GITHUB_USERNAME --password-stdin

# This dry run refuses dirty, unlicensed, mislabelled, or non-amd64 images.
./engibench/problems/mto2d/model/runtime/publish_source_image.sh

# This is the only command in the helper that changes the registry.
./engibench/problems/mto2d/model/runtime/publish_source_image.sh \
  --confirm-redistribution-rights \
  --confirm-reference \
  --reference-dataset dataset_output/mto_2d_exact_source_v0 \
  --push
```

The guarded helper runs the structural smoke test and the deterministic
q=0.01 simulation against the exact local image ID. It also requires the
licensed expression pinned in source control, a clean build and checkout, and
a revision publicly reachable from `IDEALLab/EngiBench`. It refuses to change
an existing commit-scoped tag, but can resume a partial release when that tag
already contains the same validated image. It publishes `v0-$REV` and the
moving `v0` alias and verifies that both resolve to the same remote digest and
local image config. Only the resulting `@sha256:...` reference is immutable.

The first GHCR publication is private. In the IDEALLab package settings,
explicitly change `engibench-mto2d` to public only after the release checks
pass; GitHub does not allow a public package to be made private again. Copy
the helper's printed digest and verify it from an unauthenticated environment:

```bash
PIN='ghcr.io/ideallab/engibench-mto2d@sha256:<digest printed by the helper>'
docker pull "$PIN"
```

Then replace the prospective `:v0` value in `MTO2D.container_id` with that
exact digest. Keep `v0` and `v0-$REV` as human-readable aliases only.
Publishing from a GitHub Actions workflow is preferred once all licensed
source inputs are available to that workflow: it links the package to the
repository automatically and can attach build-provenance and SBOM
attestations.

See GitHub's documentation for
[the Container registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
and
[package visibility](https://docs.github.com/en/packages/learn-github-packages/configuring-a-packages-access-control-and-visibility).

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

The helper applies the frozen-evaluation and named optimization-schedule
patches in this directory and rebuilds `EXEC` inside the local parity image.
The patched solver honors `updateDesign = false`, recording objectives without
running the adjoint, sensitivity, or MMA stages, or writing the invalid `nan`
field produced by the original zero-movement update. It also restores the
exact source timing for `optimization_schedule="legacy"` while keeping the
warm-ready interpolation path for `optimization_schedule="strict"`. A
capability marker is written only after a successful rebuild; the Python
runner refuses either optimization schedule without it. Recreate previously
prepared cases so their executable and marker contain both fixes.

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

The source-built `source-fast` image passed the one-rank exact oracle gate on
an ARM64 Docker Desktop host. It reproduced objective float bits, all six
deterministic scalar-history files, and the serialized gamma field byte for
byte. All six files in the image-provided mesh also matched fresh
`blockMesh` output. On that emulated host the solver-reported time was 179 s
versus 233 s for the adjoint-inclusive SIF oracle (about 23% faster), and a
warm no-op container launch dropped from a median of about 2.1 s to 0.4 s.
These timings are machine-specific; the byte comparison is the acceptance
criterion.

The converted image was exercised through EngiBench on an ARM64 Docker
Desktop host with four emulated amd64 MPI ranks:

- `v0.py --simulate` completed for converted dataset row 0 with objectives
  `[22.2645, 67.6276]`;
- `simulate_verbose()` completed for the exact retained `app/200/gamma` with
  objectives `[9.47532, 70.805]`; and
- a one-iteration strict warm `optimize_verbose()` used `D0 = D1 = 63.1`,
  returned a finite `(400, 200)` design bounded by `[0, 1]`, and that returned
  design frozen-evaluated at `[9.55802, 40.7588]`;
- a two-iteration strict cold smoke run traversed the configured physical
  continuation endpoints and reconstructed a finite output design;
- the rebuilt v2 runtime completed a three-iteration legacy cold prefix with
  recorded `(alphaMax, qu, Heaviside)` values
  `(2500, .005, .1)`, `(3214.29, .005, .1)`, and
  `(3571.43, .005, .4)`, confirming the source schedule's pre-projection
  Heaviside timing; and
- after applying the runtime patches, the completed reference-campaign
  design reproduced strict objectives `[9.40088, 70.0166]` and legacy-profile
  objectives `[9.33296, 62.0621]`. Both runs skipped sensitivity/MMA, wrote
  finite unchanged gamma fields, and passed the runner's frozen-field
  validation.

These are runtime-validation values, not a new canonical numerical reference.
The two-step cold run is intentionally too abrupt to assess optimization
quality; meaningful cold validation requires the full, smoother continuation.
The retained scalar history `[9.45825, 62.2588]` also uses the legacy cold
physics, while strict evaluation uses the final RAMP parameters.

## Source pins and publication gates

The source-image candidate currently pins:

1. OpenFOAM-5.x commit
   `7f7d351b741bf6406366a043cac98de56d2d44dd`;
2. ThirdParty-5.x commit
   `a807587a7babd4d03b62794b26e5ef4105301416`;
3. OpenMPI 4.0.4 and PETSc 3.12.5 commit
   `30e86313c4ca9b4414d2a7d1611388a22af15427`;
4. the `openfoam-extend-swak4Foam-dev` mirror commit
   `9d8f12af95f5b6496ce28849d013b729a6a73abf`;
5. the recovered modified MMA source from the preserved `MTO-Scripts` Git
   history;
6. the warm-start MTO2D solver; and
7. the caller-supplied case archive and patches by SHA-256 at build time.

The structural build test is necessary but insufficient. Only after licensing
and fresh amd64 numerical parity are resolved should an OCI digest be
published and assigned to `MTO2D.container_id`. The v0 policy is amd64-only;
Linux ARM remains skipped by the shared suite until a native multi-architecture
image is available.
