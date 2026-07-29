# MTO2D runtime

MTO2D uses the same Docker-backed execution path as other containerized
EngiBench problems. The release is Linux/AMD64 and is referenced by immutable
digest from `MTO2D.container_id`.

## Pull and smoke test

```bash
docker pull ghcr.io/ideallab/engibench-mto2d@sha256:<pinned digest>
docker run --rm --platform linux/amd64 \
  ghcr.io/ideallab/engibench-mto2d@sha256:<pinned digest> \
  mto2d-source-smoke
```

The image exports a pristine writable case through `mto2d-export-case`.
`model/runner.py` uses that protocol automatically.

## Maintainer build

`Dockerfile.source` builds OpenMPI, PETSc, OpenFOAM 5, swak4Foam, the MTO2D
solver, and its MMA library from the revisions and hashes in
`source-pins.env`. The case archive and historical MMA source come from the
maintainer `IDEALLab/MTO-Scripts` checkout:

```bash
./engibench/problems/mto2d/model/runtime/build_source_image.sh \
  /path/to/MTO-Scripts/warm_start/2D/templates/warm-ready-2d.zip \
  /path/to/MTO-Scripts \
  engibench-mto2d:source-local
```

`MTO2D_BUILD_JOBS` controls compilation parallelism and defaults to eight.
Optional BuildKit registry caches can be configured with
`MTO2D_BUILDX_CACHE_FROM` and `MTO2D_BUILDX_CACHE_TO`. A native AMD64 builder
is much faster than ARM emulation.

The build:

- pins the AMD64 Ubuntu base image, dependency commits, archive hashes, and
  snapshot package versions;
- rebuilds the solver after applying the frozen-evaluation and continuation
  patches;
- embeds a pristine case and a hash-verified prebuilt mesh;
- records build inputs and installed packages under `/opt/mto2d`; and
- labels the image with source revision, tree state, and
  `GPL-3.0-or-later`.

The recipe is maintainer-reproducible because its MTO2D inputs live in the
maintainer source repository; all third-party dependency pins are recorded
here.

## Exact reference

The release gate uses train row 2010 from
`IDEALLab/mto_2d_v0`, conditions `(-0.074, 63.1, 0.61)`, and `q=0.01`:

```bash
DOCKER_DEFAULT_PLATFORM=linux/amd64 \
python engibench/problems/mto2d/model/runtime/verify_source_reference.py \
  --image engibench-mto2d:source-local
```

The verifier requires objectives `[13.8912, 63.8033]`, unchanged design bytes,
and exact hashes for six scalar histories and the final 86,400-cell gamma
field. `source-reference-golden.json` records the trusted oracle.
Optimization is not part of this release gate.

## Publish

Authenticate Docker to GHCR with a classic token carrying `write:packages`,
then run:

```bash
./engibench/problems/mto2d/model/runtime/publish_source_image.sh \
  --image engibench-mto2d:source-local \
  --confirm-redistribution-rights \
  --confirm-reference \
  --push
```

The helper refuses dirty, unlicensed, non-AMD64, mismatched-revision, or
numerically different images. It publishes `v0-$REV` and `v0`, verifies their
manifest/config digests, and prints the immutable digest to pin in EngiBench.
The GHCR package must be public before anonymous integration testing.
