#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  build_source_image.sh /path/to/warm-ready-2d.zip /path/to/MTO-Scripts [image]

Builds a local linux/amd64 image entirely from pinned dependency sources plus
the caller-supplied MTO2D case and recovered MMA source. The output defaults
to engibench-mto2d:source-local.

This command does not grant redistribution rights. Do not push the resulting
image until the MTO2D solver, exact case, and MMA licensing review is complete.
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
    usage >&2
    exit 2
fi

archive_input=$1
mto_scripts_input=$2
output_image=${3:-engibench-mto2d:source-local}
runtime_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# shellcheck source=source-pins.env
source "$runtime_dir/source-pins.env"

absolute_file() {
    local input=$1
    local directory
    directory=$(cd "$(dirname "$input")" && pwd)
    printf '%s/%s\n' "$directory" "$(basename "$input")"
}

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

archive=$(absolute_file "$archive_input")
mto_scripts=$(cd "$mto_scripts_input" && pwd)
engibench_root=$(git -C "$runtime_dir" rev-parse --show-toplevel)
engibench_revision=$(git -C "$engibench_root" rev-parse HEAD)
engibench_tree_state=clean
if [[ -n "$(git -C "$engibench_root" status --porcelain --untracked-files=normal)" ]]; then
    engibench_tree_state=dirty
fi
image_licenses=$APPROVED_IMAGE_LICENSES
if [[ -n "${MTO2D_IMAGE_LICENSES+x}" && "$MTO2D_IMAGE_LICENSES" != "$image_licenses" ]]; then
    echo "MTO2D_IMAGE_LICENSES must match APPROVED_IMAGE_LICENSES in source-pins.env." >&2
    exit 1
fi

if [[ -z "$image_licenses" || "$image_licenses" == *$'\n'* || "$image_licenses" == *$'\r'* ]]; then
    echo "APPROVED_IMAGE_LICENSES must be a non-empty, single-line SPDX expression." >&2
    exit 1
fi

if [[ ! -f "$archive" ]]; then
    echo "Case archive does not exist: $archive" >&2
    exit 1
fi
case_archive_sha256=$(sha256_file "$archive")
if [[ "$case_archive_sha256" != "$MTO2D_CASE_ARCHIVE_SHA256" ]]; then
    echo "Case archive SHA-256 mismatch." >&2
    echo "Expected: $MTO2D_CASE_ARCHIVE_SHA256" >&2
    echo "Actual:   $case_archive_sha256" >&2
    exit 1
fi
if ! git -C "$mto_scripts" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "MTO-Scripts path is not a Git checkout: $mto_scripts" >&2
    exit 1
fi
if [[ "$OPENMPI_VERSION" != "4.0.4" ]]; then
    echo "This v0 recipe fixes all runtime MPI paths to OpenMPI 4.0.4; got: $OPENMPI_VERSION" >&2
    exit 1
fi
if ! command -v docker >/dev/null 2>&1 || ! docker buildx version >/dev/null 2>&1; then
    echo "Docker with Buildx is required." >&2
    exit 1
fi
if docker image inspect "$output_image" >/dev/null 2>&1; then
    echo "Output image already exists; refusing to replace it: $output_image" >&2
    exit 1
fi

for object in \
    "$MMA_SOURCE_COMMIT:$MMA_IMPLEMENTATION_PATH" \
    "$MMA_SOURCE_COMMIT:$MMA_HEADER_PATH"; do
    if ! git -C "$mto_scripts" cat-file -e "$object"; then
        echo "MTO-Scripts does not contain required historical object: $object" >&2
        exit 1
    fi
done

temporary_root=$(mktemp -d "${TMPDIR:-/tmp}/engibench-mto2d-source.XXXXXX")
case_context="$temporary_root/case"
mma_context="$temporary_root/mma"

cleanup() {
    if [[ -n "${temporary_root:-}" && "$temporary_root" == */engibench-mto2d-source.* ]]; then
        rm -rf -- "$temporary_root"
    fi
}
trap cleanup EXIT

mkdir -p "$case_context" "$mma_context"
unzip -q "$archive" -d "$case_context"
if [[ ! -d "$case_context/app" || ! -d "$case_context/src_TF" ]]; then
    echo "Case archive must extract app/ and src_TF/ at its root." >&2
    exit 1
fi

patch --batch --forward --strip=1 --directory="$case_context" \
    <"$runtime_dir/frozen-evaluation.patch"
patch --batch --forward --strip=1 --directory="$case_context" \
    <"$runtime_dir/optimization-schedules.patch"

# A source image must not inherit the legacy executable or stale wmake output.
rm -f -- "$case_context/src_TF/EXEC"
if [[ -d "$case_context/src_TF/Make" ]]; then
    find "$case_context/src_TF/Make" \
        -mindepth 1 \
        -maxdepth 1 \
        ! -name files \
        ! -name options \
        -exec rm -rf -- {} +
fi

# Remove generated histories, decomposed output, and editor backups before the
# case enters a Docker layer. Deleting them inside the image would leave their
# bytes recoverable from the earlier COPY layer.
find "$case_context/app" -maxdepth 1 -type f \
    \( -name 'meanT.txt' -o -name 'Disspower.txt' -o -name 'Voluse.txt' \
    -o -name 'Time.txt' -o -name 'aMax.txt' -o -name 'qu.txt' \
    -o -name 'HEAV.txt' \) -delete
find "$case_context/app" -maxdepth 1 -mindepth 1 -type d \
    \( -name 'processor[0-9]*' -o -name '[1-9]*' \) -exec rm -rf -- {} +
find "$case_context" -type f -name '*~' -delete

git -C "$mto_scripts" show \
    "$MMA_SOURCE_COMMIT:$MMA_IMPLEMENTATION_PATH" >"$mma_context/MMA.c"
git -C "$mto_scripts" show \
    "$MMA_SOURCE_COMMIT:$MMA_HEADER_PATH" >"$mma_context/MMA.h"

mma_implementation_sha256=$(sha256_file "$mma_context/MMA.c")
mma_header_sha256=$(sha256_file "$mma_context/MMA.h")
frozen_patch_sha256=$(sha256_file "$runtime_dir/frozen-evaluation.patch")
schedule_patch_sha256=$(sha256_file "$runtime_dir/optimization-schedules.patch")
build_jobs=${MTO2D_BUILD_JOBS:-8}

if [[ ! "$build_jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "MTO2D_BUILD_JOBS must be a positive integer; got: $build_jobs" >&2
    exit 1
fi

buildx_cache_args=()
if [[ -n "${MTO2D_BUILDX_CACHE_FROM:-}" ]]; then
    buildx_cache_args+=(--cache-from "$MTO2D_BUILDX_CACHE_FROM")
fi
if [[ -n "${MTO2D_BUILDX_CACHE_TO:-}" ]]; then
    buildx_cache_args+=(--cache-to "$MTO2D_BUILDX_CACHE_TO")
fi

docker buildx build \
    --platform linux/amd64 \
    --load \
    "${buildx_cache_args[@]}" \
    --file "$runtime_dir/Dockerfile.source" \
    --build-context "mto2d_case=$case_context" \
    --build-context "mma_source=$mma_context" \
    --build-arg "BASE_IMAGE=$BASE_IMAGE" \
    --build-arg "BASE_IMAGE_NAME=$BASE_IMAGE_NAME" \
    --build-arg "BASE_IMAGE_DIGEST=$BASE_IMAGE_DIGEST" \
    --build-arg "APT_SNAPSHOT=$APT_SNAPSHOT" \
    --build-arg "CA_CERTIFICATES_VERSION=$CA_CERTIFICATES_VERSION" \
    --build-arg "LIBSSL1_1_VERSION=$LIBSSL1_1_VERSION" \
    --build-arg "OPENSSL_VERSION=$OPENSSL_VERSION" \
    --build-arg "OPENSSH_CLIENT_VERSION=$OPENSSH_CLIENT_VERSION" \
    --build-arg "OPENFOAM_REPOSITORY=$OPENFOAM_REPOSITORY" \
    --build-arg "OPENFOAM_COMMIT=$OPENFOAM_COMMIT" \
    --build-arg "THIRD_PARTY_REPOSITORY=$THIRD_PARTY_REPOSITORY" \
    --build-arg "THIRD_PARTY_COMMIT=$THIRD_PARTY_COMMIT" \
    --build-arg "SCOTCH_REPOSITORY=$SCOTCH_REPOSITORY" \
    --build-arg "SCOTCH_COMMIT=$SCOTCH_COMMIT" \
    --build-arg "SCOTCH_MAKEFILE_SHA256=$SCOTCH_MAKEFILE_SHA256" \
    --build-arg "SCOTCH_DGRAPH_HALO_SHA256=$SCOTCH_DGRAPH_HALO_SHA256" \
    --build-arg "OPENMPI_VERSION=$OPENMPI_VERSION" \
    --build-arg "OPENMPI_ARCHIVE_URL=$OPENMPI_ARCHIVE_URL" \
    --build-arg "OPENMPI_ARCHIVE_SHA256=$OPENMPI_ARCHIVE_SHA256" \
    --build-arg "PETSC_REPOSITORY=$PETSC_REPOSITORY" \
    --build-arg "PETSC_COMMIT=$PETSC_COMMIT" \
    --build-arg "SWAK4FOAM_REPOSITORY=$SWAK4FOAM_REPOSITORY" \
    --build-arg "SWAK4FOAM_COMMIT=$SWAK4FOAM_COMMIT" \
    --build-arg "MMA_SOURCE_COMMIT=$MMA_SOURCE_COMMIT" \
    --build-arg "MMA_IMPLEMENTATION_SHA256=$mma_implementation_sha256" \
    --build-arg "MMA_HEADER_SHA256=$mma_header_sha256" \
    --build-arg "MTO2D_CASE_ARCHIVE_SHA256=$case_archive_sha256" \
    --build-arg "FROZEN_PATCH_SHA256=$frozen_patch_sha256" \
    --build-arg "SCHEDULE_PATCH_SHA256=$schedule_patch_sha256" \
    --build-arg "ENGIBENCH_REVISION=$engibench_revision" \
    --build-arg "ENGIBENCH_TREE_STATE=$engibench_tree_state" \
    --build-arg "IMAGE_LICENSES=$image_licenses" \
    --build-arg "SOURCE_BUILD_JOBS=$build_jobs" \
    --tag "$output_image" \
    "$runtime_dir"

echo "Built local source image: $output_image"
echo "Run its structural smoke test with:"
echo "  docker run --rm --platform linux/amd64 $output_image mto2d-source-smoke"
echo "Do not push this image until redistribution rights and numerical parity are approved."
