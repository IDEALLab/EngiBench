#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_SIZE=1670721536
readonly EXPECTED_SHA256=d53c0b6f8ec566b0d165be485efefde814e9f2af7e1e39f1ebc30a9a86ca62a6
readonly SQUASHFS_OFFSET=45056

usage() {
    echo "Usage: $0 /path/to/MTO_GEN.sif [output-image]"
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage >&2
    exit 2
fi

sif_path=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
output_image=${2:-engibench-mto2d:sif-parity}
runtime_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
extractor_image=engibench-mto2d:sif-extractor
rootfs_volume="engibench-mto2d-rootfs-$$"
tar_container="engibench-mto2d-tar-$$"

if [[ ! -f "$sif_path" ]]; then
    echo "SIF does not exist: $sif_path" >&2
    exit 1
fi
if docker image inspect "$output_image" >/dev/null 2>&1; then
    echo "Output image already exists; refusing to replace it: $output_image" >&2
    exit 1
fi

actual_size=$(wc -c <"$sif_path" | tr -d '[:space:]')
if [[ "$actual_size" != "$EXPECTED_SIZE" ]]; then
    echo "Unexpected SIF size: got $actual_size, expected $EXPECTED_SIZE" >&2
    exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
    actual_sha256=$(sha256sum "$sif_path" | awk '{print $1}')
else
    actual_sha256=$(shasum -a 256 "$sif_path" | awk '{print $1}')
fi
if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
    echo "Unexpected SIF SHA-256: got $actual_sha256" >&2
    exit 1
fi

cleanup() {
    docker container rm --force "$tar_container" >/dev/null 2>&1 || true
    docker volume rm --force "$rootfs_volume" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker buildx build --load \
    --file "$runtime_dir/Dockerfile.extractor" \
    --tag "$extractor_image" \
    "$runtime_dir"
docker volume create "$rootfs_volume" >/dev/null

docker run --rm \
    --mount "type=bind,src=$sif_path,dst=/input/MTO_GEN.sif,readonly" \
    --mount "type=volume,src=$rootfs_volume,dst=/rootfs" \
    "$extractor_image" \
    -c "unsquashfs -no-progress -offset $SQUASHFS_OFFSET -dest /rootfs /input/MTO_GEN.sif"

docker run --rm \
    --mount "type=bind,src=$runtime_dir/docker-entrypoint.sh,dst=/input/docker-entrypoint.sh,readonly" \
    --mount "type=volume,src=$rootfs_volume,dst=/rootfs" \
    "$extractor_image" \
    -c "mkdir -p /rootfs/usr/local/bin /rootfs/tmp \
        && chmod 1777 /rootfs/tmp \
        && install -m 0755 /input/docker-entrypoint.sh /rootfs/usr/local/bin/mto2d-entrypoint"

docker run --rm --name "$tar_container" \
    --mount "type=volume,src=$rootfs_volume,dst=/rootfs,readonly" \
    "$extractor_image" \
    -c "tar --numeric-owner --xattrs --acls -C /rootfs -cf - ." \
    | docker import \
        --platform linux/amd64 \
        --change 'ENV MPI_DIR=/opt/openmpi-4.0.4' \
        --change 'ENV PETSC_DIR=/opt/petsc' \
        --change 'ENV PETSC_ARCH=arch-linux2-c-opt' \
        --change 'ENTRYPOINT ["/usr/local/bin/mto2d-entrypoint"]' \
        --change 'CMD ["/bin/bash"]' \
        - \
        "$output_image"

echo "Built local parity image: $output_image"
echo "This image is an opaque compatibility oracle; do not publish it."
