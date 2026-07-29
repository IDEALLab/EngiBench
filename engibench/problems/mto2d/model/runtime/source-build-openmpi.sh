#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=source-build-common.sh
source /usr/local/lib/mto2d/source-build-common.sh
require_variables \
    OPENMPI_VERSION \
    OPENMPI_ARCHIVE_URL \
    OPENMPI_ARCHIVE_SHA256 \
    SOURCE_BUILD_JOBS

mkdir -p /usr/local/src
openmpi_archive="/usr/local/src/openmpi-${OPENMPI_VERSION}.tar.gz"
curl --fail --location --retry 5 --retry-delay 2 \
    "$OPENMPI_ARCHIVE_URL" \
    --output "$openmpi_archive"
printf '%s  %s\n' "$OPENMPI_ARCHIVE_SHA256" "$openmpi_archive" | sha256sum --check -
tar -xzf "$openmpi_archive" -C /usr/local/src
openmpi_source="/usr/local/src/openmpi-${OPENMPI_VERSION}"
openmpi_prefix="/opt/openmpi-${OPENMPI_VERSION}"
(
    cd "$openmpi_source"
    ./configure \
        "--prefix=$openmpi_prefix" \
        --disable-static \
        --enable-shared \
        --disable-mpi-fortran \
        --with-hwloc=internal \
        --with-libevent=internal
    make -j "$SOURCE_BUILD_JOBS"
    make install
)
test -x "$openmpi_prefix/bin/mpirun"
