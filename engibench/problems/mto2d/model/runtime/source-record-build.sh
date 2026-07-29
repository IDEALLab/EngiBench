#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=source-build-common.sh
source /usr/local/lib/mto2d/source-build-common.sh
# shellcheck source=source-openfoam-environment.sh
source /usr/local/lib/mto2d/source-openfoam-environment.sh
require_variables \
    BASE_IMAGE \
    APT_SNAPSHOT \
    CA_CERTIFICATES_VERSION \
    LIBSSL1_1_VERSION \
    OPENSSL_VERSION \
    OPENSSH_CLIENT_VERSION \
    OPENFOAM_REPOSITORY \
    OPENFOAM_COMMIT \
    THIRD_PARTY_REPOSITORY \
    THIRD_PARTY_COMMIT \
    SCOTCH_REPOSITORY \
    SCOTCH_COMMIT \
    SCOTCH_MAKEFILE_SHA256 \
    SCOTCH_DGRAPH_HALO_SHA256 \
    OPENMPI_VERSION \
    OPENMPI_ARCHIVE_URL \
    OPENMPI_ARCHIVE_SHA256 \
    PETSC_REPOSITORY \
    PETSC_COMMIT \
    SWAK4FOAM_REPOSITORY \
    SWAK4FOAM_COMMIT

activate_openfoam_environment
test -x "$MPI_DIR/bin/mpirun"
test -f "$PETSC_DIR/$PETSC_ARCH/lib/libpetsc.so"
test -x "$FOAM_APPBIN/blockMesh"
test -f "$FOAM_USER_LIBBIN/libswak4FoamParsers.so"
test -f "$FOAM_USER_LIBBIN/libgroovyBC.so"

{
    printf 'base_image=%s\n' "$BASE_IMAGE"
    printf 'ubuntu_snapshot=%s\n' "$APT_SNAPSHOT"
    printf 'ca-certificates=%s\n' "$CA_CERTIFICATES_VERSION"
    printf 'libssl1.1=%s\n' "$LIBSSL1_1_VERSION"
    printf 'openssl=%s\n' "$OPENSSL_VERSION"
    printf 'openssh-client=%s\n' "$OPENSSH_CLIENT_VERSION"
    printf 'openfoam=%s@%s\n' "$OPENFOAM_REPOSITORY" "$OPENFOAM_COMMIT"
    printf 'third_party=%s@%s\n' "$THIRD_PARTY_REPOSITORY" "$THIRD_PARTY_COMMIT"
    printf 'scotch=%s@%s\n' "$SCOTCH_REPOSITORY" "$SCOTCH_COMMIT"
    printf 'scotch_src_makefile_sha256=%s\n' "$SCOTCH_MAKEFILE_SHA256"
    printf 'scotch_dgraph_halo_sha256=%s\n' "$SCOTCH_DGRAPH_HALO_SHA256"
    printf 'openmpi=%s sha256:%s\n' "$OPENMPI_ARCHIVE_URL" "$OPENMPI_ARCHIVE_SHA256"
    printf 'petsc=%s@%s\n' "$PETSC_REPOSITORY" "$PETSC_COMMIT"
    printf 'swak4foam=%s@%s\n' "$SWAK4FOAM_REPOSITORY" "$SWAK4FOAM_COMMIT"
} >/opt/mto2d/source-revisions.txt
dpkg-query --show --showformat='${Package}\t${Version}\n' \
    | LC_ALL=C sort >/opt/mto2d/dpkg-manifest.txt
