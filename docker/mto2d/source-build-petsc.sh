#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=source-build-common.sh
source /usr/local/lib/mto2d/source-build-common.sh
require_variables PETSC_REPOSITORY PETSC_COMMIT SOURCE_BUILD_JOBS

# PETSc 3.12 imports distutils.sysconfig from its makefile generator. Ubuntu's
# minimal Python installation splits that standard-library module into this
# compatibility package.
apt-get update
apt-get install --yes --no-install-recommends python3-distutils
rm -rf /var/lib/apt/lists/*

activate_runtime_environment
fetch_commit "$PETSC_REPOSITORY" "$PETSC_COMMIT" "$PETSC_DIR"
(
    cd "$PETSC_DIR"
    python3 ./configure \
        "PETSC_ARCH=$PETSC_ARCH" \
        --with-debugging=0 \
        --with-shared-libraries=1 \
        "--with-mpi-dir=$MPI_DIR" \
        --with-fc=0 \
        '--with-blaslapack-lib=-llapack -lblas' \
        --with-x=0
    make -j "$SOURCE_BUILD_JOBS" "PETSC_DIR=$PETSC_DIR" "PETSC_ARCH=$PETSC_ARCH" all
)
test -f "$PETSC_DIR/$PETSC_ARCH/lib/libpetsc.so"
