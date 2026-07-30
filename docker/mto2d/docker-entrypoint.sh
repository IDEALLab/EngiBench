#!/bin/bash
set -e

export MPI_DIR=/opt/openmpi-4.0.4
export PETSC_DIR=/opt/petsc
export PETSC_ARCH=arch-linux2-c-opt
export PATH="${MPI_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="/opt:${PETSC_DIR}/${PETSC_ARCH}/lib:${MPI_DIR}/lib:${LD_LIBRARY_PATH:-}"

# The source image installs a static, build-validated OpenFOAM environment to
# avoid reparsing the interactive bashrc on every run. The SIF-derived parity
# oracle does not carry that helper and retains its historical fallback.
if [[ -r /usr/local/lib/mto2d/source-runtime-environment.sh ]]; then
    # shellcheck source=source-runtime-environment.sh
    source /usr/local/lib/mto2d/source-runtime-environment.sh
else
    # OpenFOAM's MPI setup calls mpicc while bashrc is sourced, hence MPI must
    # already be on PATH. Its optional completion hook may return non-zero.
    set +e
    # shellcheck source=/dev/null
    source /opt/OpenFOAM/OpenFOAM-5.x/etc/bashrc
    set -e
fi
if [[ -z "${WM_PROJECT_DIR:-}" ]] || ! command -v wmake >/dev/null 2>&1; then
    echo "Failed to activate the OpenFOAM runtime environment." >&2
    exit 1
fi

export PATH="${MPI_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="/opt:${PETSC_DIR}/${PETSC_ARCH}/lib:${MPI_DIR}/lib:${LD_LIBRARY_PATH:-}"

if [[ -d /opt/mto2d/lib ]]; then
    # The source image installs user-built libraries in a fixed location so
    # they do not depend on HOME or the numeric UID selected by EngiBench.
    export FOAM_USER_LIBBIN=/opt/mto2d/lib
    export FOAM_USER_APPBIN=/opt/mto2d/bin
    export PATH="${FOAM_USER_APPBIN}:${PATH}"
    export LD_LIBRARY_PATH="${FOAM_USER_LIBBIN}:${LD_LIBRARY_PATH}"
fi

exec "$@"
