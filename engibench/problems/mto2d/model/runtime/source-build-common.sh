#!/usr/bin/env bash
set -euo pipefail

require_variables() {
    local name
    for name in "$@"; do
        if [[ -z "${!name:-}" ]]; then
            echo "Missing source-build input: $name" >&2
            exit 1
        fi
    done
}

fetch_commit() {
    local repository=$1
    local commit=$2
    local destination=$3
    local attempt

    rm -rf -- "$destination"
    mkdir -p "$destination"
    git -C "$destination" init --quiet
    git -C "$destination" remote add origin "$repository"
    for attempt in 1 2 3 4 5; do
        if git -C "$destination" fetch --quiet --depth=1 origin "$commit"; then
            break
        fi
        if [[ "$attempt" == 5 ]]; then
            echo "Could not fetch $repository at $commit after $attempt attempts." >&2
            exit 1
        fi
        sleep "$((attempt * 2))"
    done
    git -C "$destination" -c advice.detachedHead=false checkout --quiet --detach "$commit"
    [[ "$(git -C "$destination" rev-parse HEAD)" == "$commit" ]]
}

activate_runtime_environment() {
    export MPI_DIR=/opt/openmpi-4.0.4
    export PETSC_DIR=/opt/petsc
    export PETSC_ARCH=arch-linux2-c-opt
    export PATH="$MPI_DIR/bin:$PATH"
    export LD_LIBRARY_PATH="$MPI_DIR/lib:$PETSC_DIR/$PETSC_ARCH/lib:${LD_LIBRARY_PATH:-}"
}

activate_openfoam_environment() {
    activate_runtime_environment
    set +u
    # shellcheck source=/dev/null
    source /opt/OpenFOAM/OpenFOAM-5.x/etc/bashrc
    set -u
    export FOAM_USER_LIBBIN=/opt/mto2d/lib
    export FOAM_USER_APPBIN=/opt/mto2d/bin
    mkdir -p "$FOAM_USER_LIBBIN" "$FOAM_USER_APPBIN"
}
