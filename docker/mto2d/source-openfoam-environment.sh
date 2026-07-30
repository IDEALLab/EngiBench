#!/usr/bin/env bash

# Override the common helper for non-interactive source builds. OpenFOAM's
# bashrc can return a non-zero status solely because its optional completion
# hook is unavailable. Source it without errexit/nounset, then verify the
# environment explicitly.
activate_openfoam_environment() {
    activate_runtime_environment
    set +eu
    # shellcheck source=/dev/null
    source /opt/OpenFOAM/OpenFOAM-5.x/etc/bashrc
    set -eu
    : "${WM_PROJECT_DIR:?OpenFOAM bashrc did not set WM_PROJECT_DIR}"
    if ! command -v wmake >/dev/null 2>&1; then
        echo "OpenFOAM bashrc did not make wmake available." >&2
        exit 1
    fi
    export FOAM_USER_LIBBIN=/opt/mto2d/lib
    export FOAM_USER_APPBIN=/opt/mto2d/bin
    mkdir -p "$FOAM_USER_LIBBIN" "$FOAM_USER_APPBIN"
}
