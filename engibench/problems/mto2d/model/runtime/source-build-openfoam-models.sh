#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=source-build-common.sh
source /usr/local/lib/mto2d/source-build-common.sh
# shellcheck source=source-openfoam-environment.sh
source /usr/local/lib/mto2d/source-openfoam-environment.sh
require_variables SOURCE_BUILD_JOBS

activate_openfoam_environment
# OpenFOAM's bashrc selects the decomposition backend but does not export the
# third-party Scotch include/library paths needed by the standalone wmake
# calls below. The upstream parallel/Allwmake script sources this same file.
# shellcheck source=/dev/null
source "$WM_PROJECT_DIR/etc/config.sh/scotch"
export WM_NCOMPPROCS="$SOURCE_BUILD_JOBS"

(
    cd "$WM_PROJECT_DIR/src"
    wmake lagrangian/basic
    wmake lagrangian/distributionModels
    wmake genericPatchFields
    wmake conversion
    wmake mesh/extrudeModel
    wmake dynamicMesh
    wmake sampling
    wmake dynamicFvMesh
    wmake topoChangerFvMesh

    # Build the decomposition implementations linked by decomposePar. The
    # runner selects "simple", while PT-Scotch remains available to match the
    # historical runtime's command surface.
    (
        cd parallel/decompose
        wmakeLnInclude decompositionMethods
        wmake scotchDecomp
        wmake ptscotchDecomp
        wmake decompositionMethods
        wmake decompose
    )
    (
        cd parallel/reconstruct
        wmake reconstruct
    )
    wmake parallel/distributed

    wmake ODE
    wmake randomProcesses
    transportModels/Allwmake -j "$SOURCE_BUILD_JOBS"
    thermophysicalModels/Allwmake -j "$SOURCE_BUILD_JOBS"
    TurbulenceModels/Allwmake -j "$SOURCE_BUILD_JOBS"
    wmake combustionModels
    regionModels/Allwmake -j "$SOURCE_BUILD_JOBS"
    lagrangian/Allwmake -j "$SOURCE_BUILD_JOBS"
    mesh/Allwmake -j "$SOURCE_BUILD_JOBS"
    renumber/Allwmake -j "$SOURCE_BUILD_JOBS"
    fvAgglomerationMethods/Allwmake -j "$SOURCE_BUILD_JOBS"
    wmake fvMotionSolver
    wmake engine
    wmake fvOptions
    wmake regionCoupled
    functionObjects/Allwmake -j "$SOURCE_BUILD_JOBS"
    wmake sixDoFRigidBodyMotion
    wmake rigidBodyDynamics
    wmake rigidBodyMeshMotion
    wmake waves
)

test -f "$FOAM_LIBBIN/libincompressibleTransportModels.so"
test -f "$FOAM_LIBBIN/libturbulenceModels.so"
test -f "$FOAM_LIBBIN/libincompressibleTurbulenceModels.so"
test -f "$FOAM_LIBBIN/libfvOptions.so"
test -f "$FOAM_LIBBIN/libdecompose.so"
test -f "$FOAM_LIBBIN/libreconstruct.so"
