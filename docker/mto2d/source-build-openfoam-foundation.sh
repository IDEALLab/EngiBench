#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=source-build-common.sh
source /usr/local/lib/mto2d/source-build-common.sh
# shellcheck source=source-openfoam-environment.sh
source /usr/local/lib/mto2d/source-openfoam-environment.sh
require_variables SOURCE_BUILD_JOBS

activate_openfoam_environment
export WM_NCOMPPROCS="$SOURCE_BUILD_JOBS"

(
    cd "$WM_PROJECT_DIR/src"
    wmakePrintBuild -check || wrmo OpenFOAM/global/global.o 2>/dev/null
    wmakeLnInclude OpenFOAM
    wmakeLnInclude "OSspecific/${WM_OSTYPE:-POSIX}"
    Pstream/Allwmake -j "$SOURCE_BUILD_JOBS"
    "OSspecific/${WM_OSTYPE:-POSIX}/Allwmake" -j "$SOURCE_BUILD_JOBS"
    wmake OpenFOAM
    wmake fileFormats
    wmake surfMesh
    wmake triSurface
    wmake meshTools
    parallel/decompose/AllwmakeLnInclude
    dummyThirdParty/Allwmake -j "$SOURCE_BUILD_JOBS"
    wmakeLnInclude fvOptions
    wmake finiteVolume
)

test -f "$FOAM_LIBBIN/libOpenFOAM.so"
test -f "$FOAM_LIBBIN/libmeshTools.so"
test -f "$FOAM_LIBBIN/libfiniteVolume.so"
