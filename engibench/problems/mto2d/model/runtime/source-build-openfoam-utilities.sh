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
    cd "$WM_PROJECT_DIR"
    wmake applications/utilities/mesh/generation/blockMesh
    wmake applications/utilities/parallelProcessing/decomposePar
    wmake applications/utilities/parallelProcessing/reconstructPar
)

test -x "$FOAM_APPBIN/blockMesh"
test -x "$FOAM_APPBIN/decomposePar"
test -x "$FOAM_APPBIN/reconstructPar"
