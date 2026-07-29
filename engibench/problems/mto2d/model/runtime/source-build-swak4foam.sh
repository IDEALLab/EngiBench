#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=source-build-common.sh
source /usr/local/lib/mto2d/source-build-common.sh
# shellcheck source=source-openfoam-environment.sh
source /usr/local/lib/mto2d/source-openfoam-environment.sh
require_variables SWAK4FOAM_REPOSITORY SWAK4FOAM_COMMIT

activate_openfoam_environment
fetch_commit "$SWAK4FOAM_REPOSITORY" "$SWAK4FOAM_COMMIT" /opt/swak4Foam
(
    cd /opt/swak4Foam
    python3 maintainanceScripts/makeSwakVersionFile.py
    cd Libraries
    python3 ../maintainanceScripts/makeFoamVersionHeader.py \
        "$WM_PROJECT_VERSION" >swak4FoamParsers/foamVersion4swak.H
    sed -ne 's/.*SWAK_IS_COM \([0-9][0-9]*\)/OPENFOAM_COM=\1/p' \
        <swak4FoamParsers/foamVersion4swak.H >rules/foamVersion
    sed -ne 's/.*SWAK_IS_ORG \([0-9][0-9]*\)/OPENFOAM_ORG=\1/p' \
        <swak4FoamParsers/foamVersion4swak.H >>rules/foamVersion
    wmakeLnInclude simpleFunctionObjects
    wmake libso swak4FoamParsers
    wmake libso groovyBC
)
test -f "$FOAM_USER_LIBBIN/libswak4FoamParsers.so"
test -f "$FOAM_USER_LIBBIN/libgroovyBC.so"
