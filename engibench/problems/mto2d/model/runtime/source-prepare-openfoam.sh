#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=source-build-common.sh
source /usr/local/lib/mto2d/source-build-common.sh
# shellcheck source=source-openfoam-environment.sh
source /usr/local/lib/mto2d/source-openfoam-environment.sh
require_variables \
    OPENFOAM_REPOSITORY \
    OPENFOAM_COMMIT \
    THIRD_PARTY_REPOSITORY \
    THIRD_PARTY_COMMIT \
    SCOTCH_REPOSITORY \
    SCOTCH_COMMIT \
    SCOTCH_MAKEFILE_SHA256 \
    SCOTCH_DGRAPH_HALO_SHA256 \
    SOURCE_BUILD_JOBS

mkdir -p /opt/OpenFOAM
fetch_commit "$OPENFOAM_REPOSITORY" "$OPENFOAM_COMMIT" /opt/OpenFOAM/OpenFOAM-5.x
fetch_commit "$THIRD_PARTY_REPOSITORY" "$THIRD_PARTY_COMMIT" /opt/OpenFOAM/ThirdParty-5.x
fetch_commit "$SCOTCH_REPOSITORY" "$SCOTCH_COMMIT" /usr/local/src/scotch-6.0.8

# OpenFOAM 5.x names this dependency directory scotch_6.0.3, while the
# historical MTO2D SIF actually contains byte-for-byte upstream Scotch 6.0.8.
# Reconstruct that input exactly. Version 6.0.8 uses MPI_Type_get_extent with
# OpenMPI 4 and therefore needs no global MPI-1 compatibility switch.
scotch_source=/opt/OpenFOAM/ThirdParty-5.x/scotch_6.0.3
rm -rf -- "$scotch_source"
mkdir -p "$scotch_source"
git -C /usr/local/src/scotch-6.0.8 archive "$SCOTCH_COMMIT" \
    | tar -x -C "$scotch_source"
printf '%s  %s\n' "$SCOTCH_MAKEFILE_SHA256" "$scotch_source/src/Makefile" \
    | sha256sum --check -
printf '%s  %s\n' \
    "$SCOTCH_DGRAPH_HALO_SHA256" \
    "$scotch_source/src/libscotch/dgraph_halo.c" \
    | sha256sum --check -
grep -Eq '^PATCHLEVEL[[:space:]]*=[[:space:]]*8$' \
    "$scotch_source/src/Makefile"

activate_openfoam_environment
export WM_NCOMPPROCS="$SOURCE_BUILD_JOBS"

(
    cd "$WM_PROJECT_DIR/wmake/src"
    make
)

# Build both serial Scotch and PT-Scotch as carried by the historical image.
"$WM_THIRD_PARTY_DIR/Allwmake"

scotch_arch_path=$(
    # shellcheck source=/dev/null
    source "$WM_PROJECT_DIR/etc/config.sh/scotch"
    printf '%s' "$SCOTCH_ARCH_PATH"
)
test -f "$scotch_arch_path/include/scotch.h"
test -f "$FOAM_EXT_LIBBIN/libscotch.so"
test -f "$FOAM_EXT_LIBBIN/libscotcherrexit.so"
test -f "$FOAM_EXT_LIBBIN/$FOAM_MPI/libptscotch.so"
undefined_ptscotch_symbols=$(
    nm -D --undefined-only "$FOAM_EXT_LIBBIN/$FOAM_MPI/libptscotch.so"
)
if grep -q 'MPI_Type_extent' <<<"$undefined_ptscotch_symbols"; then
    echo "PT-Scotch unexpectedly references removed MPI_Type_extent." >&2
    exit 1
fi
grep -q 'MPI_Type_get_extent' <<<"$undefined_ptscotch_symbols"
