#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=source-build-common.sh
source /usr/local/lib/mto2d/source-build-common.sh
# shellcheck source=source-openfoam-environment.sh
source /usr/local/lib/mto2d/source-openfoam-environment.sh

require_variables \
    MMA_IMPLEMENTATION_SHA256 \
    MMA_HEADER_SHA256 \
    MMA_SOURCE_COMMIT \
    MTO2D_CASE_ARCHIVE_SHA256 \
    FROZEN_PATCH_SHA256 \
    SCHEDULE_PATCH_SHA256

printf '%s  %s\n' "$MMA_IMPLEMENTATION_SHA256" /usr/local/src/mto2d-mma/MMA.c \
    | sha256sum --check -
printf '%s  %s\n' "$MMA_HEADER_SHA256" /usr/local/src/mto2d-mma/MMA.h \
    | sha256sum --check -

activate_openfoam_environment
export LD_LIBRARY_PATH="/opt:/opt/mto2d/lib:$LD_LIBRARY_PATH"

mpicxx \
    -O3 \
    -std=c++11 \
    -fPIC \
    -shared \
    -I"$PETSC_DIR/include" \
    -I"$PETSC_DIR/$PETSC_ARCH/include" \
    /usr/local/src/mto2d-mma/MMA.c \
    -L"$PETSC_DIR/$PETSC_ARCH/lib" \
    -Wl,-rpath,"$PETSC_DIR/$PETSC_ARCH/lib" \
    -lpetsc \
    -o /opt/libMMA_yu.so
ln -sfn "$MPI_DIR/include" /opt/MPI_INC

case_template=/opt/mto2d/case-template
test -d "$case_template/app"
test -d "$case_template/src_TF"
test ! -e "$case_template/src_TF/EXEC"
# Compile both the recovered MMA library and the solver against the same
# hash-pinned declaration. The retained archive carries an older compatible
# header, but silently mixing the two weakens the source-provenance record.
install -m 0644 /usr/local/src/mto2d-mma/MMA.h "$case_template/src_TF/MMA.h"
printf '%s  %s\n' "$MMA_HEADER_SHA256" "$case_template/src_TF/MMA.h" \
    | sha256sum --check -
grep -q 'updateDesign' "$case_template/src_TF/MTO_TF.C"
grep -q 'optimizationSchedule' "$case_template/src_TF/continuation.H"

(
    cd "$case_template/src_TF"
    wmake
)
test -x "$case_template/src_TF/EXEC"

# Remove source-run histories and decomposed/time output without changing the
# pristine zero-time fields needed by the Python runner.
find "$case_template/app" -maxdepth 1 -type f \
    \( -name 'meanT.txt' -o -name 'Disspower.txt' -o -name 'Voluse.txt' \
    -o -name 'Time.txt' -o -name 'aMax.txt' -o -name 'qu.txt' \
    -o -name 'HEAV.txt' \) -delete
find "$case_template/app" -maxdepth 1 -mindepth 1 -type d \
    \( -name 'processor[0-9]*' -o -name '[1-9]*' \) -exec rm -rf -- {} +

printf '2\n' >"$case_template/.engibench-mto2d-runtime-version"

{
    printf 'mma_source_commit=%s\n' "$MMA_SOURCE_COMMIT"
    printf 'mma_implementation_sha256=%s\n' "$MMA_IMPLEMENTATION_SHA256"
    printf 'mma_header_sha256=%s\n' "$MMA_HEADER_SHA256"
    printf 'case_archive_sha256=%s\n' "$MTO2D_CASE_ARCHIVE_SHA256"
    printf 'frozen_patch_sha256=%s\n' "$FROZEN_PATCH_SHA256"
    printf 'schedule_patch_sha256=%s\n' "$SCHEDULE_PATCH_SHA256"
} >/opt/mto2d/mto2d-source-inputs.txt

sha256sum \
    /opt/libMMA_yu.so \
    "$case_template/src_TF/EXEC" \
    >/opt/mto2d/binary-sha256.txt

if ldd /opt/libMMA_yu.so "$case_template/src_TF/EXEC" | grep -q 'not found'; then
    echo "A built MTO2D binary has unresolved shared-library dependencies." >&2
    exit 1
fi
