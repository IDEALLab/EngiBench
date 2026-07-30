#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=source-build-common.sh
source /usr/local/lib/mto2d/source-build-common.sh
# shellcheck source=source-openfoam-environment.sh
source /usr/local/lib/mto2d/source-openfoam-environment.sh
activate_openfoam_environment

readonly source_app=/opt/mto2d/case-template/app
readonly cache=/opt/mto2d/prebuilt-polyMesh
readonly mesh_files=(boundary cellZones faces neighbour owner points)

test -f "$source_app/system/blockMeshDict"
test ! -e "$source_app/constant/polyMesh"

temporary=$(mktemp -d /tmp/mto2d-prebuilt-mesh.XXXXXX)
cleanup() {
    if [[ -n "${temporary:-}" && "$temporary" == /tmp/mto2d-prebuilt-mesh.* ]]; then
        rm -rf -- "$temporary"
    fi
}
trap cleanup EXIT

cp -a "$source_app" "$temporary/app"
(
    cd "$temporary/app"
    blockMesh >/dev/null
)

mkdir -p "$cache/files"
for name in "${mesh_files[@]}"; do
    install -m 0644 "$temporary/app/constant/polyMesh/$name" "$cache/files/$name"
done
sha256sum "$source_app/system/blockMeshDict" \
    | awk '{print $1}' >"$cache/blockMeshDict.sha256"
(
    cd "$cache/files"
    sha256sum "${mesh_files[@]}" >../files.sha256
)

test ! -e "$source_app/constant/polyMesh"
