#!/usr/bin/env bash
set -euo pipefail

readonly cache=/opt/mto2d/prebuilt-polyMesh

if [[ $# -gt 1 ]]; then
    echo "Usage: mto2d-prepare-mesh [app-directory]" >&2
    exit 2
fi

app=${1:-.}
dictionary=$app/system/blockMeshDict
if [[ ! -f "$dictionary" ]]; then
    echo "MTO2D case has no blockMeshDict: $dictionary" >&2
    exit 1
fi

if [[ ! -s "$cache/blockMeshDict.sha256" || ! -s "$cache/files.sha256" ]]; then
    (
        cd "$app"
        blockMesh
    )
    exit
fi

actual_dictionary_hash=$(sha256sum "$dictionary" | awk '{print $1}')
expected_dictionary_hash=$(tr -d '[:space:]' <"$cache/blockMeshDict.sha256")
if [[ "$actual_dictionary_hash" != "$expected_dictionary_hash" ]]; then
    echo "blockMeshDict differs from the image cache; regenerating mesh."
    (
        cd "$app"
        blockMesh
    )
    exit
fi

(
    cd "$cache/files"
    sha256sum --check --status ../files.sha256
)

destination=$app/constant/polyMesh
rm -rf -- "$destination"
mkdir -p "$destination"
while read -r _hash name; do
    install -m 0644 "$cache/files/$name" "$destination/$name"
done <"$cache/files.sha256"
(
    cd "$destination"
    sha256sum --check --status "$cache/files.sha256"
)
echo "Reused hash-verified MTO2D mesh from the image."
