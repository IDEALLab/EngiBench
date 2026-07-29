#!/usr/bin/env bash
set -euo pipefail

for command in blockMesh decomposePar reconstructPar mpirun ssh mto2d-export-case mto2d-prepare-mesh; do
    command -v "$command" >/dev/null
done
test -f /opt/libMMA_yu.so
test -f /opt/mto2d/lib/libgroovyBC.so
test -r /usr/local/lib/mto2d/source-runtime-environment.sh
test -x /opt/mto2d/case-template/src_TF/EXEC
test -s /opt/mto2d/source-revisions.txt
test -s /opt/mto2d/dpkg-manifest.txt
test -s /opt/mto2d/mto2d-source-inputs.txt
test -s /opt/mto2d/binary-sha256.txt
test -s /opt/mto2d/prebuilt-polyMesh/blockMeshDict.sha256
test -s /opt/mto2d/prebuilt-polyMesh/files.sha256
sha256sum --check /opt/mto2d/binary-sha256.txt

for package in ca-certificates libssl1.1 openssl openssh-client; do
    expected_version=$(sed -n "s/^${package}=//p" /opt/mto2d/source-revisions.txt)
    installed_version=$(dpkg-query -W -f='${Version}' "$package")
    if [[ -z "$expected_version" || "$installed_version" != "$expected_version" ]]; then
        echo "Pinned package mismatch for $package." >&2
        exit 1
    fi
done

unexpected_case_file=$(
    find /opt/mto2d/case-template -type f \
        \( -name '*~' -o -name 'meanT.txt' -o -name 'Disspower.txt' \
        -o -name 'Voluse.txt' -o -name 'Time.txt' -o -name 'aMax.txt' \
        -o -name 'qu.txt' -o -name 'HEAV.txt' \) \
        -print -quit
)
if [[ -n "$unexpected_case_file" ]]; then
    echo "Generated or backup file retained in case template: $unexpected_case_file" >&2
    exit 1
fi

if ldd /opt/libMMA_yu.so /opt/mto2d/case-template/src_TF/EXEC | grep -q 'not found'; then
    echo "Unresolved shared-library dependency." >&2
    exit 1
fi

smoke_root=$(mktemp -d /tmp/mto2d-source-smoke.XXXXXX)
cleanup() {
    if [[ -n "${smoke_root:-}" && "$smoke_root" == /tmp/mto2d-source-smoke.* ]]; then
        rm -rf -- "$smoke_root"
    fi
}
trap cleanup EXIT

mto2d-export-case "$smoke_root/case"
test -x "$smoke_root/case/src_TF/EXEC"
test "$(tr -d '[:space:]' <"$smoke_root/case/.engibench-mto2d-runtime-version")" = 2
mto2d-prepare-mesh "$smoke_root/case/app" >/dev/null
(
    cd "$smoke_root/case/app/constant/polyMesh"
    sha256sum --check --status /opt/mto2d/prebuilt-polyMesh/files.sha256
)
blockMesh -help >/dev/null
echo "MTO2D source-image structural smoke test passed."
