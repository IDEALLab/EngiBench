#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 /path/to/warm-ready-2d.zip /output/case [runtime-image]"
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
    usage >&2
    exit 2
fi

archive=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
output=$2
runtime_image=${3:-engibench-mto2d:sif-parity}
runtime_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ ! -f "$archive" ]]; then
    echo "Case archive does not exist: $archive" >&2
    exit 1
fi
if [[ -e "$output" ]]; then
    echo "Output already exists; refusing to overwrite it: $output" >&2
    exit 1
fi

mkdir -p "$output"
unzip -q "$archive" -d "$output"
patch --batch --forward --strip=1 --directory="$output" \
    <"$runtime_dir/frozen-evaluation.patch"
patch --batch --forward --strip=1 --directory="$output" \
    <"$runtime_dir/optimization-schedules.patch"

if ! docker image inspect "$runtime_image" >/dev/null 2>&1; then
    echo "Runtime image does not exist: $runtime_image" >&2
    exit 1
fi

output_absolute=$(cd "$output" && pwd)
docker run --rm \
    --platform linux/amd64 \
    --user "$(id -u):$(id -g)" \
    --env HOME=/tmp \
    --mount "type=bind,src=$output_absolute,dst=/work/case" \
    "$runtime_image" \
    bash -lc "cd /work/case/src_TF && wmake"
chmod u+x "$output/src_TF/EXEC"
printf '2\n' >"$output/.engibench-mto2d-runtime-version"

echo "Prepared case template: $output"
