#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 /path/to/warm-ready-2d.zip /output/case"
}

if [[ $# -ne 2 ]]; then
    usage >&2
    exit 2
fi

archive=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
output=$2

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
chmod u+x "$output/src_TF/EXEC"

echo "Prepared case template: $output"
