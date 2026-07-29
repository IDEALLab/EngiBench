#!/usr/bin/env bash
set -euo pipefail

readonly source_case=/opt/mto2d/case-template

if [[ $# -ne 1 ]]; then
    echo "Usage: mto2d-export-case /empty/output/directory" >&2
    exit 2
fi

destination=$1
if [[ "$destination" != /* || "$destination" == "/" ]]; then
    echo "Destination must be an absolute path other than /: $destination" >&2
    exit 2
fi
if [[ ! -d "$source_case/app" || ! -x "$source_case/src_TF/EXEC" ]]; then
    echo "Image does not contain a built MTO2D case template." >&2
    exit 1
fi
if [[ "$(tr -d '[:space:]' <"$source_case/.engibench-mto2d-runtime-version")" != "2" ]]; then
    echo "Image case template has an unsupported runtime marker." >&2
    exit 1
fi

mkdir -p "$destination"
if find "$destination" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "Destination is not empty; refusing to overwrite it: $destination" >&2
    exit 1
fi

cp -a --no-preserve=ownership "$source_case/." "$destination/"
test -x "$destination/src_TF/EXEC"
test -f "$destination/.engibench-mto2d-runtime-version"
