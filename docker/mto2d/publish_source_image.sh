#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  publish_source_image.sh [options]

Validate a local MTO2D source image and optionally publish it to GHCR.
The default is a non-mutating validation run.

Options:
  --image IMAGE                         Local image (default: engibench-mto2d:source-local)
  --remote REPOSITORY                   Remote repository (default: ghcr.io/arthurdrake1/engibench-mto2d)
  --confirm-redistribution-rights       Confirm that publication rights are resolved
  --reference-dataset SOURCE            HF dataset ID or saved DatasetDict used by the q=0.01 check
  --confirm-reference                   Run and require q=0.01 numerical reference parity
  --push                                Tag and push v0 and v0-<Git SHA>
  -h, --help                            Show this help

Set MTO2D_PYTHON to select the EngiBench development Python interpreter.
EOF
}

image=engibench-mto2d:source-local
remote=ghcr.io/arthurdrake1/engibench-mto2d
confirm_rights=false
confirm_reference=false
reference_dataset=${MTO2D_REFERENCE_DATASET:-IDEALLab/mto_2d_v0}
push=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)
            [[ $# -ge 2 ]] || { echo "--image requires a value" >&2; exit 2; }
            image=$2
            shift 2
            ;;
        --remote)
            [[ $# -ge 2 ]] || { echo "--remote requires a value" >&2; exit 2; }
            remote=$2
            shift 2
            ;;
        --confirm-redistribution-rights)
            confirm_rights=true
            shift
            ;;
        --reference-dataset)
            [[ $# -ge 2 ]] || { echo "--reference-dataset requires a value" >&2; exit 2; }
            reference_dataset=$2
            shift 2
            ;;
        --confirm-reference)
            confirm_reference=true
            shift
            ;;
        --push)
            push=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

runtime_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repository=$(git -C "$runtime_dir" rev-parse --show-toplevel)
revision=$(git -C "$repository" rev-parse HEAD)
# shellcheck source=source-pins.env
source "$runtime_dir/source-pins.env"

command -v docker >/dev/null
docker image inspect "$image" >/dev/null
image_id=$(docker image inspect --format '{{.Id}}' "$image")

image_os=$(docker image inspect --format '{{.Os}}' "$image")
image_arch=$(docker image inspect --format '{{.Architecture}}' "$image")
image_revision=$(
    docker image inspect --format \
        '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$image"
)
tree_state=$(
    docker image inspect --format \
        '{{index .Config.Labels "org.opencontainers.image.mto2d.source-tree-state"}}' "$image"
)
licenses=$(
    docker image inspect --format \
        '{{index .Config.Labels "org.opencontainers.image.licenses"}}' "$image"
)
source_url=$(
    docker image inspect --format \
        '{{index .Config.Labels "org.opencontainers.image.source"}}' "$image"
)
version=$(
    docker image inspect --format \
        '{{index .Config.Labels "org.opencontainers.image.version"}}' "$image"
)

[[ "$image_os/$image_arch" == "linux/amd64" ]] || {
    echo "Release image must be linux/amd64; got $image_os/$image_arch" >&2
    exit 1
}
[[ "$image_revision" == "$revision" ]] || {
    echo "Image revision $image_revision does not match checkout $revision" >&2
    exit 1
}
[[ "$tree_state" == "clean" ]] || {
    echo "Image was built from a $tree_state source tree; rebuild from a clean commit." >&2
    exit 1
}
[[ -z "$(git -C "$repository" status --porcelain --untracked-files=normal)" ]] || {
    echo "EngiBench checkout is dirty; commit the release source before publication." >&2
    exit 1
}
[[ "$APPROVED_IMAGE_LICENSES" != "NOASSERTION" && -n "$APPROVED_IMAGE_LICENSES" ]] || {
    echo "source-pins.env has no legally approved image license expression." >&2
    exit 1
}
[[ "$licenses" == "$APPROVED_IMAGE_LICENSES" ]] || {
    echo "Image license label does not match APPROVED_IMAGE_LICENSES in source-pins.env." >&2
    exit 1
}
[[ "$source_url" == "https://github.com/IDEALLab/EngiBench" ]] || {
    echo "Unexpected OCI source label: $source_url" >&2
    exit 1
}
[[ "$version" == "v0" ]] || {
    echo "Unexpected OCI version label: $version" >&2
    exit 1
}

docker run --rm --platform linux/amd64 "$image" mto2d-source-smoke

if [[ "$push" != true ]]; then
    echo "Local image passed structural publication checks: $image"
    echo "No remote changes made. Add --push with both confirmation flags to publish."
    exit 0
fi

[[ "$confirm_rights" == true ]] || {
    echo "--push requires --confirm-redistribution-rights" >&2
    exit 2
}
[[ "$confirm_reference" == true ]] || {
    echo "--push requires --confirm-reference" >&2
    exit 2
}
if [[ -d "$reference_dataset" ]]; then
    reference_dataset=$(cd "$reference_dataset" && pwd)
fi
python_command=${MTO2D_PYTHON:-"$repository/.venv/bin/python"}
if [[ ! -x "$python_command" ]]; then
    python_command=$(command -v python || true)
fi
[[ -n "$python_command" && -x "$python_command" ]] || {
    echo "An EngiBench Python interpreter is required; set MTO2D_PYTHON." >&2
    exit 1
}
python_command=$(cd "$(dirname "$python_command")" && pwd)/$(basename "$python_command")
command -v jq >/dev/null || {
    echo "jq is required to verify remote OCI digests." >&2
    exit 1
}
command -v curl >/dev/null || {
    echo "curl is required to verify the public source revision." >&2
    exit 1
}

(
    cd "$repository"
    DOCKER_DEFAULT_PLATFORM=linux/amd64 \
        "$python_command" "$runtime_dir/verify_source_reference.py" \
        --image "$image" \
        --dataset "$reference_dataset"
)
[[ "$(docker image inspect --format '{{.Id}}' "$image")" == "$image_id" ]] || {
    echo "Local image tag changed during release validation; refusing to publish." >&2
    exit 1
}

canonical_commit_url="https://github.com/IDEALLab/EngiBench/commit/${revision}"
curl --fail --silent --show-error --location --output /dev/null "$canonical_commit_url" || {
    echo "Revision is not publicly reachable from IDEALLab/EngiBench: $revision" >&2
    exit 1
}

revision_tag="${remote}:v0-${revision}"
version_tag="${remote}:v0"

manifest_digest() {
    docker buildx imagetools inspect "$1" --format '{{json .Manifest}}' \
        | jq -er '.digest | select(test("^sha256:[0-9a-f]{64}$"))'
}

remote_config_digest() {
    docker buildx imagetools inspect "$1" --raw \
        | jq -er '.config.digest | select(test("^sha256:[0-9a-f]{64}$"))'
}

revision_exists=false
if revision_probe=$(docker buildx imagetools inspect "$revision_tag" 2>&1); then
    revision_exists=true
    existing_revision_config=$(remote_config_digest "$revision_tag")
    [[ "$existing_revision_config" == "$image_id" ]] || {
        echo "Commit-scoped tag already points to different image content: $revision_tag" >&2
        exit 1
    }
    echo "Commit-scoped image is already present; resuming release: $revision_tag"
elif ! grep -Eqi 'not found|manifest unknown|404' <<<"$revision_probe"; then
    echo "Could not prove that the commit-scoped tag is absent:" >&2
    echo "$revision_probe" >&2
    exit 1
fi

docker tag "$image_id" "$revision_tag"
docker tag "$image_id" "$version_tag"
if [[ "$revision_exists" != true ]]; then
    docker push "$revision_tag"
fi
docker push "$version_tag"

revision_digest=$(manifest_digest "$revision_tag")
version_digest=$(manifest_digest "$version_tag")
[[ "$revision_digest" == "$version_digest" ]] || {
    echo "Remote release tags resolved to different digests." >&2
    exit 1
}
revision_config=$(remote_config_digest "$revision_tag")
version_config=$(remote_config_digest "$version_tag")
[[ "$revision_config" == "$image_id" && "$version_config" == "$image_id" ]] || {
    echo "Published config digest does not match the validated local image ID." >&2
    exit 1
}

echo "Published GHCR candidate: ${remote}@${version_digest}"
echo "Ensure the package is public in GitHub, verify an anonymous digest pull, and pin this digest in MTO2D.container_id."
