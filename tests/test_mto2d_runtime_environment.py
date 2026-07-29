"""Regression tests for the non-interactive OpenFOAM environment shim."""

import os
from pathlib import Path
import subprocess

RUNTIME_DIR = Path(__file__).parents[1] / "engibench" / "problems" / "mto2d" / "model" / "runtime"
OPENFOAM_BASHRC = "/opt/OpenFOAM/OpenFOAM-5.x/etc/bashrc"


def _run_activation(shell_body: str, *, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", shell_body],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "MTO2D_ENVIRONMENT_HELPER": str(RUNTIME_DIR / "source-openfoam-environment.sh"),
            **environment,
        },
    )


def test_openfoam_environment_tolerates_nonzero_bashrc_with_valid_environment(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_wmake = fake_bin / "wmake"
    fake_wmake.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_wmake.chmod(0o755)

    result = _run_activation(
        f"""
set -eu
activate_runtime_environment() {{ :; }}
mkdir() {{ :; }}
source() {{
    test "$1" = "{OPENFOAM_BASHRC}"
    export WM_PROJECT_DIR=/opt/OpenFOAM/OpenFOAM-5.x
    export PATH="$MTO2D_FAKE_BIN"
    return 42
}}
. "$MTO2D_ENVIRONMENT_HELPER"
activate_openfoam_environment
printf '%s\\n' "$WM_PROJECT_DIR|$FOAM_USER_LIBBIN|$FOAM_USER_APPBIN"
""",
        environment={"MTO2D_FAKE_BIN": str(fake_bin)},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "/opt/OpenFOAM/OpenFOAM-5.x|/opt/mto2d/lib|/opt/mto2d/bin\n"


def test_openfoam_environment_fails_when_wmake_is_missing(tmp_path: Path) -> None:
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()

    result = _run_activation(
        f"""
set -eu
activate_runtime_environment() {{ :; }}
source() {{
    test "$1" = "{OPENFOAM_BASHRC}"
    export WM_PROJECT_DIR=/opt/OpenFOAM/OpenFOAM-5.x
    export PATH="$MTO2D_EMPTY_BIN"
    return 0
}}
. "$MTO2D_ENVIRONMENT_HELPER"
activate_openfoam_environment
""",
        environment={"MTO2D_EMPTY_BIN": str(empty_bin)},
    )

    assert result.returncode == 1
    assert result.stderr == "OpenFOAM bashrc did not make wmake available.\n"


def test_source_image_keeps_case_inputs_after_dependency_builds() -> None:
    dockerfile = (RUNTIME_DIR / "Dockerfile.source").read_text(encoding="utf-8")

    dependencies_finished = dockerfile.index("/usr/local/sbin/mto2d-record-source-build")
    tls_snapshot_upgrade = dockerfile.index('ca-certificates="${CA_CERTIFICATES_VERSION}"')
    openmpi_build = dockerfile.index("/usr/local/sbin/mto2d-build-openmpi")
    case_arguments = dockerfile.index("ARG FROZEN_PATCH_SHA256")
    case_copy = dockerfile.index("COPY --from=mto2d_case")
    release_labels = dockerfile.index("LABEL org.opencontainers.image.title")
    final_smoke = dockerfile.index("/usr/local/bin/mto2d-entrypoint mto2d-source-smoke")

    assert tls_snapshot_upgrade < openmpi_build < dependencies_finished
    assert dependencies_finished < case_arguments < case_copy
    assert final_smoke < release_labels


def test_source_image_uses_static_runtime_environment() -> None:
    dockerfile = (RUNTIME_DIR / "Dockerfile.source").read_text(encoding="utf-8")
    entrypoint = (RUNTIME_DIR / "docker-entrypoint.sh").read_text(encoding="utf-8")
    environment = (RUNTIME_DIR / "source-runtime-environment.sh").read_text(encoding="utf-8")

    assert "COPY source-runtime-environment.sh /usr/local/lib/mto2d/source-runtime-environment.sh" in dockerfile
    assert "source /usr/local/lib/mto2d/source-runtime-environment.sh" in entrypoint
    assert "source /opt/OpenFOAM/OpenFOAM-5.x/etc/bashrc" in entrypoint
    for value in (
        "WM_OPTIONS=linux64GccDPInt32Opt",
        "WM_MPLIB=SYSTEMOPENMPI",
        "FOAM_MPI=openmpi-system",
        'MPI_ARCH_PATH="$MPI_DIR"',
    ):
        assert value in environment


def test_source_image_uses_hash_verified_prebuilt_mesh() -> None:
    dockerfile = (RUNTIME_DIR / "Dockerfile.source").read_text(encoding="utf-8")
    builder = (RUNTIME_DIR / "source-build-prebuilt-mesh.sh").read_text(encoding="utf-8")
    helper = (RUNTIME_DIR / "prepare-mesh.sh").read_text(encoding="utf-8")
    runner = (RUNTIME_DIR.parent / "runner.py").read_text(encoding="utf-8")

    assert "mto2d-build-prebuilt-mesh" in dockerfile
    assert "mto2d-prepare-mesh" in dockerfile
    assert "blockMeshDict.sha256" in builder
    assert "files.sha256" in builder
    assert "sha256sum --check --status" in helper
    assert "blockMeshDict differs from the image cache" in helper
    assert "command -v mto2d-prepare-mesh" in runner
    assert "else blockMesh" in runner


def test_source_build_sanitizes_case_before_docker_copy() -> None:
    build_script = (RUNTIME_DIR / "build_source_image.sh").read_text(encoding="utf-8")
    docker_build = build_script.index("docker buildx build")

    for marker in ("meanT.txt", "processor[0-9]*", "-name '*~'"):
        assert build_script.index(marker) < docker_build
    assert 'buildx_cache_args+=(--cache-from "$MTO2D_BUILDX_CACHE_FROM")' in build_script
    assert 'buildx_cache_args+=(--cache-to "$MTO2D_BUILDX_CACHE_TO")' in build_script


def test_source_image_carries_release_provenance_labels() -> None:
    dockerfile = (RUNTIME_DIR / "Dockerfile.source").read_text(encoding="utf-8")

    for label in (
        "org.opencontainers.image.source",
        "org.opencontainers.image.documentation",
        "org.opencontainers.image.version",
        "org.opencontainers.image.revision",
        "org.opencontainers.image.mto2d.source-tree-state",
        "org.opencontainers.image.licenses",
        "org.opencontainers.image.base.name",
        "org.opencontainers.image.base.digest",
    ):
        assert label in dockerfile

    pins = (RUNTIME_DIR / "source-pins.env").read_text(encoding="utf-8")
    assert "BASE_IMAGE_NAME='docker.io/library/ubuntu:20.04'" in pins
    assert "BASE_IMAGE_DIGEST='sha256:" in pins
    assert "APPROVED_IMAGE_LICENSES='GPL-3.0-or-later'" in pins
    assert "OPENSSH_CLIENT_VERSION='1:8.2p1-4ubuntu0.13'" in pins


def test_source_image_publisher_is_guarded_and_non_mutating_by_default() -> None:
    publisher = RUNTIME_DIR / "publish_source_image.sh"
    result = subprocess.run(
        ["bash", str(publisher), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--confirm-redistribution-rights" in result.stdout
    assert "--confirm-reference" in result.stdout
    assert "--reference-dataset" in result.stdout
    assert "--push" in result.stdout

    script = publisher.read_text(encoding="utf-8")
    assert "push=false" in script
    assert script.index('[[ "$confirm_rights" == true ]]') < script.index('docker push "$revision_tag"')
    assert script.index('[[ "$confirm_reference" == true ]]') < script.index('docker push "$revision_tag"')
    assert script.index("verify_source_reference.py") < script.index('docker push "$revision_tag"')
    assert script.index("APPROVED_IMAGE_LICENSES") < script.index('docker push "$revision_tag"')
    assert script.index("canonical_commit_url=") < script.index('docker push "$revision_tag"')
    assert script.index("revision_probe=") < script.index('docker push "$revision_tag"')


def test_source_reference_verifier_is_simulation_only() -> None:
    verifier = (RUNTIME_DIR / "verify_source_reference.py").read_text(encoding="utf-8")
    golden = (RUNTIME_DIR / "source-reference-golden.json").read_text(encoding="utf-8")

    assert "EXPECTED_INDEX = 2010" in verifier
    assert "EXPECTED_SOURCE_CASE = 6799" in verifier
    assert "EXPECTED_DESIGN_SHA256" in verifier
    assert "[13.8912, 63.8033]" in verifier
    assert "simulate_verbose" in verifier
    assert "DETERMINISTIC_OUTPUTS" in verifier
    assert "DEFAULT_GOLDEN" in verifier
    assert "_assert_golden_match" in verifier
    assert "assert_array_equal" in verifier
    assert "default=MTO2D.dataset_id" in verifier
    assert ".optimize(" not in verifier
    assert "d53c0b6f8ec566b0d165be485efefde814e9f2af7e1e39f1ebc30a9a86ca62a6" in golden
    assert "5f2f11fba64a3a15229994f7af7914593ac412b22faab4173192bb5ad067cb7f" in golden
