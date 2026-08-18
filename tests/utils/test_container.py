from pathlib import Path
import subprocess
import sys

import pytest

from engibench.utils import container

available_runtimes = [rt for rt in container.RUNTIMES if rt.is_available()]
TEST_TIMEOUT = 12.0


def test_run_rejects_nonpositive_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject invalid timeouts before invoking a runtime."""
    monkeypatch.setattr(container, "RUNTIME", container.Docker)

    with pytest.raises(ValueError, match="timeout must be positive"):
        container.run(["true"], "alpine", timeout=0.0)


def test_docker_timeout_force_removes_container_by_cid(monkeypatch: pytest.MonkeyPatch) -> None:
    """A client-side timeout must also stop the daemon-owned container."""
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append((command, kwargs))
        if command[1] == "run":
            cidfile = Path(command[command.index("--cidfile") + 1])
            cidfile.write_text("abc123\n", encoding="ascii")
            timeout = kwargs["timeout"]
            assert isinstance(timeout, float)
            raise subprocess.TimeoutExpired(command, timeout)
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(container, "RUNTIME", container.Docker)

    with pytest.raises(TimeoutError, match="timed out after 12"):
        container.run(["sleep", "60"], "alpine", timeout=TEST_TIMEOUT)

    run_command, run_kwargs = calls[0]
    name = run_command[run_command.index("--name") + 1]
    assert name.startswith("engibench-")
    assert run_kwargs["timeout"] == TEST_TIMEOUT
    assert calls[1][0] == ["docker", "rm", "--force", "abc123"]


def test_docker_timeout_warns_if_force_removal_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not hide a daemon-side cleanup failure after a client timeout."""

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if command[1] == "run":
            cidfile = Path(command[command.index("--cidfile") + 1])
            cidfile.write_text("abc123\n", encoding="ascii")
            raise subprocess.TimeoutExpired(command, TEST_TIMEOUT)
        return subprocess.CompletedProcess(command, 1, stdout=b"", stderr=b"cleanup failed")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(container, "RUNTIME", container.Docker)

    with pytest.warns(RuntimeWarning, match="may still be running"), pytest.raises(TimeoutError):
        container.run(["sleep", "60"], "alpine", timeout=TEST_TIMEOUT)


def test_podman_inherits_host_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Podman inherits the host settings needed by rootless helpers."""
    calls: list[dict[str, object]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin")
    monkeypatch.setattr(container.shutil, "which", lambda executable: f"/usr/local/bin/{executable}")
    monkeypatch.setattr(subprocess, "run", fake_run)

    container.Podman.run(["true"], "alpine")

    assert calls[0]["env"] is None


@pytest.mark.parametrize("runtime", available_runtimes)
@pytest.mark.skipif(sys.platform == "win32", reason="Skip Singularity tests on Windows")
def test_run_singularity_sets_correct_environment(runtime: type[container.ContainerRuntime]) -> None:
    """Test if singularity can run a container with an environment variable."""

    runtime.run(command=["sh", "-c", "[ $TEST_VAR = test ]"], env={"TEST_VAR": "test"}, image="alpine").check_returncode()


@pytest.mark.parametrize("runtime", available_runtimes)
@pytest.mark.skipif(sys.platform == "win32", reason="Skip Singularity tests on Windows")
def test_run_singularity_mounts_files(runtime: type[container.ContainerRuntime]) -> None:
    """Test if singularity can run a container with a mount."""

    check_string = "A string which appears in this file"

    runtime.run(
        command=["grep", check_string, "/mnt/test.py"],
        mounts=[(__file__, "/mnt/test.py")],
        image="alpine",
    ).check_returncode()
