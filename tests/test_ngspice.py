"""Tests for locating and validating the ngspice executable."""

import hashlib
from pathlib import Path
import subprocess

import pytest

from engibench.problems.power_electronics.utils import ngspice as ngspice_module
from engibench.problems.power_electronics.utils.ngspice import MAX_SUPPORTED_VERSION
from engibench.problems.power_electronics.utils.ngspice import NgSpice

VERSION_OUTPUT = "******\n** ngspice-44.2 : Circuit level simulation program\n******\n"
EXPECTED_MAJOR_VERSION = 44


def mock_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every ngspice version probe report the supported CI version."""

    def run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=VERSION_OUTPUT, stderr="")

    monkeypatch.setattr(ngspice_module.subprocess, "run", run)


def executable(tmp_path: Path, name: str = "ngspice") -> Path:
    """Create a file suitable for path-resolution tests."""
    path = tmp_path / name
    path.write_text("test executable")
    return path


def test_explicit_path_takes_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The constructor argument must override the environment and PATH."""
    explicit = executable(tmp_path, "explicit-ngspice")
    monkeypatch.setenv("NGSPICE_PATH", str(tmp_path / "environment-ngspice"))
    monkeypatch.setattr(ngspice_module.platform, "system", lambda: "Darwin")
    mock_version(monkeypatch)

    assert NgSpice(ngspice_path=str(explicit)).executable_path == str(explicit)


def test_legacy_windows_path_keyword_is_supported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep the previous public keyword working for existing callers."""
    configured = executable(tmp_path)
    monkeypatch.setattr(ngspice_module.platform, "system", lambda: "Windows")
    mock_version(monkeypatch)

    with pytest.warns(DeprecationWarning, match="ngspice_windows_path is deprecated"):
        assert NgSpice(ngspice_windows_path=str(configured)).executable_path == str(configured)


def test_environment_path_works_on_macos(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """NGSPICE_PATH must work outside Windows, which is the issue #249 use case."""
    configured = executable(tmp_path)
    monkeypatch.setenv("NGSPICE_PATH", str(configured))
    monkeypatch.setattr(ngspice_module.platform, "system", lambda: "Darwin")
    mock_version(monkeypatch)

    assert NgSpice().executable_path == str(configured)


def test_path_lookup_is_cross_platform(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Without an override, use the ngspice executable found on PATH."""
    discovered = executable(tmp_path)
    monkeypatch.delenv("NGSPICE_PATH", raising=False)
    monkeypatch.setattr(ngspice_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(ngspice_module.shutil, "which", lambda _name: str(discovered))
    mock_version(monkeypatch)

    assert NgSpice().executable_path == str(discovered)


def test_invalid_configured_path_fails_clearly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Do not silently ignore a bad path selected by the user or CI."""
    missing = tmp_path / "missing-ngspice"
    monkeypatch.setenv("NGSPICE_PATH", str(missing))
    monkeypatch.setattr(ngspice_module.platform, "system", lambda: "Darwin")

    with pytest.raises(FileNotFoundError, match="Configured ngspice executable does not exist"):
        NgSpice()


def test_version_can_be_reported_on_stderr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Some ngspice packages write their version banner to stderr."""
    configured = executable(tmp_path)
    monkeypatch.setattr(ngspice_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        ngspice_module.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr="ngspice-45.2 : Circuit level simulation program"
        ),
    )

    assert NgSpice(ngspice_path=str(configured)).version == MAX_SUPPORTED_VERSION


def test_version_string_preserves_patch_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Provenance must distinguish ngspice 44 from the 44.2 bug-fix release."""
    configured = executable(tmp_path)
    monkeypatch.setattr(ngspice_module.platform, "system", lambda: "Linux")
    mock_version(monkeypatch)

    ngspice = NgSpice(ngspice_path=str(configured))

    assert ngspice.version == EXPECTED_MAJOR_VERSION
    assert ngspice.version_string == "44.2"


def test_identity_fingerprints_binary_and_platform(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Dataset manifests need an immutable backend identity, not only a major version."""
    configured = executable(tmp_path)
    monkeypatch.setattr(ngspice_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(ngspice_module.platform, "machine", lambda: "x86_64")
    mock_version(monkeypatch)

    identity = NgSpice(ngspice_path=str(configured)).identity

    assert identity.version == "44.2"
    assert identity.major_version == EXPECTED_MAJOR_VERSION
    assert identity.executable_path == str(configured)
    assert identity.executable_sha256 == hashlib.sha256(configured.read_bytes()).hexdigest()
    assert identity.platform_system == "Linux"
    assert identity.platform_machine == "x86_64"
    assert "ngspice-44.2" in identity.version_output
