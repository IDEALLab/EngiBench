"""NgSpice wrapper for cross-platform support."""

from dataclasses import dataclass
from functools import cache
from functools import cached_property
import hashlib
import os
import platform
import re
import shutil
import subprocess
import warnings

MIN_SUPPORTED_VERSION: int = 42  # Major version number of ngspice
MAX_SUPPORTED_VERSION: int = 45  # Major version number of ngspice
NGSPICE_PATH_ENV = "NGSPICE_PATH"
ARM64_MACHINES = frozenset({"aarch64", "arm64"})
ARCHITECTURE_ISSUE_URL = "https://sourceforge.net/p/ngspice/bugs/622/"


class NgSpiceArchitectureWarning(RuntimeWarning):
    """Warn that ngspice results may differ between CPU architectures."""


@cache
def _warn_if_arm64(machine: str) -> None:
    """Warn once per process when ngspice runs on a 64-bit ARM host."""
    if machine not in ARM64_MACHINES:
        return

    warnings.warn(
        "PowerElectronics is running on an ARM64/AArch64 host. ngspice "
        "transient results are numerically backend-dependent, and native "
        "ARM64/AArch64 and x86_64 binaries may differ. Do not assume that this "
        "run reproduces published dataset labels; use the pinned Linux x86_64 "
        "runtime for canonical v1 results. See ngspice bug #622 "
        f"(closed-wont-fix): {ARCHITECTURE_ISSUE_URL}",
        NgSpiceArchitectureWarning,
        stacklevel=3,
    )


@dataclass(frozen=True)
class NgSpiceIdentity:
    """Immutable identity of the ngspice backend used for a simulation."""

    version: str
    major_version: int
    executable_path: str
    executable_sha256: str
    platform_system: str
    platform_machine: str
    version_output: str


class NgSpice:
    """A class to handle ngspice execution across different operating systems."""

    def __init__(
        self,
        ngspice_path: str | None = None,
        *,
        ngspice_windows_path: str | None = None,
    ) -> None:
        """Initialize the NgSpice wrapper.

        Args:
            ngspice_path: Path to the ngspice executable on any supported platform.
                Takes precedence over ``NGSPICE_PATH`` and ``PATH``.
            ngspice_windows_path: Deprecated alias for ``ngspice_path``.
        """
        if ngspice_windows_path is not None:
            warnings.warn(
                "ngspice_windows_path is deprecated; use ngspice_path instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        if ngspice_path is not None and ngspice_windows_path is not None:
            raise ValueError("Pass either ngspice_path or ngspice_windows_path, not both.")

        self.configured_path = ngspice_path or ngspice_windows_path or os.environ.get(NGSPICE_PATH_ENV)
        self.system = platform.system().lower()
        self._ngspice_path = self._get_ngspice_path()
        if not MIN_SUPPORTED_VERSION <= self.version <= MAX_SUPPORTED_VERSION:
            raise UnsupportedNgSpiceVersionError(self.version)
        _warn_if_arm64(platform.machine().lower())

    def _get_ngspice_path(self) -> str:
        """Get the path to the ngspice executable based on the operating system.

        Returns:
            The path to the ngspice executable
        """
        if self.system not in {"darwin", "linux", "windows"}:
            raise RuntimeError(
                f"Unsupported operating system for ngspice: {self.system}. EngiBench supports Windows, macOS, and Linux."
            )

        if self.configured_path:
            path = os.path.abspath(os.path.expanduser(self.configured_path))
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Configured ngspice executable does not exist: {path}")
            return path

        path_from_env = shutil.which("ngspice")
        if path_from_env:
            return path_from_env

        if self.system == "windows":
            common_paths = (
                "C:/Program Files/Spice64/bin/ngspice.exe",
                "C:/Program Files (x86)/ngspice/bin/ngspice.exe",
            )
            installed_path = next((os.path.normpath(path) for path in common_paths if os.path.isfile(path)), None)
            if installed_path:
                return installed_path

        raise FileNotFoundError(
            f"ngspice was not found. Set {NGSPICE_PATH_ENV} to a supported ngspice executable "
            "or add it to PATH. See engibench/problems/power_electronics/README.md for installation instructions."
        )

    def run(self, netlist_path: str, log_file_path: str, timeout: int = 30) -> None:
        """Run ngspice with the given netlist file.

        Args:
            netlist_path: Path to the netlist file
            log_file_path: Path to the log file
            timeout: Maximum time to wait for the simulation in seconds

        Raises:
            subprocess.CalledProcessError: If ngspice fails to run
            subprocess.TimeoutExpired: If the simulation takes too long
        """
        cmd = [
            self._ngspice_path,
            "-o",
            log_file_path,
            netlist_path,
        ]
        print(f"Running command: {cmd}")
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
        except subprocess.CalledProcessError as e:
            print(f"ngspice execution failed with return code {e.returncode}")
            print(f"Error output: {e.stderr}")
            raise
        except subprocess.TimeoutExpired:
            print(f"ngspice simulation timed out after {timeout} seconds")
            raise

    @property
    def executable_path(self) -> str:
        """Return the resolved ngspice executable path."""
        return self._ngspice_path

    @cached_property
    def version_output(self) -> str:
        """Return the complete version banner emitted by ngspice."""
        result = subprocess.run(
            [self._ngspice_path, "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        return f"{result.stdout}\n{result.stderr}".strip()

    @cached_property
    def version_string(self) -> str:
        """Return the complete ngspice version, including patch components."""
        match = re.search(r"\bngspice-(\d+(?:\.\d+)*)\b", self.version_output, flags=re.IGNORECASE)
        if match is None:
            raise RuntimeError(f"Could not determine ngspice version from: {self.version_output!r}")
        return match.group(1)

    @cached_property
    def identity(self) -> NgSpiceIdentity:
        """Return the exact simulator and host identity used by this wrapper."""
        digest = hashlib.sha256()
        with open(self._ngspice_path, "rb") as executable:
            for chunk in iter(lambda: executable.read(1024 * 1024), b""):
                digest.update(chunk)
        return NgSpiceIdentity(
            version=self.version_string,
            major_version=self.version,
            executable_path=self._ngspice_path,
            executable_sha256=digest.hexdigest(),
            platform_system=platform.system(),
            platform_machine=platform.machine(),
            version_output=self.version_output,
        )

    @property
    def version(self) -> int:
        """Get the version of ngspice.

        Returns:
            The major version number of ngspice as an integer

        Raises:
            subprocess.CalledProcessError: If ngspice fails to run
        """
        if self.system == "windows":
            # Try finding the version from the docs folder (for SourceForge binary package)
            pattern_int = re.compile(r"ngspice-(\d+)-manual\.pdf")
            pattern_dec = re.compile(r"ngspice-(\d+\.\d+)-manual\.pdf")

            docs_path = os.path.normpath(os.path.join(os.path.dirname(self._ngspice_path), "../docs/"))
            try:
                for filename in os.listdir(docs_path):
                    match_int = pattern_int.match(filename)
                    match_dec = pattern_dec.match(filename)
                    if match_int:
                        return int(match_int.group(1))  # Already returns just the major version
                    if match_dec:
                        return int(match_dec.group(1).split(".")[0])  # Return only the major version
            except OSError:
                print(f"Could not read ngspice docs folder at {docs_path!r}, falling back to --version flag.")

        return int(self.version_string.split(".", maxsplit=1)[0])


class NgSpiceManualNotFoundError(FileNotFoundError):
    """Custom exception for missing ngspice manual file on Windows."""

    def __init__(self):
        """Initialize the exception with a custom message."""
        super().__init__("ngspice-*-manual.pdf not found in the docs folder.")


class UnsupportedNgSpiceVersionError(RuntimeError):
    """Custom exception for unsupported ngspice versions."""

    def __init__(self, version: int):
        """Initialize the exception with a custom message."""
        super().__init__(
            f"Unsupported ngspice version: {version!s}. We only support version {MIN_SUPPORTED_VERSION} to {MAX_SUPPORTED_VERSION}."
        )


if __name__ == "__main__":
    ngspice = NgSpice()
    print(ngspice.version)
