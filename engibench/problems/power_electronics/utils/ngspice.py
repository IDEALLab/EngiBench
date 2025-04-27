"""NgSpice wrapper for cross-platform support."""

from __future__ import annotations

import os
import platform
import re
import subprocess

MIN_SUPPORTED_VERSION: int = 42  # Major version number of ngspice
MAX_SUPPORTED_VERSION: int = 44  # Major version number of ngspice


class NgSpice:
    """A class to handle ngspice execution across different operating systems."""

    def __init__(self, ngspice_windows_path: str | None = None) -> None:
        """Initialize the NgSpice wrapper.

        Args:
            ngspice_windows_path: The path to the ngspice executable for Windows.
        """
        self.ngspice_windows_path = os.path.normpath(ngspice_windows_path) if ngspice_windows_path else None
        self.system = platform.system().lower()
        self._ngspice_path = self._get_ngspice_path()
        if not MIN_SUPPORTED_VERSION <= self.version <= MAX_SUPPORTED_VERSION:
            raise UnsupportedNgSpiceVersionError(self.version)

    def _get_ngspice_path(self) -> str:
        """Get the path to the ngspice executable based on the operating system.

        Returns:
            The path to ngspice executable or None if not found
        """
        if self.system == "windows":
            # For Windows, use the bundled ngspice.exe
            # Look for ngspice in Spice64 folder and common install locations
            possible_paths = [
                self.ngspice_windows_path,
                os.path.normpath(os.path.join("C:/Program Files/Spice64/bin/ngspice.exe")),
                os.path.normpath(os.path.join("C:/Program Files (x86)/ngspice/bin/ngspice.exe")),
            ]

            for path in possible_paths:
                if path and os.path.exists(path):
                    ngspice_path: str | None = path
                    break
            else:
                ngspice_path = possible_paths[0]  # Default to first path if none found
            if ngspice_path is None or not os.path.exists(ngspice_path):
                raise FileNotFoundError(
                    f"ngspice.exe not found at {ngspice_path}. You can download it from https://sourceforge.net/projects/ngspice/files/ng-spice-rework/44.2/"
                )
            return ngspice_path
        if self.system in ["darwin", "linux"]:
            # For MacOS and Linux, use system-installed ngspice
            try:
                # Check if ngspice is installed
                subprocess.run(["ngspice", "--version"], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise RuntimeError(
                    "ngspice is not installed on your system. "
                    "Please install it using your package manager:\n"
                    "  - MacOS: brew install ngspice\n"
                    "  - Linux: sudo apt-get install ngspice"
                ) from None
            return "ngspice"
        raise RuntimeError(
            f"Unsupported operating system for ngspice: {self.system}, we only support Windows, MacOS and Linux."
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
    def version(self) -> int:
        """Get the version of ngspice.

        Returns:
            The major version number of ngspice as an integer

        Raises:
            subprocess.CalledProcessError: If ngspice fails to run
        """
        if self.system == "windows":
            pattern_int = re.compile(r"ngspice-(\d+)-manual\.pdf")
            pattern_dec = re.compile(r"ngspice-(\d+\.\d+)-manual\.pdf")

            docs_path = os.path.normpath(os.path.join(os.path.dirname(self._ngspice_path), "../docs/"))
            for filename in os.listdir(docs_path):
                match_int = pattern_int.match(filename)
                match_dec = pattern_dec.match(filename)
                if match_int:
                    return int(match_int.group(1))  # Already returns just the major version
                if match_dec:
                    return int(match_dec.group(1).split(".")[0])  # Return only the major version
            raise NgSpiceManualNotFoundError

        cmd = [self._ngspice_path, "--version"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Extract version number from second line of output and get only the major version

        # Example output:
        # ******
        # ** ngspice-44.2 : Circuit level simulation program
        # ** Compiled with KLU Direct Linear Solver
        # ** The U. C. Berkeley CAD Group
        # ** Copyright 1985-1994, Regents of the University of California.
        # ** Copyright 2001-2024, The ngspice team.
        # ** Please get your ngspice manual from https://ngspice.sourceforge.io/docs.html
        # ** Please file your bug-reports at http://ngspice.sourceforge.net/bugrep.html
        # ******
        full_version = result.stdout.splitlines()[1].split()[1].split("-")[1]
        return int(full_version.split(".")[0])  # Return only the major version


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
