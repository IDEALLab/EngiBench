"""Abstraction over container runtimes."""

from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def pull(image: str) -> None:
    """Pull an image using the selected runtime.

    Args:
        image: Container image to pull.
    """
    if RUNTIME is None:
        msg = "No container runtime found"
        raise FileNotFoundError(msg)

    RUNTIME.pull(image)


def run(
    command: list[str],
    image: str,
    mounts: Sequence[tuple[str, str]] = (),
    env: dict[str, str] | None = None,
    name: str | None = None,
) -> None:
    """Run a command in a container using the selected runtime.

    Args:
        command: Command (as a list of strings) to run inside the container.
        image: Container image to use.
        mounts: Pairs of host folder and destination folder inside the container.
        env: Mapping of environment variable names and values to set inside the container.
        name: Optional name for the container (not supported by all runtimes).
    """
    if RUNTIME is None:
        msg = "No container runtime found. Please ensure Docker, Podman, or Singularity is installed and running."
        raise FileNotFoundError(msg)

    try:
        result = RUNTIME.run(command, image, mounts, env, name)
        result.check_returncode()
    except subprocess.CalledProcessError as e:
        msg = f"Container command failed with exit code {e.returncode}:\nCommand: {' '.join(command)}\nOutput: {e.output.decode() if e.output else 'No output'}"
        raise RuntimeError(msg) from e


def runtime() -> type[ContainerRuntime] | None:
    """Determine the container runtime to use according to the environment variable `CONTAINER_RUNTIME`.

    If not set, check for availability.

    Returns:
        Class object of the first available container runtime or the container runtime selected by the
        `CONTAINER_RUNTIME` environment variable if set.
    """
    runtimes_by_name = {rt.name: rt for rt in RUNTIMES}
    rt_name = os.environ.get("CONTAINER_RUNTIME")
    rt = runtimes_by_name.get(rt_name) if rt_name is not None else None
    if rt is not None:
        return rt
    for rt in RUNTIMES:
        if rt.is_available():
            return rt
    return None


class ContainerRuntime:
    """Abstraction over container runtimes."""

    name: str
    executable: str

    @classmethod
    def is_available(cls) -> bool:
        """Check if the container runtime is installed and executable.

        Returns:
            `True` if the container runtime appears to be installed on the system and if required daemons are running,
            `false` otherwise.
        """
        try:
            return (
                subprocess.run(
                    [cls.executable, "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
                ).returncode
                == 0
            )
        except FileNotFoundError:
            return False

    @classmethod
    def pull(cls, image: str) -> None:
        """Pull an image.

        Args:
            image: Container image to pull.
        """
        raise NotImplementedError("Must be implemented by a subclass")

    @classmethod
    def run(
        cls,
        command: list[str],
        image: str,
        mounts: Sequence[tuple[str, str]] = (),
        env: dict[str, str] | None = None,
        name: str | None = None,
    ) -> subprocess.CompletedProcess:
        """Run a command in a container.

        Args:
            command: Command (as a list of strings) to run inside the container.
            image: Container image to use.
            mounts: Pairs of host folder and destination folder inside the container.
            env: Mapping of environment variable names and values to set inside the container.
            name: Optional name for the container (not supported by all runtimes).
        """
        raise NotImplementedError("Must be implemented by a subclass")


class Docker(ContainerRuntime):
    """Docker 🐋 runtime."""

    name = "docker"
    executable = "docker"

    @classmethod
    def is_available(cls) -> bool:
        """Check if the container runtime is installed and executable.

        Returns:
            `True` if the container runtime appears to be installed on the system and if required daemons are running,
            `false` otherwise.
        """
        try:
            return (
                subprocess.run(
                    [cls.executable, "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
                ).returncode
                == 0
            )
        except FileNotFoundError:
            return False

    @classmethod
    def pull(cls, image: str) -> None:
        """Pull an image.

        Args:
            image: Container image to pull.
        """
        subprocess.run([cls.executable, "pull", image], check=True)

    @classmethod
    def run(
        cls,
        command: list[str],
        image: str,
        mounts: Sequence[tuple[str, str]] = (),
        env: dict[str, str] | None = None,
        name: str | None = None,
    ) -> subprocess.CompletedProcess:
        """Run a command in a container.

        Args:
            command: Command (as a list of strings) to run inside the container.
            image: Container image to use.
            mounts: Pairs of host folder and destination folder inside the container.
            env: Mapping of environment variable names and values to set inside the container.
            name: Optional name for the container (not supported by all runtimes).
        """
        name_args = [] if name is None else ["--name", name]
        mount_args = (["--mount", f"type=bind,src={src},target={target}"] for src, target in mounts)
        env_args = (["--env", f"{var}={value}"] for var, value in (env or {}).items())

        return subprocess.run(
            [
                cls.executable,
                "run",
                "--rm",
                *name_args,
                *(arg for args in mount_args for arg in args),
                *(arg for args in env_args for arg in args),
                image,
                *command,
            ],
            check=False,
        )


class Podman(Docker):
    """Podman 🦭 runtime."""

    name = "podman"
    executable = "podman"

    @classmethod
    def is_available(cls) -> bool:
        """Check if the container runtime is installed and executable.

        Returns:
            `True` if the container runtime appears to be installed on the system and if required daemons are running,
            `false` otherwise.
        """
        # `podman info` seems to take some more time than `docker info`.
        # Just use `podman --help` here.
        try:
            return (
                subprocess.run(
                    [cls.executable, "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
                ).returncode
                == 0
            )
        except FileNotFoundError:
            return False


class Singularity(ContainerRuntime):
    """Singularity / Apptainer."""

    name = "singularity"
    executable = "singularity"

    @classmethod
    def pull(cls, image: str) -> None:
        """Pull an image.

        Args:
            image: Container image to pull.
        """
        if "://" not in image:
            image = "docker://" + image
        subprocess.run([cls.executable, "pull", image], check=True)

    @classmethod
    def run(
        cls,
        command: list[str],
        image: str,
        mounts: Sequence[tuple[str, str]] = (),
        env: dict[str, str] | None = None,
        _name: str | None = None,
    ) -> subprocess.CompletedProcess:
        """Run a command in a container.

        Args:
            command: Command (as a list of strings) to run inside the container.
            image: Container image to use.
            mounts: Pairs of host folder and destination folder inside the container.
            env: Mapping of environment variable names and values to set inside the container.
            name: Optional name for the container (not supported by all runtimes).
        """
        mount_args = (["--mount", f"type=bind,src={src},target={target}"] for src, target in mounts)
        env_args = (["--env", f"{var}={value}"] for var, value in (env or {}).items())
        if "://" not in image:
            image = "docker://" + image
        return subprocess.run(
            [
                cls.executable,
                "run",
                "--compat",
                *(arg for args in mount_args for arg in args),
                *(arg for args in env_args for arg in args),
                image,
                *command,
            ],
            check=False,
        )


RUNTIMES = [
    rt
    for rt in globals().values()
    if isinstance(rt, type) and issubclass(rt, ContainerRuntime) and rt is not ContainerRuntime
]


RUNTIME = runtime()
