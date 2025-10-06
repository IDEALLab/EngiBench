"""Abstraction over container runtimes."""

from collections.abc import Sequence
import os
import shutil
import subprocess
import tempfile


def pull(image: str) -> None:
    """Pull an image using the selected runtime.

    Args:
        image: Container image to pull.
    """
    if RUNTIME is None:
        msg = "No container runtime found"
        raise FileNotFoundError(msg)

    RUNTIME.pull(image)


def run(  # noqa: PLR0913
    command: list[str],
    image: str,
    mounts: Sequence[tuple[str, str]] = (),
    env: dict[str, str] | None = None,
    name: str | None = None,
    *,
    sync_uid: bool = False,
) -> None:
    """Run a command in a container using the selected runtime.

    Args:
        command: Command (as a list of strings) to run inside the container.
        image: Container image to use.
        mounts: Pairs of host folder and destination folder inside the container.
        env: Mapping of environment variable names and values to set inside the container.
        name: Optional name for the container (not supported by all runtimes).
        sync_uid: Use the uid of the current process as uid inside the container.
    """
    if RUNTIME is None:
        msg = "No container runtime found. Please ensure Docker, Podman, or Singularity is installed and running."
        raise FileNotFoundError(msg)

    try:
        result = RUNTIME.run(command, image, mounts, env, name, sync_uid=sync_uid)
        result.check_returncode()
    except subprocess.CalledProcessError as e:
        msg = f"""Container command failed with exit code {e.returncode}:
Command: {" ".join(command)}
stdout: {result.stdout.decode() if result.stdout else "No output"}
stderr: {result.stderr.decode() if result.stderr else "No output"}"""
        raise RuntimeError(msg) from e


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
    def run(  # noqa: PLR0913
        cls,
        command: list[str],
        image: str,
        mounts: Sequence[tuple[str, str]] = (),
        env: dict[str, str] | None = None,
        name: str | None = None,
        *,
        sync_uid: bool = False,
    ) -> subprocess.CompletedProcess:
        """Run a command in a container.

        Args:
            command: Command (as a list of strings) to run inside the container.
            image: Container image to use.
            mounts: Pairs of host folder and destination folder inside the container.
            env: Mapping of environment variable names and values to set inside the container.
            name: Optional name for the container (not supported by all runtimes).
            sync_uid: Use the uid of the current process as uid inside the container.
        """
        raise NotImplementedError("Must be implemented by a subclass")


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
    def run(  # noqa: PLR0913
        cls,
        command: list[str],
        image: str,
        mounts: Sequence[tuple[str, str]] = (),
        env: dict[str, str] | None = None,
        name: str | None = None,
        *,
        sync_uid: bool = False,
    ) -> subprocess.CompletedProcess:
        """Run a command in a container.

        Args:
            command: Command (as a list of strings) to run inside the container.
            image: Container image to use.
            mounts: Pairs of host folder and destination folder inside the container.
            env: Mapping of environment variable names and values to set inside the container.
            name: Optional name for the container (not supported by all runtimes).
            sync_uid: Use the uid of the current process as uid inside the container.
        """
        name_args = [] if name is None else ["--name", name]
        user_args = cls._user_args() if sync_uid else ()

        return subprocess.run(
            [
                cls.executable,
                "run",
                "--rm",
                *name_args,
                *_mount_args(mounts),
                *_env_args(env or {}),
                *user_args,
                image,
                *command,
            ],
            check=False,
            capture_output=True,
            env=cls._env(),
        )

    @classmethod
    def _user_args(cls) -> tuple[str, ...]:
        return ("--user", str(os.getuid()))

    @classmethod
    def _env(cls) -> dict[str, str] | None:
        return None


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

    @classmethod
    def _user_args(cls) -> tuple[str, ...]:
        return ("--userns=keep-id", "--user", str(os.getuid()))

    @classmethod
    def _env(cls) -> dict[str, str] | None:
        # podman needs to have pasta in the PATH variable to configure
        # network namespaces
        pasta_executable = shutil.which("pasta")
        if pasta_executable is None:
            msg = "pasta executable not available. This is needed for podman to work properly"
            raise RuntimeError(msg)
        return {"PATH": os.path.dirname(pasta_executable)}


DOCKER_PREFIX = "docker://"


class Apptainer(ContainerRuntime):
    """Apptainer."""

    name = "apptainer"
    executable = "apptainer"

    @classmethod
    def _set_apptainer_env(cls) -> None:
        """Set Apptainer environment variables."""
        # See https://scicomp.ethz.ch/wiki/Apptainer#Settings
        # Set cache directory to SCRATCH if available, otherwise use default
        scratch_dir = os.environ.get("SCRATCH")
        if scratch_dir:
            # stores apptainer images in your $SCRATCH directory
            os.environ["APPTAINER_CACHEDIR"] = f"{scratch_dir}/.apptainer"

        # uses the local temporary directory to store temporary data when building images
        os.environ["APPTAINER_TMPDIR"] = os.environ.get("TMPDIR", tempfile.gettempdir())

    @classmethod
    def sif_filename(cls, image: str) -> str:
        """Construct the sif filename from an image specifier."""
        # Extract just the image part if it's a docker URI
        image = image.removeprefix(DOCKER_PREFIX)

        # Parse the image name to match Singularity's naming convention
        # For "mdolab/public:u22-gcc-ompi-stable", Singularity creates "public_u22-gcc-ompi-stable.sif"
        image_name = image.rsplit("/", 1)[-1] if "/" in image else image

        # Replace ":" with "_" in the image name
        return image_name.replace(":", "_") + ".sif"

    @classmethod
    def pull(cls, image: str) -> None:
        """Pull an image.

        Args:
            image: Container image to pull.
        """
        # Set Apptainer environment variables
        cls._set_apptainer_env()
        # Get sif filename
        sif_filename = cls.sif_filename(image)

        # Check if the image already exists
        if os.path.exists(sif_filename):
            print(f"Image file already exists: {sif_filename} - skipping pull")
            return
        # Convert to docker URI if needed
        docker_uri = DOCKER_PREFIX + image if "://" not in image else image
        # Image doesn't exist, proceed with pull
        subprocess.run([cls.executable, "pull", docker_uri], check=True)

    @classmethod
    def run(  # noqa: PLR0913
        cls,
        command: list[str],
        image: str,
        mounts: Sequence[tuple[str, str]] = (),
        env: dict[str, str] | None = None,
        name: str | None = None,  # noqa: ARG003
        *,
        sync_uid: bool = False,  # noqa: ARG003
    ) -> subprocess.CompletedProcess:
        """Run a command in a container.

        Args:
            command: Command (as a list of strings) to run inside the container.
            image: Container image to use.
            mounts: Pairs of host folder and destination folder inside the container.
            env: Mapping of environment variable names and values to set inside the container.
            name: Optional name for the container (not supported by all runtimes).
            sync_uid: Use the uid of the current process as uid inside the container.
        """
        # Set Apptainer environment variables
        cls._set_apptainer_env()

        # Get sif filename
        sif_image = cls.sif_filename(image)
        cls.pull(image)

        return subprocess.run(
            [
                cls.executable,
                "run",
                "--compat",
                *_mount_args(mounts),
                *_env_args(env or {}),
                sif_image,
                *command,
            ],
            check=False,
        )


def _mount_args(mounts: Sequence[tuple[str, str]]) -> list[str]:
    return [arg for args in (["--mount", f"type=bind,src={src},target={target}"] for src, target in mounts) for arg in args]


def _env_args(env: dict[str, str]) -> list[str]:
    return [arg for args in (["--env", f"{var}={value}"] for var, value in (env or {}).items()) for arg in args]


RUNTIMES = [
    rt
    for rt in globals().values()
    if isinstance(rt, type) and issubclass(rt, ContainerRuntime) and rt is not ContainerRuntime
]


RUNTIME = runtime()
