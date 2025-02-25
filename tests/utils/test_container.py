import pytest

from engibench.utils import container

available_runtimes = [rt for rt in container.RUNTIMES if rt.is_available()]


@pytest.mark.parametrize("runtime", available_runtimes)
def test_run_singularity_sets_correct_environment(runtime: type[container.ContainerRuntime]) -> None:
    """Test if singularity can run a container with an environment variable."""

    runtime.run(command=["sh", "-c", "[ $TEST_VAR = test ]"], env={"TEST_VAR": "test"}, image="alpine").check_returncode()


@pytest.mark.parametrize("runtime", available_runtimes)
def test_run_singularity_mounts_files(runtime: type[container.ContainerRuntime]) -> None:
    """Test if singularity can run a container with a mount."""

    check_string = "A string which appears in this file"

    runtime.run(
        command=["grep", check_string, "/mnt/test.py"],
        mounts=[(__file__, "/mnt/test.py")],
        image="alpine",
    ).check_returncode()
