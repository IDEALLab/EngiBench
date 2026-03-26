"""Shared functionality between 2D and 3D."""

import os
from pathlib import Path
import shutil
from typing import Any

from engibench.utils import cli
from engibench.utils import container


def run_container_script(
    image: str,
    script: Path,
    args: tuple[Any, ...],
    output_path: str,
    stdin: bytes | None = None,
) -> str:
    """Run a script inside container."""
    output_path = f"templates/output_by_pid/{os.getpid()}/{output_path}"
    cli_module_path = cli.__file__

    # Copy required stuff to a templates subfolder.
    # We cannot just mount, as the engibench folder might not be a subfolder of
    # `os.getcwd()`.
    os.makedirs("templates/engibench/utils", exist_ok=True)
    shutil.copy(cli_module_path, "templates/engibench/utils")
    shutil.copy(script, "templates/")

    container.run(
        command=[
            "python3",
            f"/home/fenics/shared/templates/{script.name}",
            *(str(arg) for arg in args),
            "/home/fenics/shared/" + output_path,
        ],
        image=image,
        name="dolfin",
        mounts=[
            (os.getcwd(), "/home/fenics/shared"),
            (cli_module_path, "/home/fenics/shared/templates/engibench/utils/cli.py"),
        ],
        stdin=stdin,
    )
    return output_path


def load_float(path: str) -> float:
    """Load a float from a file."""
    with open(path, "rb") as fp:
        return float(fp.read())
