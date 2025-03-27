"""Utility functions for file operations."""

import os
import shutil
from string import Template
from typing import Any


def _create_study_dir(study_dir: str) -> None:
    """Create a directory for the study.

    Args:
        study_dir (str): Path to the study directory.

    """
    print(f"Creating study directory: {study_dir}")
    if not os.path.exists(study_dir):
        os.makedirs(study_dir)


def clone_dir(source_dir: str, target_dir: str) -> None:
    """Clone the source directory to the study directory.

    Args:
        source_dir (str): Path to the source directory.
        target_dir (str): Path to the target directory.

    """
    if not os.path.exists(source_dir):
        msg = f"Source directory does not exist: {source_dir}"
        raise FileNotFoundError(msg)

    _create_study_dir(target_dir)
    print(f"Copying files from {source_dir} to {target_dir}")

    try:
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, source_dir)
                study_file_path = os.path.join(target_dir, rel_path)

                try:
                    if not os.path.exists(os.path.dirname(study_file_path)):
                        os.makedirs(os.path.dirname(study_file_path))
                    # Use shutil.copy2 to preserve metadata including permissions
                    shutil.copy2(file_path, study_file_path)
                except OSError as e:
                    msg = f"Failed to copy file {file_path} to {study_file_path}: {e!s}"
                    raise RuntimeError(msg) from e
    except Exception as e:
        msg = f"Failed to clone directory from {source_dir} to {target_dir}: {e!s}"
        raise RuntimeError(msg) from e


def replace_template_values(template_fname: str, values: dict[str, Any]) -> None:
    """Replace values in a template file.

    Args:
        template_fname (str): Path to the template file.
        values (dict[str, Any]): Dictionary with the values to replace.
    """
    if not os.path.exists(template_fname):
        msg = f"Template file does not exist: {template_fname}"
        raise FileNotFoundError(msg)

    try:
        with open(template_fname) as f:
            template = Template(f.read())
            try:
                content = template.substitute(values)
            except KeyError as e:
                msg = f"Missing required template value: {e}"
                raise ValueError(msg) from e
            except ValueError as e:
                msg = f"Invalid template value: {e}"
                raise ValueError(msg) from e

        with open(template_fname, "w") as f:
            f.write(content)
    except OSError as e:
        msg = f"Failed to process template file {template_fname}: {e!s}"
        raise RuntimeError(msg) from e
