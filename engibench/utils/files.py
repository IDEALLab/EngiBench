import os
from string import Template
from typing import Any


def _create_study_dir(study_dir: str) -> None:
    """Create a directory for the study.

    Args:
        study_dir (str): Path to the study directory.

    """
    if not os.path.exists(study_dir):
        os.makedirs(study_dir)


def clone_template(template_dir: str, study_dir: str) -> None:
    """Clone the template directory to the study directory.

    Args:
        template_dir (str): Path to the template directory.
        study_dir (str): Path to the study directory.

    """
    _create_study_dir(study_dir)
    for root, _, files in os.walk(template_dir):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, template_dir)
            study_file_path = os.path.join(study_dir, rel_path)
            if not os.path.exists(os.path.dirname(study_file_path)):
                os.makedirs(os.path.dirname(study_file_path))
            with open(file_path) as f:
                content = f.read()
            with open(study_file_path, "w") as f:
                f.write(content)


def replace_template_values(template_fname: str, values: dict[str, Any]) -> None:
    """Replace values in a template file.

    Args:
        template_fname (str): Path to the template file.
        values (dict[str, Any]): Dictionary with the values to replace.
    """
    with open(template_fname) as f:
        template = Template(f.read())
        content = template.substitute(values)
    with open(template_fname, "w") as f:
        f.write(content)
