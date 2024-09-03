import pathlib

from setuptools import setup

CWD = pathlib.Path(__file__).absolute().parent


def get_version():
    """Gets the current version of the lib (avoids double sourcing in code and pyproject.toml)."""
    path = CWD / "engibench" / "__init__.py"
    content = path.read_text()
    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


print("aaaaaaaaa", get_version())
setup(name="engibench", version=get_version(), long_description=open("README.md").read())
