# Installation

⚠️ Some problems run under Docker or Singularity. Make sure you have one of them installed.

## From PyPI

```bash
pip install engibench
```

You can also specify additional dependencies for specific problems:

```bash
pip install "engibench[airfoil]"
```


## From source

Typically, developers will install the package from source, with all extras.

```bash
git clone git@github.com:IDEALLab/EngiBench.git
cd engibench
pip install -e ".[dev]"
```
