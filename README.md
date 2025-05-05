
<p align="center">
<img src="docs/\_static/img/logo_text_large2.png" align="center" width="90%"/>
</p>

[![Python](https://img.shields.io/pypi/pyversions/engibench.svg)](https://badge.fury.io/py/engibench)
![tests](https://github.com/IDEALLab/engibench/workflows/Python%20tests/badge.svg)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![code style: Ruff](
    https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](
    https://github.com/astral-sh/ruff)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ideallab/engibench/blob/main/tutorial.ipynb)


<!-- start elevator-pitch -->
EngiBench offers a collection of engineering design problems, datasets, and benchmarks to facilitate the development and evaluation of optimization and machine learning algorithms for engineering design. Our goal is to provide a standard API to enable researchers to easily compare and evaluate their algorithms on a wide range of engineering design problems.
<!-- end elevator-pitch -->

## Installation
⚠️ Some problems run under Docker or Singularity. Others require native installation of dependencies, please consult the documentation of the specific problem.

<!-- start install -->
```bash
pip install engibench
```

You can also specify additional dependencies for specific problems:

```bash
pip install "engibench[beams2d]"
```

Or you can install all dependencies for all problems:

```bash
pip install "engibench[all]"
```
<!-- end install -->

## API

<!-- start api -->
```python
from engibench.problems.beams2d.v0 import Beams2D

# Create a problem
problem = Beams2D()

# Inspect problem
problem.design_space  # Box(0.0, 1.0, (50, 100), float64)
problem.objectives  # (("compliance", "MINIMIZE"),)
problem.conditions  # (("volfrac", 0.35), ("forcedist", 0.0),...)
problem.dataset # A HuggingFace Dataset object

# Train your models, e.g., inverse design
# inverse_model = train_inverse(problem.dataset)
desired_conds = {"volfrac": 0.7, "forcedist": 0.3}
# generated_design = inverse_model.predict(desired_conds)

random_design, _ = problem.random_design()
# check constraints on the design, config pair
violated_constraints = problem.check_constraints(design=random_design, config=desired_conds)

if not violated_constraints:
   # Only simulate to get objective values
   objs = problem.simulate(design=random_design, config=desired_conds)
   # Or run a gradient-based optimizer to polish the design
   opt_design, history = problem.optimize(starting_point=random_design, config=desired_conds)
```

You can also play with the API here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ideallab/engibench/blob/main/tutorial.ipynb)

<!-- end api -->

## Development

Both EngiBench and EngiOpt are open source projects and we welcome contributions! If you want to add a new problem, please reach out to us first to see if it is a good fit for EngiBench.

### Installation
<!-- start dev-install -->
To install EngiBench for development, clone the repo, install the pre-commit hooks, and install all dev dependencies:

```bash
git clone git@github.com:IDEALLab/EngiBench.git
cd EngiBench
pre-commit install
pip install -e ".[dev]"
```
Also worth installing [`ruff`](https://docs.astral.sh/ruff/) and [`mypy`](https://www.mypy-lang.org/) in your editor as we are checking the code style and type safety on our CI.
<!-- end dev-install -->

### Adding a new problem
See [docs/tutorials/new_problem.md](docs/tutorials/new_problem.md).

## License

The code of EngiBench and EngiOpt is licensed under the GPLv3 license. See the [LICENSE](LICENSE) file for details.
All the associated datasets are licensed under the CC-BY-NC-SA 4.0 license.

## Citing

<!-- start citing -->
If you use EngiBench in your research, please cite the following paper:

```bibtex
TODO
```
<!-- end citing -->
