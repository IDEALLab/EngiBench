![tests](https://github.com/IDEALLab/engibench/workflows/Python%20tests/badge.svg)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
<img src="docs/\_static/img/logo_text_large.png" align="center" width="25%"/>
</p>


<!-- start elevator-pitch -->
EngiBench offers a collection of engineering design problems, datasets, and benchmarks to facilitate the development and evaluation of optimization and machine learning algorithms for engineering design. Our goal is to provide a standard API to enable researchers to easily compare and evaluate their algorithms on a wide range of engineering design problems.
<!-- end elevator-pitch -->

## Installation
<!-- start install -->
⚠️ You might need Docker or Singularity to run some of the benchmarks.

```bash
pip install -e .
```
<!-- end install -->

## API

<!-- start api -->
```python
from engibench.problems.airfoil2d import airfoil2d_v0

# Create a problem
problem = airfoil2d_v0.build()
problem.reset(seed=0)

# Inspect problem
problem.design_space  # Box(0.0, 1.0, (2, 192), float32)
problem.possible_objectives  # frozenset({('lift', 'maximize'), ('drag', 'minimize')})

# Get the dataset
dataset = problem.dataset
# Train your model and use it to predict designs!
my_design = model.predict(desired_objs)

# Evaluate a design using a simulator
objs = problem.simulate(design=my_design, config={"mach": 0.2, "reynolds": 1e6})
# or optimize a design if available!
opt_design, objs = problem.optimize(starting_point=my_design, config={"mach": 0.2, "reynolds": 1e6})
```
<!-- end api -->

## Citing

<!-- start citing -->
If you use EngiBench in your research, please cite the following paper:

```bibtex
TODO
```
<!-- end citing -->
