[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# EngiBench
Benchmarks for automated engineering design

## Installation
(!) You might need Docker or Singularity to run some of the benchmarks.

```bash
pip install -e .
```

## API

```python
from engibench.problems.airfoil2d import Airfoil2D

problem = Airfoil2D()
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
