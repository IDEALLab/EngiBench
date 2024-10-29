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
import numpy as np

problem = Airfoil2D()
problem.reset(seed=0)

# Inspect problem
problem.design_space # Box(0.0, 1.0, (2, 192), float32)
problem.possible_objectives # frozenset({('lift', 'maximize'), ('drag', 'minimize')})

# Get the dataset
dataset = problem.dataset
first_design = np.array(dataset["features"][0])

# Evaluate a design
problem.design_to_simulator_input(design=first_design, filename="first_design")
problem.simulate(filename="first_design")

# Optimize a design
problem.optimize(starting_point="first_design")
```
