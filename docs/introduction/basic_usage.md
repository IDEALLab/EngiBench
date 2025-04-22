# Basic Usage

Our API is designed to be simple and easy to use. Here is a basic example of how to use EngiBench to create a problem, get the dataset, and evaluate a design.

```python
from engibench.problems.beams2d.v0 import Beams2D

# Create a problem
problem = Beams2D()

# Inspect problem
problem.design_space  # Box(0.0, 1.0, (50, 100), float64)
problem.objectives  # (("compliance", "MINIMIZE"),)
problem.conditions  # (("volfrac", 0.35), ("forcedist", 0.0),...)
problem.dataset # A HuggingFace Dataset object

# Train your inverse design model or surrogate model
conditions = problem.dataset["train"].select_columns(problem.conditions_keys)
designs = problem.dataset["train"].select_columns("optimal_design")
cond_designs_keys = problem.conditions_keys + ["optimal_design"]
cond_designs = problem.dataset["train"].select_columns(cond_designs_keys)
objs = problem.dataset["train"].select_columns(problem.objectives_keys)

# Train your models
inverse_model = train_inverse(inputs=conditions, outputs=designs)
surr_model = train_surrogate(inputs=cond_designs, outputs=objs)

# Use the model predictions, inverse design here
desired_conds = {"volfrac": 0.7, "forcedist": 0.3}
generated_design = inverse_model.predict(desired_conds)
# Only simulate to get objective values
objs = problem.simulate(design=generated_design, config=desired_conds)
# Or run a gradient-based optimizer to polish the generate design
opt_design, history = problem.optimize(starting_point=generated_design, config=desired_conds)
```

You can also play with the API here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ideallab/engibench/blob/main/tutorial.ipynb)


Under the hood, the design representation is converted into a format that the simulator can understand. The simulator then evaluates or optimizes the design and returns the results. Note that the underlying simulators are often written in other languages and necessitate running in containerized environments. This is completely abstracted away from the user, who only needs to provide the design and the configuration.
