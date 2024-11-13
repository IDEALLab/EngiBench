
<p align="center">
<img src="docs/\_static/img/logo_text_large2.png" align="center" width="90%"/>
</p>

![tests](https://github.com/IDEALLab/engibench/workflows/Python%20tests/badge.svg)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<!-- start elevator-pitch -->
EngiBench offers a collection of engineering design problems, datasets, and benchmarks to facilitate the development and evaluation of optimization and machine learning algorithms for engineering design. Our goal is to provide a standard API to enable researchers to easily compare and evaluate their algorithms on a wide range of engineering design problems.
<!-- end elevator-pitch -->

## Installation
<!-- start install -->
⚠️ You might need Docker or Singularity to run some of the benchmarks.

```bash
pip install -e .
```

If you want to install with additional dependencies for a specific envs, e.g., airfoil2d, run:

```bash
pip install -e ".[airfoil2d]"
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
problem.boundary_conditions  # [('marchDist', 100.0), ('s0', 3e-06)]
problem.dataset # A HuggingFace Dataset object

# Train your model and use it to predict designs!
for i in range(100):
    desired_objs = ...
    my_design = model.predict(desired_objs)
    # Evaluate a design using a simulator
    objs = problem.simulate(design=my_design, config={"mach": 0.2, "reynolds": 1e6})

# or optimize a design if available!
opt_design, objs = problem.optimize(starting_point=my_design, config={"mach": 0.2, "reynolds": 1e6})
```
<!-- end api -->

## Development

Clone the repo and install the pre-commit hooks:

```bash
git clone git@github.com:IDEALLab/EngiBench.git
cd EngiBench
pre-commit install
```

### Adding a new problem
<!-- start new_problem -->
In general, follow the `airfoil2d/` example.

#### Code
1. Create a new problem file in `engibench/problems/` following the existing structure.
2. Add the problem to the `__init__.py` file in the same directory.
3. Create your environment file and class, implementing the `Problem` interface. Complete your docstring thoroughly, LLMs + coding IDE will help a lot.
4. Add a `build` function to the problem file that returns an instance of your problem.
5. Add a file `my_problem_v0.py` in the folder. It just exposes the `build` function.
6. Add your problem in `utils/all_problems.py` to register it.

#### Documentation
1. `cd docs` and run `pip install -r requirements.txt` to install the necessary packages.
2. Run `python _scripts/gen_problems_docs.py` and pray.
3. If it is a new problem family, add a new `.md` file in `docs/problems/` following the existing structure and add your problem family in the `toctree` of `docs/problems/index.md`.
4. Add your problem markdown file to the `toctree` in `docs/problems/your_problem_family.md`.
5. Run `sphinx-autobuild -b dirhtml --watch ../engibench --re-ignore "pickle$" . _build`
6. Go to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) and check if everything is fine.

Congrats! You can commit your changes and open a PR.
<!-- end new_problem -->

## Citing

<!-- start citing -->
If you use EngiBench in your research, please cite the following paper:

```bibtex
TODO
```
<!-- end citing -->
