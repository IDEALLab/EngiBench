
<p align="center">
<img src="docs/\_static/img/logo_text_large2.png" align="center" width="90%"/>
</p>

![tests](https://github.com/IDEALLab/engibench/workflows/Python%20tests/badge.svg)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![code style: Ruff](
    https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](
    https://github.com/astral-sh/ruff)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ideallab/engibench/blob/main/tutorial.ipynb)


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
problem.possible_objectives  # (('cd', 'minimize'), ('cl', 'maximize'))
problem.boundary_conditions  # frozenset({('marchDist', 100.0), ('s0', 3e-06)})
problem.dataset # A HuggingFace Dataset object

# Train your inverse design or surrogate model and use it to predict/optimize designs!
for i in range(100):
    desired_objs = ...
    my_design = model.predict(desired_objs) # replace with your model
    # Evaluate a design using a simulator
    objs = problem.simulate(design=my_design, config={"mach": 0.2, "reynolds": 1e6})

# or optimize a design using an integrated optimizer if available!
opt_design, opt_history = problem.optimize(starting_point=my_design, config={"mach": 0.2, "reynolds": 1e6})
```

You can also play with the API here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ideallab/engibench/blob/main/tutorial.ipynb)

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
1. Create a new problem folder in `engibench/problems/` following the existing structure, e.g., `engibench/problems/airfoil2d/`.
2. Add an `__init__.py` file in the same directory. Leave it empty.
3. Create your environment file, e.g. `airfoil2d/airfoil2d.py`. Inside it, define a class that implements the `Problem` interface and its functions and attributes. You can consult the documentation for info about the API; see below for how to build the website locally.
4. Complete your docstring (Python documentation) thoroughly, LLMs + coding IDE will greatly help.
5. Add a `build` function to the problem file that returns an instance of your problem. This is essentially a copy-paste of the airfoil example.
6. Add a file `<my_problem>_v0.py` in the folder. It just exposes the `build` function. Same, you can copy the airfoil example.
7. Add your problem in `utils/all_problems.py` to register it. This is just adding a line and the import for your own problem.

#### Documentation
1. Install necessary documentation tools: `pip install .[doc]`.
2. Run `python docs/_scripts/gen_problems_docs.py` and pray.
3. If it is a new problem family, add a new `.md` file in [docs/problems/](docs/problems/) following
   the existing structure and add your problem family in the `toctree` of [docs/problems/index.md](docs/problems/index.md).
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
