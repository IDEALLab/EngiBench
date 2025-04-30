
<p align="center">
<img src="docs/\_static/img/logo_text_large2.png" align="center" width="90%"/>
</p>

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
<!-- start new_problem -->
In general, follow the `beams2d/` example.

#### Code
1. Create a new problem module in [engibench/problems/](engibench/problems/) following the following layout (e.g. [engibench/problems/beams2d/](engibench/problems/beams2d/)), where you later also can add other versions / variant of the problem:
   ```
   📦 engibench
   └─ 📂 problems
      └─ 📂 new_problem
         ├── 📄 __init__.py
         └── 📄 v0.py
   ```

   `__init__.py`
   ```py
   """NewProblem problem module."""

    from engibench.problems.new_problem.v0 import NewProblem

    __all__ = ["NewProblem"]
   ```

   The `v0` module already proactively introduces versioning.

   Ideally, all non-breaking changes should not create a new versioned module.
   Also in many cases, code duplication can be avoided, by introducing a new parameter to the problem class.

2. Define your problem class that implements the `Problem` interface with its functions and attributes in `problems/new_problem/v0.py` (e.g. [beams2d/v0.py](engibench/problems/beams2d/v0.py)).

   `problems/new_problem/v0.py`
   ```py
   from engibench.core import Problem

   class NewProblem(Problem[...]) # <- insert type for DesignType here
       ... # define your problem here
   ```

   You can consult the documentation for info about the API; see below for how to build the website locally.
3. Run `pytest tests/test_problem_implementations.py` (requires `pip install ".[test]"`)
   to verify that the new `Problem` class defines all required metadata attributes.
4. Complete your docstring (Python documentation) thoroughly, LLMs + coding IDE will greatly help.

#### Documentation
1. Install necessary documentation tools: `pip install ".[doc]"`.
2. If it is a new problem family, add a new `.md` file in [docs/problems/](docs/problems/) following
   the existing structure and add your problem family in the `toctree` of [docs/problems/index.md](docs/problems/index.md).
3. Add a problem markdown file to the `toctree` in `docs/problems/new_problem.md`. In the md file, use EngiBench's own `problem` directive:
   ``````md
   # Your Problem

   ``` {problem} new_problem
   ```
   ``````

   Here, `new_problem` must match the name of the top level module where your problem class is defined.
   Here, `new_problem/__init__.py` is crucial as it makes the problem class discoverable to the `problem` directive by
   the reexport `from engibench.problems.new_problem.v0 import NewProblem`.
4. Add an image (result of `problem.render(design)`) in `docs/_static/img/problems`. The file's name should be `<new_problem>.png`, with your problem module as in the point above.
5. `cd docs/`
6. Run `sphinx-autobuild -b dirhtml --watch ../engibench --re-ignore "pickle$" . _build`
7. Go to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) and check if everything is fine.

Congrats! You can commit your changes and open a PR.
<!-- end new_problem -->

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
