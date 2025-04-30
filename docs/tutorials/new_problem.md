# Adding a new problem

## Install
```{include} ../../README.md
:start-after: <!-- start dev-install -->
:end-before: <!-- end dev-install -->
```

### Code
In general, follow the `beams2d/` example.

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
