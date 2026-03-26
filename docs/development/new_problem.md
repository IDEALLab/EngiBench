# Adding a new problem

Note:
   Read our [contribution guide](https://github.com/IDEALLab/EngiBench/blob/main/CONTRIBUTING.md) before diving into the code!

## Install
```{include} ../../README.md
:start-after: <!-- start dev-install -->
:end-before: <!-- end dev-install -->
```

### Code
In general, follow the `beams2d/` example.

1. Create a new problem module in [engibench/problems/](source:engibench/problems/) following the following layout (e.g. [engibench/problems/beams2d/](source:engibench/problems/beams2d/)), where you later also can add other versions / variant of the problem:
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

2. Define your problem class that implements the `Problem` interface with its functions and attributes in `problems/new_problem/v0.py` (e.g. [beams2d/v0.py](source:engibench/problems/beams2d/v0.py)).

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
2. If it is a new problem family, add a new `.md` file in [docs/problems/](source:docs/problems/) following
   the existing structure and add your problem family in the `toctree` of [docs/problems/index.md](source:docs/problems/index.md).
3. Add a problem markdown file to the `toctree` in `docs/problems/new_problem.md`. In the md file, use EngiBench's own `problem:table` and `problem:conditions` directives in the docs of your new problem:
   ``````md
   # Your Problem

   ``` {problem:table}
   :lead: Chuck Norris @chucknorris
   ```

   ...

   ## Conditions

   ``` {problem:conditions}
   ```

   ...
   ``````

   **`problem:table`**: This directive extracts metadata from a problem
   and inserts a table filled with the metadata.
   By default, the directive will try to import the problem
   `engibench.problems.<problem_id>`, where `<problem_id>` is the filename
   (without `.md` extension) of the markdown file the directive is used.

   Options (optional):
   * `:problem_id:` override `<problem_id>`,
   * `:lead:` Add a row "Lead" to the table, containing the specified value.
      If the value ends with `@username`, a link to `https://github.com/username` will be inserted.

   **`problem:conditions`**: This directive lists the conditions
   extracted from a problem as in the "Conditions" row produced by the
   `problem:table` directive. The `<problem_id>` is determined the same
   way as in `problem:table`.

   Options:
   * `:problem_id:` override `<problem_id>`,
   * `:defaults:` include default values in the list of conditions

   Here, `new_problem/__init__.py` is crucial as it makes the problem class discoverable to the `problem` directive by
   the reexport `from engibench.problems.new_problem.v0 import NewProblem`.
4. Add an image (result of `problem.render(design)`) in `docs/_static/img/problems`. The file's name should be `<new_problem>.png`, with your problem module as in the point above.
5. `cd docs/`
6. Run `sphinx-autobuild -b dirhtml --watch ../engibench --re-ignore "pickle$" . _build`
7. Go to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) and check if everything is fine.

Congrats! You can commit your changes and open a PR.
