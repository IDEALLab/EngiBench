# EngiBench-docs

This folder contains the documentation for EngiBench.

## Instructions for modifying environment pages

### Editing a problem page

Fork EngiBench and edit the docstring in the problem's Python file. Then, pip install your fork.

### Adding a new problem

Ensure the problem is in EngiBench (or your fork). Ensure that its Python file has a properly formatted markdown docstring. Install using `pip install -e .[doc]` and add a markdown file in the [./problems/](problems/) of the repo.
Use Engibench's own `problem` directive in the docs of your new problem:
   ``````md
   # Your Problem

   ``` {problem} your_problem
   ```
   ``````

Here, `your_problem` must match the name of the module where your problem class is defined.
This will automatically include the docstrings of your `Problem` class as well as a table with its metadata. Then complete the [other steps](#other-steps).

#### Other steps

- Edit `docs/problems/index.md`, and add the name of the file corresponding to your new environment to the `toctree`.

## Build the Documentation

Install the required packages and EngiBench (or your fork):

```
pip install engibench[doc]
```

To build the documentation once:

```
cd docs
make dirhtml
```

To rebuild the documentation automatically every time a change is made:

```
cd docs
sphinx-autobuild -b dirhtml --watch ../engibench --re-ignore "pickle$" . _build
```

You can then open http://localhost:8000 in your browser to watch a live updated version of the documentation.

## Writing Tutorials

We use Sphinx-Gallery to build the tutorials inside the `docs/tutorials` directory. Check `docs/tutorials/demo.py` to see an example of a tutorial and [Sphinx-Gallery documentation](https://sphinx-gallery.github.io/stable/syntax.html) for more information.

To convert Jupyter Notebooks to the python tutorials you can use [this script](https://gist.github.com/mgoulao/f07f5f79f6cd9a721db8a34bba0a19a7).

If you want Sphinx-Gallery to execute the tutorial (which adds outputs and plots) then the file name should start with `run_`. Note that this adds to the build time so make sure the script doesn't take more than a few seconds to execute.
