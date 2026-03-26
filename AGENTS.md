# AGENTS instructions

Developer relevant documentation is kept in `README.md` as well as `docs/tutorials/new_problem.md`.
In the following, steps to set up the environment, run tests, build docs and follow the PR workflow are given.

## Project structure

- [engibench/core.py](engibench/core.py) — `Problem` base class, `OptiStep`, `ObjectiveDirection`
- [engibench/constraint.py](engibench/constraint.py) — constraint system
- [engibench/problems/](engibench/problems/)`<name>/` — one subpackage per problem, each exports a single `Problem` subclass
- [engibench/utils/](engibench/utils/) — container runtimes, slurm, CLI helpers, problem discovery
- [tests/](tests/) — pytest suite; [test_problem_implementations.py](tests/test_problem_implementations.py)
  parametrizes over all builtin problems
- [docs/](docs/) — Sphinx (MyST), built with `cd docs && make dirhtml`

## Setup of a local virtual environment

- Run `python -m venv .venv` to create a virtual environment.
- `. .venv/bin/activate` to activate the environment.
- The project itself, including all dependencies is installed in development mode by `pip install -e ".[dev]"`.

## Setup of pre-commit hooks

- Installation of pre-commit is done by `pip install pre-commit`.
- Setup of the hooks: `pre-commit install`.

## Running tests / Code quality assurance

- To run the tests, use `pytest`.
- Lint: `ruff check`.
- Check formatting by `ruff format --check`.
- Check typing consistency with `mypy .`.

## Building documentation

- Documentation can be built locally using `cd docs` followed by `make dirhtml`
  as described in [docs/README.md](./docs/README.md).
  `make dirhtml` will essentially call `sphinx-build . _build`.

## Pull request guidelines

- Follow the PR guidance in [CONTRIBUTING.md](CONTRIBUTING.md).
- Add tests when adding new features or fixing bugs.
- For commits, comply with [conventional commit messages](https://www.conventionalcommits.org/) (see next section)

## Conventional commits

- For commits, use the structure of [conventional commit messages](https://www.conventionalcommits.org/):
  ```
  <type>[optional scope]: <description>

  <optional body>

  <optional footer(s)>
  ```

### Valid types for `<type>` based on the changed files obtained by `git status` and
  the changes obtained by `git diff --staged`:
  - `feat`: A new feature
  - `fix`: A bug fix
  - `docs`: Documentation only changes, i.e. inside the directory [docs](docs)
  - `style`: Code style changes (i.e. reformatting due to a new version of [ruff](https://docs.astral.sh/ruff/))
  - `refactor`: Code change that neither fixes a bug nor adds a feature
  - `perf`: Performance improvements
  - `test`: Adding or updating tests in the [tests](tests) directory
  - `build`: Changes to build system or dependencies, i.e. [pyproject.toml](pyproject.toml)
  - `ci`: Changes to CI configuration [.github/](.github)
  - `chore`: Other changes that don't modify files in the directories [engibench](engibench), [docs](docs) or [tests](tests)
- `optional scope`: Optional but recommended. Describes the changed component or module name (enclosed in parentheses).
- `<description>`: Must follow the following rules:
   - MUST be lowercase (except proper nouns)
   - MUST be imperative mood ("add feature" not "added feature" or "adding feature")
   - MUST be max 72 characters
   - MUST NOT end with a period
- `<optional body>`: Optional but welcome. Is separated from the subject line (1st line)
  by a blank line and explains WHAT changed and WHY but not HOW. It is wrapped at 72 chars.
- `<optional footer(s)>`: Used for breaking changes. In this case it starts with `BREAKING CHANGE: `
  followed by a description the breaking change and how to migrate

### Validation of the commit message

- Use [conventional-commit-lint](https://gitlab.ethz.ch/sis/tools/conventional-commit-lint) as follows:
  `echo "<commit_message>" | conventional-commit-lint` and make sure the exit code is `0`.
