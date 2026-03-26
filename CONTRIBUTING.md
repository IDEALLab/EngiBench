## How to Contribute to EngiBench

EngiBench aims to simplify research and adoption of ML for engineering design. We warmly welcome contributions from the community. Here are some ways you can help:

### Reporting a Bug

* First, check if the bug has already been reported in our [GitHub issues](https://github.com/IDEALLab/EngiBench/issues).
* If not, [open a new issue](https://github.com/IDEALLab/EngiBench/issues/new) and include:
  * A clear **title and description** of the problem.
  * Relevant details: system info, Python version, EngiBench version.
  * A **code sample** or **executable test case** that reproduces the issue.

### Fixing a Bug

* Open a GitHub pull request (PR) with your patch. Keep in mind that changing problem behavior should be minimized as that requires releasing a new version of the problem and makes results hard to compare across versions.
* Ensure your PR description clearly explains the problem and your solution. Reference the related issue if applicable.
* Verify your code passes all required checks (see our [PR template](https://github.com/IDEALLab/EngiBench/blob/main/.github/PULL_REQUEST_TEMPLATE.md) for details).

### Commit Message Format

We follow the rules of [conventional commit](https://www.conventionalcommits.org).
This will be checked in our CI by [conventional-commit-lint](https://gitlab.ethz.ch/sis/tools/conventional-commit-lint).
A [commit-msg hook](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks) is provided by our
[.pre-commit-config.yaml](./.pre-commit-config.yaml).

### Documentation or Cosmetic Changes

* Open a PR for fixes such as typos, formatting, or minor documentation updates.
* If no code is changed, running the full test suite is not required (`ruff format` is still recommended).

### Adding a New Feature or Problem

Before starting development, **please reach out to us**:

* **Features**: Open an issue describing the feature, expected behavior, and proposed API. Keep in mind that features should scale to the general problem set; we do not include features tied to a specific simulator in the core API.
* **Problems**: Problems are carefully vetted before inclusion. You have two options depending on how you want to use or share a problem:

  - **Contribute to EngiBench**:
    If you believe your problem would be broadly useful to the community and is stable and clean enough to be maintained long-term, please [open an issue](https://github.com/IDEALLab/EngiBench/issues) describing the problem, its relevance, and (if applicable) linking to the associated paper.

  - **Link an external problem**:
    If your problem is interesting but does not meet the inclusion criteria for the main repository, you can fork our repo and follow [this guide](https://engibench.ethz.ch/tutorials/new_problem/) to add it in your own copy of EngiBench. You can then share it with the community by linking it in our documentation via a PR that modifies [this file](https://github.com/IDEALLab/EngiBench/blob/main/docs/problems/external.md).
