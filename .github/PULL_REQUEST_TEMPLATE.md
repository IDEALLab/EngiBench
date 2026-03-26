# Description

Please include a summary of the change and which issue is fixed. Please also include relevant motivation and context. List any dependencies that are required for this change.

Fixes # (issue)

## Type of change

Please delete options that are not relevant.

- [ ] Documentation only change (no code changed)
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] This change requires a documentation update

### Screenshots

Please attach before and after screenshots of the change if applicable.

<!--
Example:

| Before | After |
| ------ | ----- |
| _gif/png before_ | _gif/png after_ |


To upload images to a PR -- simply drag and drop an image while in edit mode and it should upload the image directly. You can then paste that source into the above before/after sections.
-->

# Checklist:

- [ ] I have run the [`pre-commit` checks](https://pre-commit.com/) with `pre-commit run --all-files`
- [ ] I have run `ruff check .` and `ruff format`
- [ ] I have run `mypy .`
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes

<!--
As you go through the checklist above, you can mark something as done by putting an x character in it

For example,
- [x] I have done this task
- [ ] I have not done this task
-->


# Reviewer Checklist:

- [ ] The content of this PR brings value to the community. It is not too specific to a particular use case.
- [ ] The tests and checks pass (linting, formatting, type checking). For a new problem, double check the github actions workflow to ensure the problem is being tested.
- [ ] The documentation is updated.
- [ ] The code is understandable and commented. No large code blocks are left unexplained, no huge file. Can I read and understand the code easily?
- [ ] There is no merge conflict.
- [ ] The changes are not breaking the existing results (datasets, training curves, etc.). If they do, is there a good reason for it? And is the associated problem version bumped?
- [ ] For a new problem, has the dataset been generated with our slurm script so we can re-generate it if needed? (This also ensures that the problem is running on the HPC.)
- [ ] For bugfixes, it is a robust fix and not a hacky workaround.
