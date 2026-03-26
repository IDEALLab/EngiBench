import os

from engibench.utils.files import clone_dir


def test_clone_template() -> None:
    """Test the clone_template function.

    This tests the cloning of template directory to a study directory and the replacement of values in the template files."""
    template_dir = "tests/templates"
    study_dir = "tests/test_study"

    # Cloning
    clone_dir(template_dir, study_dir)

    # Tests the replacement
    with open(study_dir + "/template_1.py") as f:
        content = f.read()
    assert (
        content
        == """hello = "hello"
world = "world"
print(hello, world)
"""
    )

    # Another file, in another format
    with open(study_dir + "/template_2.yml") as f:
        content = f.read()
    assert (
        content
        == """---
hi: hello
world: world
"""
    )

    # Cleanup
    os.remove(study_dir + "/template_1.py")
    os.remove(study_dir + "/template_2.yml")
    os.remove(study_dir + "/__init__.py")
    os.rmdir(study_dir)
