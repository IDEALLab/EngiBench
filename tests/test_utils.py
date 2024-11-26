import os

from engibench.utils.files import clone_dir
from engibench.utils.files import replace_template_values


def test_clone_template() -> None:
    """Test the clone_template function.

    This tests the cloning of template directory to a study directory and the replacement of values in the template files."""
    template_dir = "tests/templates"
    study_dir = "tests/test_study"

    # Cloning
    clone_dir(template_dir, study_dir)

    # Replacement
    replace_template_values(study_dir + "/template_1.py", {"hello": "hello", "world": "world"})
    # Tests the replacement
    with open(study_dir + "/template_1.py") as f:
        content = f.read()
    assert content == "hello = hello\nworld = world\nprint(hello, world)\n"

    # Another file, in another format
    replace_template_values(study_dir + "/template_2.yml", {"hi": "hello", "world": "world"})
    with open(study_dir + "/template_2.yml") as f:
        content = f.read()
    assert content == "hi: hello\nworld: world\n"

    # Cleanup
    os.remove(study_dir + "/template_1.py")
    os.remove(study_dir + "/template_2.yml")
    os.rmdir(study_dir)
