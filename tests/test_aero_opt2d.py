import os

import pytest

from engibench.problems.airfoil2d.v0 import Airfoil2D

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_airfoil2d() -> None:
    """Test the Airfoil2D problem."""
    problem = Airfoil2D()
    problem.reset(seed=0, cleanup=False)

    # Get design and conditions from the dataset
    design = problem.random_design()
    problem.optimize(design, mpicores=1)
