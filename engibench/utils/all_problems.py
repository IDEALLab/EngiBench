"""Registry of all problems in EngiBench."""

from engibench.problems.airfoil2d import airfoil2d_v0
from engibench.problems.mto3d import mto3d_v0

all_problems = {
    "airfoil2d_v0": airfoil2d_v0,
    "mto3d_v0": mto3d_v0,
}
