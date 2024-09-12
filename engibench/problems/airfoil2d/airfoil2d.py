"""Airfoil 2D problem.

This problem simulates the performance of an airfoil in a 2D environment. An airfoil is represented by a set of 192 points that define its shape. The performance is evaluated by a simulator that computes the lift and drag coefficients of the airfoil.
"""

from typing import ClassVar, overload

from datasets import load_dataset
from gymnasium import spaces
import numpy as np

from engibench.core import Problem


class Airfoil2D(Problem):
    r"""Airfoil 2D problem.

    This problem simulates the performance of an airfoil in a 2D environment. The airfoil is represented by a set of 192
    points that define its shape. The performance is evaluated by a simulator that computes the lift and drag coefficients
    of the airfoil.

    The design space is represented by a 3D numpy array (vector of 192 x,y coordinates per design) that define the airfoil shape.

    The input space is a dictionary with the following keys:
    - "n_points" (int): The number of points that define the airfoil shape.
    - "naca" (str): The NACA code that defines the airfoil shape.

    The objectives are:
    - "lift": maximize
    - "drag": minimize
    """

    input_space = None
    possible_objectives: ClassVar[dict[str, str]] = {
        "lift": "maximize",
        "drag": "minimize",
    }
    design_space = spaces.Box(low=0.0, high=1.0, shape=(2, 192), dtype=np.float32)
    str_id = "ffelten/test2"

    def __init__(self, objectives: tuple[str, str] = ("lift", "drag")) -> None:
        """Initialize the problem."""
        super().__init__()
        self.reset()
        self.objectives: set[str] = set(objectives)

    def design_to_simulator_input(self, design: np.ndarray, filename: str = "design") -> None:
        """Converts a design to a simulator input.

        The simulator inputs are two files: a mesh file (.cgns) and a FFD file (.xyz). This function generates these files from the design.
        The files are saved in the current directory with the name "$filename.cgns" and "$filename_ffd.xyz".

        Args:
            design (np.ndarray): The design to convert.
            filename (str): The filename to save the design to.
        """
        with open("tmp.npy", "wb") as f:
            np.save(f, design)

        # Launches a docker container with the pre_process.py script
        # The script generates the mesh and FFD files

    @overload
    def simulate(self, design: np.ndarray, conditions: dict[str, int], **kwargs) -> dict[str, float]:
        print("simulate")
        return {}


if __name__ == "__main__":
    problem = Airfoil2D()
    problem.reset(seed=0)
    dataset = load_dataset(problem.str_id, split="train")

    first_design = np.array(dataset["features"][0])
    print("Design: ", first_design)
    print("Shape: ", first_design.shape)
    print(problem.design_to_simulator_input(first_design))
