# ruff: noqa: E741, N806, N815, N816
# Disabled variable name conventions

"""Beams 2D problem.

Filename convention is that folder paths do not end with /. For example, /path/to/folder is correct, but /path/to/folder/ is not.
"""

from __future__ import annotations

from copy import deepcopy
import dataclasses
from typing import Any

from gymnasium import spaces
import numpy as np
import numpy.typing as npt

from engibench.core import DesignType
from engibench.core import OptiStep
from engibench.core import Problem
from engibench.problems.beams2d.backend import calc_sensitivity
from engibench.problems.beams2d.backend import overhang_filter
from engibench.problems.beams2d.backend import Params
from engibench.problems.beams2d.backend import setup


@dataclasses.dataclass
class ExtendedOptiStep(OptiStep):
    """Extended OptiStep to store a single NumPy array representing a density field at a given optimization step."""

    stored_design: npt.NDArray[np.float64] = dataclasses.field(default_factory=lambda: np.array([], dtype=np.float64))


class Beams2D(Problem[npt.NDArray, npt.NDArray]):
    r"""Beam 2D topology optimization problem.

    ## Problem Description
    This problem simulates bending in an MBB beam, where the beam is symmetric about the central vertical axis and a force is applied at the top-center of the design. Problems are formulated using Density-based Topology Optimization (TO) based on an existing Python [implementation](https://github.com/arjendeetman/TopOpt-MMA-Python).

    ## Design space
    The design space is an array of solid densities in `[0.,1.]` with shape `(5000,)` that can also be represented as a `(100, 50)` image, where `nelx = 100` and `nely = 50`.

    ## Objectives
    The objectives are defined and indexed as follows:
    0. `c`: Compliance to minimize.

    ## Boundary conditions
    The boundary conditions are defined by the following parameters:
    - `nelx`: Width of the domain.
    - `nely`: Height of the domain.
    - `volfrac`: Desired volume fraction (in terms of solid material) for the design.
    - `penal`: Intermediate density penalty term.
    - `rmin`: Minimum feature length of beam members.
    - `ft`: Filtering method; 0 for sensitivity-based and 1 for density-based.
    - `overhang_constraint`: Boolean input condition to decide whether a 45 degree overhang constraint is imposed on the design.

    ## Dataset
    The dataset linked to this problem is hosted on the [Hugging Face Datasets Hub](https://huggingface.co/datasets/IDEALLab/beams_2d).

    ## Simulator
    The objective (compliance) is calculated by the equation `c = ( (Emin+xPrint**penal*(Emax-Emin))*ce ).sum()` where `xPrint` is the current true density field.

    ## References
    If you use this problem in your research, please cite the following paper:
    E. Andreassen, A. Clausen, M. Schevenels, B. S. Lazarov, and O. Sigmund, “Efficient topology optimization in MATLAB using 88 lines of code,” in Structural and Multidisciplinary Optimization, vol. 43, pp. 1-16, 2011.

    ## Lead
    Arthur Drake @arthurdrake1
    """

    version = 0
    possible_objectives: tuple[tuple[str, str]] = (("c", "minimize"),)
    boundary_conditions: frozenset[tuple[str, Any]] = frozenset(
        [
            ("nelx", 100),
            ("nely", 50),
            ("volfrac", 0.35),
            ("penal", 3.0),
            ("rmin", 2.0),
            ("ft", 1),
            ("overhang_constraint", False),
        ]
    )
    design_space = spaces.Box(low=0.0, high=1.0, shape=(5000,), dtype=np.float32)
    dataset_id = "IDEALLab/beams_2d_v0"
    _dataset = None
    container_id = None  # type: ignore

    def __init__(self) -> None:
        """Initializes the Beams2D problem."""
        super().__init__()

        self.seed = None

    def __design_to_simulator_input(self, design: npt.NDArray) -> npt.NDArray:
        r"""Convert a design to a simulator input: flattens the 2D image to a 1D vector.

        Args:
            design (DesignType): The design to convert.

        Returns:
            SimulatorInputType: The corresponding design as a simulator input.
        """
        return np.swapaxes(design, 0, 1).ravel()

    def __simulator_output_to_design(self, simulator_output: npt.NDArray, nelx: int = 100, nely: int = 50) -> npt.NDArray:
        r"""Convert a simulator input to a design.

        Args:
            simulator_output (SimulatorInputType): The input to convert.
            nelx: Width of the problem domain.
            nely: Height of the problem domain.

        Returns:
            DesignType: The corresponding design.
        """
        return np.swapaxes(simulator_output.reshape(nelx, nely), 0, 1)

    def __data_to_images(self, data: list[np.ndarray]) -> np.ndarray:
        """Converts the flattened data back to images. NOTE: Assumes the image width is twice the height.

        Args:
            data (list of np.ndarray): The designs to convert.

        Returns:
            np.ndarray: The newly converted designs as images.
        """
        ims = np.array(data)
        sh = ims.shape
        nely = int(np.sqrt(sh[-1] // 2))
        nelx = int(nely * 2)
        ims = ims.reshape(sh[0], nely, nelx)
        return ims

    def simulate(self, design: npt.NDArray, p: Params, ce: npt.NDArray | None) -> npt.NDArray:
        """Simulates the performance of a beam design. Assumes the Params object is already set up.

        Args:
            design (np.ndarray): The design to simulate.
            p: Params object with configs (e.g., boundary conditions) and needed vectors/matrices for the simulation.
            ce: (np.ndarray, optional): If applicable, the pre-calculated sensitivity of the current design.

        Returns:
            npt.NDArray: The performance of the design in terms of compliance.
        """
        if ce is None:
            ce = calc_sensitivity(design, p)
        c = ((p.Emin + design**p.penal * (p.Emax - p.Emin)) * ce).sum()  # compliance (objective)
        return np.array(c)

    def optimize(self, p: Params) -> tuple[np.ndarray, list[OptiStep]]:
        """Optimizes the design of a beam.

        Args:
            p: Params object with configs (e.g., boundary conditions) and needed vectors/matrices for the optimization.

        Returns:
            Tuple[np.ndarray, dict]: The optimized design and its performance.
        """
        # Prepares the optimization script/function with the optimization configuration
        p = setup(p)

        # Make sure to include the intermediate designs of size (5000,)
        # Make sure to return the full history of the optimization instead of just the last step
        optisteps_history = []

        dv = np.zeros(p.nely * p.nelx)
        dc = np.zeros(p.nely * p.nelx)
        ce = np.ones(p.nely * p.nelx)

        x = p.volfrac * np.ones(p.nely * p.nelx, dtype=float)
        xPhys = x = p.volfrac * np.ones(p.nely * p.nelx, dtype=float)
        xPrint, _, _ = overhang_filter(xPhys, p, dc, dv)

        loop = 0
        change = 1

        while change > p.min_change and loop < p.max_iter:  # while change>0.01 and loop<max_iter:
            loop = loop + 1

            ce = calc_sensitivity(xPrint, p=p)
            c = self.simulate(xPrint, p=p, ce=ce)

            dc = (-p.penal * xPrint ** (p.penal - 1) * (p.Emax - p.Emin)) * ce
            dv = np.ones(p.nely * p.nelx)
            xPrint, dc, dv = overhang_filter(xPhys, p, dc, dv)  # MATLAB implementation

            if p.ft == 0:
                dc = np.asarray((p.H * (x * dc))[np.newaxis].T / p.Hs)[:, 0] / np.maximum(0.001, x)  # type: ignore
            elif p.ft == 1:
                dc = np.asarray(p.H * (dc[np.newaxis].T / p.Hs))[:, 0]
                dv = np.asarray(p.H * (dv[np.newaxis].T / p.Hs))[:, 0]

            # Optimality criteria
            l1 = 0
            l2 = 1e9
            move = 0.2
            # reshape to perform vector operations
            xnew = np.zeros(p.nelx * p.nely)

            while (l2 - l1) / (l1 + l2) > p.min_ratio:
                lmid = 0.5 * (l2 + l1)
                if lmid > 0:
                    xnew = np.maximum(
                        0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid))))
                    )  # type: ignore
                else:
                    xnew = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, x + move)))

                # Filter design variables
                if p.ft == 0:
                    xPhys = xnew
                elif p.ft == 1:
                    xPhys = np.asarray(p.H * xnew[np.newaxis].T / p.Hs)[:, 0]

                xPrint, _, _ = overhang_filter(xPhys, p, dc, dv)

                if xPrint.sum() > p.volfrac * p.nelx * p.nely:
                    l1 = lmid
                else:
                    l2 = lmid
                if l1 + l2 == 0:
                    break

            # Compute the change by the inf. norm
            change = np.linalg.norm(xnew.reshape(p.nelx * p.nely, 1) - x.reshape(p.nelx * p.nely, 1), np.inf)
            x = deepcopy(xnew)

            # Record the current state in optisteps_history
            current_step = ExtendedOptiStep(obj_values=np.array([c]), step=loop)
            current_step.stored_design = np.array(xPrint)
            optisteps_history.append(current_step)

        return (xPrint, optisteps_history)

    def reset(self, seed: int | None = None, **kwargs) -> None:
        r"""Reset the simulator and numpy random to a given seed.

        Args:
            seed (int, optional): The seed to reset to. If None, a random seed is used.
            **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError

    def render(self, design: np.ndarray, open_window: bool = False) -> Any:
        """Renders the design in a human-readable format.

        Args:
            design (np.ndarray): The design to render.
            open_window (bool): If True, opens a window with the rendered design.

        Returns:
            Any: The rendered design.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(design, cmap="coolwarm", ax=ax, vmin=0, vmax=1)

        if open_window:
            plt.show()
        return fig, ax

    def random_design(self, designs: np.ndarray) -> DesignType:
        """Samples a valid random design.

        Args:
            designs (np.ndarray): The set of possible designs to choose from.

        Returns:
            DesignType: The valid random design.
        """
        rnd = self.np_random.integers(low=0, high=designs.shape[0])  # type: ignore
        return designs[rnd]


if __name__ == "__main__":
    print("Loading dataset.")
    init_params = Params()
    problem = Beams2D()
    problem.reset()
    dataset = problem.dataset

    # Example of getting the training set and reshaping into images
    xPrint_train = problem._Beams2D__data_to_images(dataset["train"]["xPrint"])  # type: ignore
    print("Shape of xPrint_train:", xPrint_train.shape)

    # Get design and conditions from the dataset
    design = problem.random_design(xPrint_train)
    fig, ax = problem.render(design, open_window=True)
    fig.savefig(
        "beam_random.png",
        dpi=300,
    )

    # Sample Optimization
    print("Now conducting a sample optimization with the provided configs.")

    # Ask the user if they want to override boundary conditions
    override = input("Do you want to override boundary conditions? (y/n): ").strip().lower()

    if override == "y":
        print("Provide new values (press Enter to skip and keep defaults).")

        updates = {}
        for key, default in dict(Beams2D.boundary_conditions).items():
            new_value = input(f"{key} (default={default}): ").strip()
            if new_value:
                try:
                    # Convert value to the appropriate type
                    if isinstance(default, bool):
                        updates[key] = new_value.lower() in ["true", "1", "yes", "y"]
                    elif isinstance(default, int):
                        updates[key] = int(new_value)
                    elif isinstance(default, float):
                        updates[key] = float(new_value)
                    else:
                        updates[key] = new_value
                except ValueError:
                    print(f"Invalid value for {key}, keeping default.")

        # Apply updates to parameters
        init_params.update(updates)

    # NOTE: xPrint and optisteps_history[-1].stored_design are interchangeable.
    xPrint, optisteps_history = problem.optimize(init_params)
    print(f"Final compliance: {optisteps_history[-1].obj_values[0]:.4f}")
    print(
        f"Final design volume fraction: {xPrint.sum() / (init_params.nelx * init_params.nely):.4f}"  # type: ignore
    )

    fig, ax = problem.render(problem._Beams2D__simulator_output_to_design(xPrint), open_window=True)  # type: ignore
    fig.savefig(
        "beam_optim.png",
        dpi=300,
    )
