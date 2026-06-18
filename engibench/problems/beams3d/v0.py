"""Structural 3D Beams Problem."""

import dataclasses
from dataclasses import dataclass
from dataclasses import field
from typing import Annotated, Any
import warnings

from datasets import Dataset
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from engibench.constraint import bounded
from engibench.constraint import constraint
from engibench.constraint import greater_than
from engibench.constraint import IMPL
from engibench.constraint import THEORY
from engibench.core import ObjectiveDirection
from engibench.core import OptiStep
from engibench.core import Problem
from engibench.core import SimulationResult
from engibench.problems.beams3d.model import fem_model
from engibench.problems.beams3d.model.fem_model import FeaModel3D
from engibench.utils.upcast import upcast

NELX = NELY = NELZ = 16
SUPPORTED_DATASET_RESOLUTIONS = (16, 32, 64)
DENSITY_THRESHOLD = 0.5  # voxels denser than this are drawn solid by render()


def _force_elements_z_from_forcedist(
    nelx: int, nely: int, nelz: int, forcedist_x: float, forcedist_y: float
) -> npt.NDArray[np.int64]:
    """Build a top-face vertical-load node mask from fractional load coordinates.

    ``forcedist_x`` / ``forcedist_y`` are fractions in ``[0, 1]`` mapped to the nearest node
    on the top face (``z = -1``). Returns a ``(nelx + 1, nely + 1, nelz + 1)`` node mask.
    """
    force_elements_z = np.zeros((nelx + 1, nely + 1, nelz + 1), dtype=np.int64)
    ix = int(np.clip(round(forcedist_x * nelx), 0, nelx))
    iy = int(np.clip(round(forcedist_y * nely), 0, nely))
    force_elements_z[ix, iy, -1] = 1
    return force_elements_z


# Default fixed elements: clamp the four bottom corners.
FIXED_ELEMENTS = np.zeros((NELX + 1, NELY + 1, NELZ + 1), dtype=np.int64)
FIXED_ELEMENTS[0, 0, 0] = FIXED_ELEMENTS[-1, 0, 0] = FIXED_ELEMENTS[0, -1, 0] = FIXED_ELEMENTS[-1, -1, 0] = 1

# Default force: a single vertical (z) load at the center of the top face (forcedist 0.5, 0.5).
FORCE_ELEMENTS_Z = _force_elements_z_from_forcedist(NELX, NELY, NELZ, 0.5, 0.5)


class Beams3D(Problem[npt.NDArray]):
    """3D structural topology optimization problem."""

    version = 0
    objectives: tuple[tuple[str, ObjectiveDirection], ...] = (("c", ObjectiveDirection.MINIMIZE),)

    @dataclass
    class Conditions:
        """Conditions."""

        volfrac: Annotated[float, bounded(lower=0.0, upper=1.0).category(THEORY)] = 0.3
        """Target volume fraction for the volume fraction constraint"""

        rmin: Annotated[float, greater_than(0.0).category(THEORY), bounded(lower=1.0).warning().category(IMPL)] = 1.5
        """Filter size used in the optimization routine"""

        forcedist_x: Annotated[float, bounded(lower=0.0, upper=1.0).category(THEORY)] = 0.5
        """Fractional x-position of the vertical load on the top face."""

        forcedist_y: Annotated[float, bounded(lower=0.0, upper=1.0).category(THEORY)] = 0.5
        """Fractional y-position of the vertical load on the top face."""

    conditions = Conditions()
    design_space = spaces.Box(low=0.0, high=1.0, shape=(NELY, NELX, NELZ), dtype=np.float32)
    dataset_id = f"IDEALLab/beams_3d_{NELX}_v0"
    container_id = None

    @dataclass
    class Config(Conditions):
        """Structured representation of configuration parameters for a numerical computation."""

        fixed_elements: Annotated[npt.NDArray[np.int64], bounded(lower=0.0, upper=1.0).category(THEORY)] = field(
            default_factory=FIXED_ELEMENTS.copy
        )
        """Binary node mask with shape (nelx + 1, nely + 1, nelz + 1), indexed [x, y, z]."""

        force_elements_z: Annotated[npt.NDArray[np.int64], bounded(lower=0.0, upper=1.0).category(THEORY)] = field(
            default_factory=FORCE_ELEMENTS_Z.copy
        )
        """Binary node mask for vertical z-loads, with shape (nelx + 1, nely + 1, nelz + 1)."""

        penal: Annotated[
            float, bounded(lower=1.0).category(THEORY), bounded(lower=0.0, upper=10.0).warning().category(IMPL)
        ] = 3.0
        """SIMP penalization parameter"""

        nelx: Annotated[int, bounded(lower=1).category(THEORY)] = NELX
        nely: Annotated[int, bounded(lower=1).category(THEORY)] = NELY
        nelz: Annotated[int, bounded(lower=1).category(THEORY)] = NELZ
        max_iter: int = fem_model.MAX_ITERATIONS

        @constraint(categories=THEORY)
        @staticmethod
        def rmin_bound(rmin: float, nelx: int, nely: int, nelz: int) -> None:
            """Constraint for rmin in (0.0, max{nelx, nely, nelz}]."""
            assert 0.0 < rmin <= max(nelx, nely, nelz), f"Params.rmin: {rmin} ∉ (0, max(nelx, nely, nelz)]"

        @constraint(categories=THEORY)
        @staticmethod
        def bc_check(
            nelx: int,
            nely: int,
            nelz: int,
            fixed_elements: npt.NDArray[np.int64],
            force_elements_z: npt.NDArray[np.int64],
        ) -> None:
            """Constraint to ensure boundary-condition masks match the configured grid."""
            assert fixed_elements.shape == (nelx + 1, nely + 1, nelz + 1), "Invalid shape for fixed_elements."
            assert force_elements_z.shape == (nelx + 1, nely + 1, nelz + 1), "Invalid shape for force_elements_z."
            assert np.any(fixed_elements), "Params.fixed_elements must contain at least one fixed node."
            assert np.any(force_elements_z), "Params.force_elements_z must contain at least one loaded node."

    def __init__(self, seed: int = 0, config: dict[str, Any] | None = None) -> None:
        """Initialize the Beams3D problem, sizing default masks and design space to the grid."""
        config = {key: (np.asarray(value) if isinstance(value, list) else value) for key, value in (config or {}).items()}
        nelx = int(config.get("nelx", NELX))
        nely = int(config.get("nely", NELY))
        nelz = int(config.get("nelz", NELZ))
        node_shape = (nelx + 1, nely + 1, nelz + 1)

        # Generate default boundary-condition masks sized to the configured grid.
        if "fixed_elements" not in config:
            fixed_elements = np.zeros(node_shape, dtype=np.int64)
            fixed_elements[0, 0, 0] = fixed_elements[-1, 0, 0] = fixed_elements[0, -1, 0] = fixed_elements[-1, -1, 0] = 1
            config["fixed_elements"] = fixed_elements
        if "force_elements_z" not in config:
            config["force_elements_z"] = _force_elements_z_from_forcedist(
                nelx,
                nely,
                nelz,
                float(config.get("forcedist_x", self.Conditions.forcedist_x)),
                float(config.get("forcedist_y", self.Conditions.forcedist_y)),
            )

        super().__init__(seed=seed)
        self.config = self.Config(**config)
        self.conditions = upcast(self.config)
        self.nelx, self.nely, self.nelz = nelx, nely, nelz
        self.max_iter = self.config.max_iter
        self.design_space = spaces.Box(low=0.0, high=1.0, shape=(nely, nelx, nelz), dtype=np.float32)
        cubic = nelx == nely == nelz and nelx in SUPPORTED_DATASET_RESOLUTIONS
        self.dataset_id = f"IDEALLab/beams_3d_{nelx}_v0" if cubic else ""

    def reset(self, seed: int | None = None) -> None:
        """Reset numpy random to a given seed."""
        super().reset(seed)

    def _boundary_conditions(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Merge per-call ``config`` overrides into the instance boundary conditions.

        ``force_elements_z`` is rebuilt from ``forcedist_x`` / ``forcedist_y`` when those are
        overridden without an explicit mask. ``force_elements_x`` / ``force_elements_y`` are not
        modeled and are ignored with a warning.
        """
        assert self.config is not None
        boundary_dict = dataclasses.asdict(self.config)
        ignored = sorted({"force_elements_x", "force_elements_y"}.intersection(config or {}))
        if ignored:
            warnings.warn(
                f"Beams3D exposes only force_elements_z; ignoring unsupported load mask config key(s): {', '.join(ignored)}.",
                UserWarning,
                stacklevel=3,
            )
        for key, value in (config or {}).items():
            if key in boundary_dict:
                boundary_dict[key] = np.asarray(value) if isinstance(value, list) else value
        if config and "force_elements_z" not in config and {"forcedist_x", "forcedist_y"}.intersection(config):
            boundary_dict["force_elements_z"] = _force_elements_z_from_forcedist(
                int(boundary_dict["nelx"]),
                int(boundary_dict["nely"]),
                int(boundary_dict["nelz"]),
                float(boundary_dict["forcedist_x"]),
                float(boundary_dict["forcedist_y"]),
            )
        return boundary_dict

    def simulate_verbose(self, design: npt.NDArray, config: dict[str, Any] | None = None) -> SimulationResult:
        """Simulate structural compliance for a design."""
        boundary_dict = self._boundary_conditions(config)
        results = FeaModel3D(plot=False, eval_only=True).run(boundary_dict, x_init=design)
        return SimulationResult(np.array([results["structural_compliance"]]))

    def optimize(
        self, starting_point: npt.NDArray, config: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, list[OptiStep]]:
        """Optimize a 3D beam topology from a starting density field."""
        boundary_dict = self._boundary_conditions(config)
        max_iter = int((config or {}).get("max_iter", self.max_iter))
        results = FeaModel3D(plot=False, eval_only=False, max_iter=max_iter).run(boundary_dict, x_init=starting_point)
        design = np.array(results["design"]).astype(np.float32)
        return design, results["opti_steps"]

    @property
    def dataset(self) -> Dataset:
        """Pull a supported cubic dataset, failing early for unsupported grids."""
        nelx, nely, nelz = self.nelx, self.nely, self.nelz
        if not (nelx == nely == nelz and nelx in SUPPORTED_DATASET_RESOLUTIONS):
            supported = ", ".join(str(resolution) for resolution in SUPPORTED_DATASET_RESOLUTIONS)
            raise ValueError(
                "Beams3D dataset access is implemented only for cubic grids "
                f"with nelx = nely = nelz in {{{supported}}}; got ({nelx}, {nely}, {nelz})."
            )
        return super().dataset

    def random_design(self, dataset_split: str = "train", design_key: str = "optimal_design") -> tuple[npt.NDArray, int]:
        """Sample a valid random design from the dataset."""
        rnd = self.np_random.integers(low=0, high=len(self.dataset[dataset_split]), dtype=int)
        design = np.array(self.dataset[dataset_split][design_key][rnd], dtype=np.float32)
        if design.shape != self.design_space.shape and design.size == np.prod(self.design_space.shape):
            design = design.reshape(self.design_space.shape)
        return design, rnd

    def render(self, design: np.ndarray, *, open_window: bool = False) -> Any:
        """Render the density field as a Matplotlib 3D voxel plot.

        Args:
            design (np.ndarray): The (nely, nelx, nelz) density field to render.
            open_window (bool): If True, display the figure in a window.

        Returns:
            The Matplotlib figure and axes.
        """
        # design is indexed [y, x, z]; transpose to [x, y, z] so the axes read naturally.
        solid = np.asarray(design).transpose(1, 0, 2) > DENSITY_THRESHOLD

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.voxels(solid, edgecolor="gray")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        if open_window:
            plt.show()
        return fig, ax


def main(problem_type: type[Problem[npt.NDArray]], *, open_window: bool = False) -> None:
    """Instantiate Beams3D, sample from the dataset, simulate, optimize, and render.

    Supported data-backed cubic resolutions are 16, 32, and 64. If a new
    resolution is not passed through the problem type, the default 16 x 16 x 16
    conditions are used.
    """
    problem = problem_type(seed=0)
    if problem.config is None:
        raise RuntimeError("Beams3D config has not been initialized.")

    print(f"Loading dataset for nelx={problem.config.nelx}, nely={problem.config.nely}, nelz={problem.config.nelz}.")
    train_split = problem.dataset["train"]

    # Get a design and its conditions from the dataset, then render the design.
    # Note that here we override any previous configs to re-optimize the same design as a test case.
    design, idx = problem.random_design()
    row = train_split[int(idx)]
    reference_compliance = float(train_split["c"][int(idx)])

    # Reuse the dataset row's conditions to re-simulate and re-optimize the sampled design.
    config: dict[str, Any] = {key: float(row[key]) for key in ("volfrac", "rmin", "forcedist_x", "forcedist_y")}

    problem.render(design, open_window=open_window)
    print(f"Verifying structural compliance via simulation. Reference value: {reference_compliance:.4f}")

    try:
        objective_values = problem.simulate(design, config=config)
        print(f"Calculated structural compliance: {objective_values[0]:.4f}")
    except ArithmeticError:
        print("Failed to calculate structural compliance for sampled design.")

    print("\nNow conducting a sample optimization with the given configs:", config)
    problem.reset(seed=1)

    design_shape = problem.design_space.shape
    if design_shape is None:
        raise RuntimeError("Beams3D design space shape has not been initialized.")
    starting_point = np.full(tuple(design_shape), float(config["volfrac"]), dtype=np.float32)
    optimal_design, optisteps_history = problem.optimize(config=config, starting_point=starting_point)
    if optisteps_history:
        print(f"Final structural compliance: {optisteps_history[-1].obj_values[0]:.4f}")
    print(f"Final design volume fraction: {optimal_design.sum() / np.prod(optimal_design.shape):.4f}")

    problem.render(optimal_design, open_window=open_window)


if __name__ == "__main__":
    main(Beams3D, open_window=True)
