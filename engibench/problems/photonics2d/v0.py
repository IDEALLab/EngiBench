"""Photonics2D problem.

This is essentially re-factored from the code at
https://nbviewer.org/github/fancompute/workshop-invdesign/blob/master/04_Invdes_wdm_scheduling.ipynb

Author: Mark Fuge @markfuge
"""

from __future__ import annotations

from dataclasses import dataclass

# Need os import for makedirs for saving plots
import os
import pprint
from typing import Annotated, Any, ClassVar, TYPE_CHECKING

# Importing autograd since the ceviche library uses it for automatic differentiation of the FDFD solver
import autograd.numpy as npa

# Import ArrayBox type for checking
import ceviche
from ceviche import fdfd_ez
from ceviche import jacobian
from ceviche.optimizers import adam_optimize
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

# --- EngiBench Core Imports ---
# Import necessary base classes and types directly from the library's core
from engibench.constraint import bounded
from engibench.constraint import IMPL
from engibench.constraint import THEORY
from engibench.core import ObjectiveDirection
from engibench.core import OptiStep
from engibench.core import Problem
from engibench.problems.photonics2d.backend import epsr_parameterization

# --- EngiBench Problem-Specific Backend ---
from engibench.problems.photonics2d.backend import init_domain
from engibench.problems.photonics2d.backend import insert_mode
from engibench.problems.photonics2d.backend import mode_overlap
from engibench.problems.photonics2d.backend import operator_blur
from engibench.problems.photonics2d.backend import operator_proj
from engibench.problems.photonics2d.backend import poly_ramp
from engibench.problems.photonics2d.backend import wavelength_to_frequency

if TYPE_CHECKING:
    from autograd.numpy.numpy_boxes import ArrayBox


class Photonics2D(Problem[npt.NDArray]):
    r"""Photonic Inverse Design 2D Problem (Wavelength Demultiplexer).

    ## Problem Description
    Optimize a 2D material distribution (`rho`) to function as a wavelength
    demultiplexer, routing wave with `lambda1` to output 1 and `lambda2` to output 2. The
    design variables represent material density which is converted to permittivity
    using filtering and projection.

    ## Design space
    2D tensor `rho` (num_elems_x, num_elems_y) with values in [0, 1], representing material density.
    Stored as `design_space` (gymnasium.spaces.Box).

    ## Objectives
    0. `neg_field_overlap`: Combined objective to minimize, defined as
       `penalty - overlap1 * overlap2`. Lower is better. This is the value
       returned by `simulate` and corresponds to the overlap in the target
       electrical fields with the desired demultiplexing locations.
       Note that `optimize` internally works with a normalized version for
       stability (`penalty - normalized_overlap`) and reports history
       (`OptiStep`) corresponding to that normalized version.
       Note: This means comparing the values returned by `simulate` and `optimize` directly
       may not yield similar numbers.

    ## Conditions
    These are designed as user-configurable parameters that alter the problem definition.
    Default problem parameters that can be overridden via the `config` dict:
    - `lambda1`: The first input wavelength in μm (default: 1.5 μm).
    - `lambda2`: The first input wavelength in μm (default: 1.3 μm).
    - `blur_radius`: Radius for the density blurring filter (default: 2).
                     Higher values correspond to larger elements, which could
                     possibly be more manufacturable.
    - `num_elems_x`: Number of grid cells in x (default: 120).
    - `num_elems_y`: Number of grid cells in y (default: 120).

    In practice, for the dataset loading, we will keep `num_elems_x` and `num_elems_y`to set
    values for each dataset, such that different resolutions correspond to different
    independent datasets.

    ## Optimization Parameters
    Note: These are advanced parameters that alter the optimization process --
    we do not recommend changing these if you are only using the library for benchmarking,
    as it could make results less reproducible across papers using this problem.)
    - `num_optimization_steps`: Total number of optimization steps (default: 300).
    - `step_size`: Adam optimizer step size (default: 1e-1).
    - `penalty_weight`: Weight for the L2 penalty term (default: 1e-2). Larger values reduce
                        unnecessary material, but may lead to worse performance if too large.
    - `eta`: Projection center parameter (default: 0.5). There is little reason to change this.
    - `N_proj`: Number of projection applications (default: 1). Increasing this can help make
                the design more binary.
    - `N_blur`: Number of blur applications (default: 1). Increasing this smooths the design more.
    - `save_frame_interval`: Interval for saving intermediate design frames during optimization.
                             If > 0, saves a frame every `save_frame_interval` iterations
                             to the `opt_frames/` directory. Default is 0 (disabled).

    ## Internal Constants
    Note: These are not typically changed by users, but provided here for technical reference
    - `dl`: Spatial resolution (meters) (default: 40e-9).
    - `Npml`: Number of PML cells (default: 20).
    - `epsr_min`: Minimum relative permittivity (default: 1.0).
    - `epsr_max`: Maximum relative permittivity (default: 12.0).
    - `space_slice`: Extra space for source/probe slices (pixels) (default: 8).

    ## Simulator
    The simulation uses the `ceviche` library's Finite Difference Frequency Domain (FDFD)
    solver (`fdfd_ez`). Optimization uses `ceviche.optimizers.adam_optimize` with
    gradients computed via automatic differentiation (`autograd`).

    ## Dataset
    This problem currently provides one dataset corresponding to resolution of 120x120, are available on the [Hugging Face Datasets Hub](https://huggingface.co/datasets/IDEALLab/photonics_2d_120_120_v0).

    ### v0

    #### Fields
    Each dataset contains:
    - `lambda1`: The first input wavelength in μm.
    - `lambda2`: The second input wavelength in μm.
    - `blur_radius`: Radius for the density blurring filter (pixels).
    - `optimal_design`: The optimal design density array (shape num_elems_x, num_elems_y).
    - `optimization_history`: A list of objective values from the optimization process (negative field overlap, where lower is better) -- This is for advanced use.

    #### Creation Method
    To generate a dataset for training, we generate (randomly, uniformly) swept over the following parameters:
    - $\lambda_1 \in [0.5\mu m, 1.25\mu m]$ = `lambda1` = `rng.uniform(low=0.5, high=1.25, size=20)` -- This corresponds roughly to a portion of the visible spectrum up to near-infrared.
    - $\lambda_2 \in [0.75\mu m, 1.5\mu m]$ = `lambda2` = `rng.uniform(low=0.75, high=1.5, size=20)` -- This corresponds roughly to a portion of the visible spectrum up to near-infrared.
    - $r_{blur}$ = `blur_radius` = `range(0, 5)`

    ## Citation
    This problem is directly refactored from the Ceviche Library:
    https://github.com/fancompute/ceviche
    and if you use this problem your experiments, you can use the citation below
    provided by the original library authors:
    ```
    @article{hughes2019forward,
        title={Forward-Mode Differentiation of Maxwell's Equations},
        author={Hughes, Tyler W and Williamson, Ian AD and Minkov, Momchil and Fan, Shanhui},
        journal={ACS Photonics},
        volume={6},
        number={11},
        pages={3010--3016},
        year={2019},
        publisher={ACS Publications}
    }
    ```

    ## Lead
    Mark Fuge @markfuge
    """

    version = 0
    # --- Objective Definition (Raw, non-normalized version for simulate) ---
    objectives: tuple[tuple[str, ObjectiveDirection]] = (("neg_field_overlap", ObjectiveDirection.MINIMIZE),)
    # Note: optimize internally uses a normalized objective, see OptiStep values.
    # We keep a single objective name for simplicity in the list.

    # Constants specific to problem design
    _pml_space = 10  # Space between PML and design region (pixels)
    _wg_width = 12  # Width of waveguides (pixels)
    _wg_shift = 9  # Lateral shift for output waveguides (pixels)
    _dl = 40e-9  # Spatial resolution (meters)
    _num_elems_pml = 20  # Number of PML cells (pixels)
    _epsr_min = 1.0  # Minimum relative permittivity (background)
    _epsr_max = 12.0  # Maximum relative permittivity (material)
    _space_slice = 8  # Extra space for source/probe slices (pixels)
    _num_elems_x_default = 120  # Default number of grid cells in x
    _num_elems_y_default = 120  # Default number of grid cells in y

    # Defaults for the optimization parameters
    _num_optimization_steps_default = 200  # Default number of optimization steps
    _step_size_default = 1e-1  # Default step size for Adam optimizer
    _eta_default = 0.5
    _num_projections_default = 1
    _penalty_weight_default = 1e-2  # Default weight for mass penalty term
    _num_blurs_default = 1

    conditions: tuple[tuple[str, Any], ...] = (
        ("lambda1", 1.5),  # First input wavelength in μm
        ("lambda2", 1.3),  # Second input wavelength in μm
        ("blur_radius", 2),  # Radius for the density blurring filter (pixels)
    )

    design_space = spaces.Box(low=0.0, high=1.0, shape=(_num_elems_x_default, _num_elems_y_default), dtype=np.float64)

    dataset_id = f"IDEALLab/photonics_2d_{_num_elems_x_default}_{_num_elems_y_default}_v0"
    container_id = None

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        num_elems_x: int = _num_elems_x_default,
        num_elems_y: int = _num_elems_y_default,
        **kwargs,
    ) -> None:
        """Initializes the Photonics2D problem.

        Args:
            config (dict): A dictionary with configuration (e.g., boundary conditions) for the simulation.
            num_elems_x (int): Number of grid cells in x (default: 120).
            num_elems_y (int): Number of grid cells in y (default: 120).
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)

        # Replace the conditions with any new configs passed in
        config = config or {}
        self.conditions = tuple((key, config.get(key, value)) for key, value in self.conditions)
        current_conditions = self.conditions_dict
        print("Initializing Photonics Problem with configuration:")
        pprint.pp(current_conditions)
        self.num_elems_x = num_elems_x
        self.num_elems_y = num_elems_y
        self.design_space = spaces.Box(low=0.0, high=1.0, shape=(num_elems_x, num_elems_y), dtype=np.float32)
        self.dataset_id = f"IDEALLab/photonics_2d_{num_elems_x}_{num_elems_y}_v{self.version}"

        # Setup basic simulation parameters
        self.omega1 = wavelength_to_frequency(current_conditions["lambda1"])
        self.omega2 = wavelength_to_frequency(current_conditions["lambda2"])
        self._current_beta: float = 1.0  # Placeholder for beta scheduling

        # Config depends on num_elems_x, num_elems_y -> define it in __init__
        @dataclass
        class Config:
            """Structured representation of configuration parameters for a numerical computation."""

            num_elems_x: ClassVar[
                Annotated[
                    int,
                    bounded(lower=0).category(THEORY),
                    bounded(lower=60).category(IMPL),
                    bounded(lower=90, upper=200).warning().category(IMPL),
                ]
            ] = self.num_elems_x
            num_elems_y: ClassVar[
                Annotated[
                    int,
                    bounded(lower=0).category(THEORY),
                    bounded(lower=105).category(IMPL),
                    bounded(lower=110, upper=300).warning().category(IMPL),
                ]
            ] = self.num_elems_y

            lambda1: Annotated[
                float,
                bounded(lower=0.0).category(THEORY),
                bounded(lower=0.5).category(IMPL),
                bounded(lower=0.5, upper=1.5).warning().category(IMPL),
            ] = 1.5
            lambda2: Annotated[
                float,
                bounded(lower=0.0).category(THEORY),
                bounded(lower=0.5).category(IMPL),
                bounded(lower=0.5, upper=1.5).warning().category(IMPL),
            ] = 1.3
            blur_radius: Annotated[
                int, bounded(lower=0).category(THEORY | IMPL), bounded(lower=0, upper=5).warning().category(IMPL)
            ] = 2

        self.Config = Config

    def _setup_simulation(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Helper function to setup simulation parameters and domain."""
        # Merge config with default conditions
        current_conditions = self.conditions_dict
        current_conditions.update(config or {})

        # Initialize domain geometry
        self._bg_rho, self._design_region, self._input_slice, self._output_slice1, self._output_slice2 = init_domain(
            num_elems_x=self.num_elems_x,
            num_elems_y=self.num_elems_y,
            num_elems_pml=self._num_elems_pml,
            space=self._pml_space,
            wg_width=self._wg_width,
            wg_shift=self._wg_shift,
            space_slice=self._space_slice,
        )

        return current_conditions

    def _run_fdfd(
        self, design: npt.NDArray, conditions: dict[str, Any]
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """Helper to run FDFD and return key components (epsr, fields, sources, probes)."""
        omega1 = self.omega1
        omega2 = self.omega2
        beta = self._current_beta
        num_blurs = conditions.get("num_blurs", self._num_blurs_default)
        num_projections = conditions.get("num_projections", self._num_projections_default)
        eta = conditions.get("eta", self._eta_default)

        # 1. Parameterize
        epsr = epsr_parameterization(
            rho=design,
            bg_rho=self._bg_rho,
            design_region=self._design_region,
            radius=conditions["blur_radius"],
            num_blurs=num_blurs,
            beta=beta,
            eta=eta,
            num_projections=num_projections,
            epsr_min=self._epsr_min,
            epsr_max=self._epsr_max,
        )

        # 2. Setup Sources and Probes (depend on epsr)
        source1 = insert_mode(omega1, self._dl, self._input_slice.x, self._input_slice.y, epsr, m=1)
        source2 = insert_mode(omega2, self._dl, self._input_slice.x, self._input_slice.y, epsr, m=1)
        probe1 = insert_mode(omega1, self._dl, self._output_slice1.x, self._output_slice1.y, epsr, m=1)
        probe2 = insert_mode(omega2, self._dl, self._output_slice2.x, self._output_slice2.y, epsr, m=1)

        # 3. Setup FDFD Simulations
        # We need to run two simulations, one for each wavelength, to see which paths the light takes in both
        self._simulation1 = fdfd_ez(omega1, self._dl, epsr, [self._num_elems_pml, self._num_elems_pml])
        self._simulation2 = fdfd_ez(omega2, self._dl, epsr, [self._num_elems_pml, self._num_elems_pml])

        # 4. Solve FDFD
        # Always solve as Ez fields are needed for return value or objective calc
        _, _, ez1 = self._simulation1.solve(source1)
        _, _, ez2 = self._simulation2.solve(source2)

        return epsr, ez1, ez2, source1, source2, probe1, probe2

    def simulate(self, design: npt.NDArray, config: dict[str, Any] | None = None, **kwargs) -> npt.NDArray:  # noqa: ARG002
        """Simulates the performance of a design, returning the raw objective value.

           Stores simulation fields (`Ez1`, `Ez2`, `epsr`) internally in `_last_Ez1`,
           `_last_Ez2`, `_last_epsr` for later access (e.g., by render).

        Args:
            design (npt.NDArray): The design density array `rho` (shape num_elems_x, num_elems_y).
            config (dict): Dictionary to override default conditions.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            npt.NDArray: 1-element array: [raw_objective], where
                         raw_objective = penalty - overlap1 * overlap2. Lower is better.
                         Note: This is non-normalized, unlike the value optimized internally.
        """
        conditions = self._setup_simulation(config)

        # --- Run Simulation ---
        # We don't need source returns here
        print("Simulating design under the following conditions:")
        pprint.pp(conditions)
        epsr, ez1, ez2, _, _, probe1, probe2 = self._run_fdfd(design, conditions)

        # --- Store Results Internally ---
        self._last_epsr = epsr.copy()
        self._last_Ez1 = ez1.copy()
        self._last_Ez2 = ez2.copy()

        # --- Calculate Raw Objective (No Normalization) ---
        # Use standard numpy here, no gradients needed
        overlap1 = np.abs(np.sum(np.conj(ez1) * probe1)) * 1e6
        overlap2 = np.abs(np.sum(np.conj(ez2) * probe2)) * 1e6
        penalty_weight = conditions.get("penalty_weight", self._penalty_weight_default)  # Default to max epsr
        penalty = penalty_weight * np.linalg.norm(design)

        raw_objective = penalty - (overlap1 * overlap2)  # Minimize this

        return np.array([raw_objective], dtype=np.float64)

    def optimize(  # noqa: PLR0915
        self,
        starting_point: npt.NDArray,
        config: dict[str, Any] | None = None,
        **kwargs,  # noqa: ARG002
    ) -> tuple[npt.NDArray, list[OptiStep]]:
        """Optimizes a topology (rho) starting from `starting_point` using Adam.

           Internally maximizes a normalized objective (`normalized_overlap - penalty`)
           for stability and convergence, using gradients from autograd via Ceviche.
           Optionally saves intermediate design frames based on `save_frame_interval`.

        Args:
            starting_point (npt.NDArray): The starting design `rho` (shape num_elems_x, num_elems_y).
            config (dict): Dictionary to override default conditions (e.g., num_optimization_steps,
                           step_size, penalty_weight, save_frame_interval).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            tuple[npt.NDArray, list[OptiStep]]:
                - The optimized design `rho` (float32, shape num_elems_x, num_elems_y).
                - A list of OptiStep history. `OptiStep.obj_values` contains the
                  value of the internally optimized objective, negated to match a
                  MINIMIZE direction (i.e., `penalty - normalized_overlap`).
                  The `step` attribute corresponds to the optimizer iteration.
        """
        conditions = self._setup_simulation(config)
        print("Attempting to run Optimization for Photonics2D under the following conditions:")
        pprint.pp(conditions)

        # Pull out problem-specific parameters from conditions
        num_elems_x = self.num_elems_x
        num_elems_y = self.num_elems_y
        # Pull out optimization parameters from conditions
        # Parameters specific to optimization
        num_optimization_steps = conditions.get("num_optimization_steps", self._num_optimization_steps_default)
        step_size = conditions.get("step_size", self._step_size_default)
        penalty_weight = conditions.get("penalty_weight", self._penalty_weight_default)
        self._eta = conditions.get("eta", self._eta_default)
        self._num_projections = conditions.get("num_projections", self._num_projections_default)
        self._num_blurs = conditions.get("num_blurs", self._num_blurs_default)
        max_beta = 300
        # --- Get the frame saving interval from conditions for plotting ---
        save_frame_interval = conditions.get("save_frame_interval", 0)

        # --- Initial Simulation for Normalization Constants E01, E02 ---
        print("Optimize: Calculating E01/E02 using initial design...")  # Keep this info message
        epsr_init, ez1_init, ez2_init, source1_init, source2_init, probe1_init, probe2_init = self._run_fdfd(
            starting_point, conditions
        )

        self._E01 = npa.abs(npa.sum(npa.conj(ez1_init) * probe1_init)) * 1e6
        self._E02 = npa.abs(npa.sum(npa.conj(ez2_init) * probe2_init)) * 1e6

        if self._E01 == 0 or self._E02 == 0:
            print(
                f"Warning: Initial overlap zero (E01={self._E01:.3e}, E02={self._E02:.3e}). Using fallback."
            )  # Keep this warning
            self._E01 = self._E01 if self._E01 != 0 else 1e-9
            self._E02 = self._E02 if self._E02 != 0 else 1e-9

        self._source1 = source1_init
        self._source2 = source2_init
        self._probe1 = probe1_init
        self._probe2 = probe2_init

        # Ensure directory exists for saving frames
        frame_dir = "opt_frames"
        os.makedirs(frame_dir, exist_ok=True)

        # --- Define Objective Function for Ceviche Optimizer ---
        def objective_for_optimizer(rho_flat: npt.NDArray | ArrayBox) -> float | ArrayBox:
            """Calculates (normalized_overlap - penalty) for maximization.

            Note: All functions or inputs here should be compatible with autograd (npa).
            """
            rho = rho_flat.reshape((num_elems_x, num_elems_y))
            conditions["beta"] = self._current_beta  # Use scheduled beta

            # --- Parameterization and Simulation ---
            epsr = epsr_parameterization(
                rho=rho,
                bg_rho=self._bg_rho,
                design_region=self._design_region,
                radius=conditions["blur_radius"],
                num_blurs=self._num_blurs,
                beta=self._current_beta,
                eta=self._eta,
                num_projections=self._num_projections,
                epsr_min=self._epsr_min,
                epsr_max=self._epsr_max,
            )
            self._simulation1.eps_r = epsr
            self._simulation2.eps_r = epsr
            _, _, ez1 = self._simulation1.solve(self._source1)
            _, _, ez2 = self._simulation2.solve(self._source2)

            # Calculate overlaps
            overlap1 = mode_overlap(ez1, self._probe1)
            overlap2 = mode_overlap(ez2, self._probe2)
            current_e01 = self._E01  # Assume already handled zero case
            current_e02 = self._E02
            normalized_overlap = (overlap1 / current_e01) * (overlap2 / current_e02)

            penalty = penalty_weight * npa.linalg.norm(rho)
            return normalized_overlap - penalty  # Value to MAXIMIZE

        # --- Define Gradient ---
        objective_jac = jacobian(objective_for_optimizer, mode="reverse")

        # --- Define Callback ---
        opti_steps_history: list[OptiStep] = []

        def callback(iteration: int, objective_history_list: list, rho_flat: npt.NDArray | ArrayBox) -> None:
            """Callback for adam_optimize. Receives the history of objective values."""
            # Handle Empty History
            if not objective_history_list:
                return

            # Get the latest objective value
            last_scalar_obj_value = objective_history_list[-1]

            # Beta Scheduling Logic -- Quadratic ramp from 0 to max_beta
            iteration = len(objective_history_list)
            self._current_beta = poly_ramp(iteration, max_iter=num_optimization_steps, b0=0, bmax=max_beta, degree=2)

            # Store OptiStep info
            neg_norm_objective_value = -last_scalar_obj_value
            step_info = OptiStep(obj_values=np.array([neg_norm_objective_value], dtype=np.float64), step=iteration)
            opti_steps_history.append(step_info)

            # --- Configurable Intermediate Frame Saving ---
            # Check if saving is enabled and if current iteration is a multiple of the interval
            # Also check iteration > 0 to avoid saving the initial state redundantly
            if (
                save_frame_interval is not None
                and save_frame_interval > 0
                and iteration > 0
                and iteration % save_frame_interval == 0
            ):
                # Reshape the current design parameters
                current_rho = rho_flat.reshape((num_elems_x, num_elems_y))
                current_rho = operator_proj(current_rho, self._eta, beta=self._current_beta, num_projections=1)

                # --- Call self.render to generate the plot ---
                # Pass the current conditions dictionary in case render needs it
                # Note: This will re-run the simulation for the current_rho
                fig = self.render(current_rho, open_window=False, config=conditions)
                # ---------------------------------------------

                save_path = os.path.join(frame_dir, f"frame_iter_{iteration:04d}.png")

                # Save the figure returned by render
                fig.savefig(save_path, dpi=200)
                plt.close(fig)  # Close the figure to free memory
                print(f"Callback Iter {iteration}: Saved frame to {save_path}")
            # --- End Frame Saving ---
            if iteration == num_optimization_steps - 1:
                print(f"Final Iteration {iteration}: Objective Value: {neg_norm_objective_value:.3e}")
                print("Saving render of final design...")
                current_rho = rho_flat.reshape((num_elems_x, num_elems_y))
                current_rho = operator_proj(current_rho, self._eta, beta=self._current_beta, num_projections=1)
                current_rho = np.rint(current_rho).astype(np.float32)  # Convert to binary for final save
                fig = self.render(current_rho, open_window=False, config=conditions)
                save_path = os.path.join(frame_dir, "frame_final.png")
                fig.savefig(save_path, dpi=200)
                plt.close(fig)

        # --- Run Optimization ---
        print(
            f"\nStarting optimization with num_optimization_steps={num_optimization_steps}, step_size={step_size}"
        )  # Keep start message
        (rho_optimum_flat, _) = adam_optimize(
            objective_for_optimizer,
            starting_point.flatten(),
            objective_jac,
            Nsteps=num_optimization_steps,
            direction="max",
            step_size=step_size,
            callback=callback,
        )

        # --- Final Result ---
        rho_optimum = rho_optimum_flat.reshape((num_elems_x, num_elems_y))
        # Project the optimized design to the valid range [0, 1]
        rho_optimum = operator_proj(rho_optimum, self._eta, beta=self._current_beta, num_projections=1)
        rho_optimum = np.rint(rho_optimum)  # Convert to binary for final
        return rho_optimum.astype(np.float32), opti_steps_history

    # --- render method remains the same as previous version ---
    def render(self, design: npt.NDArray, config: dict[str, Any] | None = None, *, open_window: bool = False) -> Any:
        """Renders the design (rho) and the resulting E-field magnitudes.

           Runs a simulation for the provided design to get the fields for plotting.
           Uses the internally stored fields if called immediately after `simulate`.

        Args:
            design (npt.NDArray): The design `rho` to render.
            open_window (bool): If True, opens a window with the rendered plot.
            config (dict): Config overrides for simulation parameters if needed for rendering.
            **kwargs: Additional keyword arguments (ignored).


        Returns:
            plt.Figure: The matplotlib Figure object containing three plots:
                        |Ez| at omega1, |Ez| at omega2, and Permittivity (eps_r).
        """
        conditions = self._setup_simulation(config)

        print("Rendering design under the following conditions:")
        pprint.pp(conditions)
        # Use run_fdfd but ignore most outputs, just need epsr, ez1, ez2
        epsr, ez1, ez2, _, _, _, _ = self._run_fdfd(design, conditions)

        # Store these fields as the "last" simulated ones as well
        self._last_epsr = epsr.copy()
        self._last_Ez1 = ez1.copy()
        self._last_Ez2 = ez2.copy()

        # --- Plotting ---
        fig, ax = plt.subplots(1, 3, constrained_layout=True, figsize=(9, 3))
        ceviche.viz.abs(
            ez1,
            outline=epsr,
            ax=ax[0],
            cbar=False,
            outline_alpha=0.25,
        )
        ceviche.viz.abs(
            ez2,
            outline=epsr,
            ax=ax[1],
            cbar=False,
            outline_alpha=0.25,
        )
        ceviche.viz.real(epsr, ax=ax[2], cmap="Greys")
        slices_to_plot = [self._input_slice, self._output_slice1, self._output_slice2]
        for sl in slices_to_plot:
            if sl:
                for axis in ax[:2]:  # Plot on field plots
                    axis.plot(sl.x * np.ones(len(sl.y)), sl.y, "w-", alpha=0.5, linewidth=1)
        lambda1_um = conditions["lambda1"]
        lambda2_um = conditions["lambda2"]
        ax[0].set_title(f"|Ez| at $\\lambda_1$ = {lambda1_um:.2f} $\\mu$m")
        ax[1].set_title(f"|Ez| at $\\lambda_2$ = {lambda2_um:.2f} $\\mu$m")
        ax[2].set_title(r"Permittivity $\epsilon_r$")
        for axis in ax:
            axis.set_xlabel("")
            axis.set_ylabel("")
            axis.set_xticks([])
            axis.set_yticks([])

        plt.tight_layout()
        if open_window:
            plt.show(block=False)
        return fig

    def _randomized_noise_field_design(self, noise: float = 0.001, blur: int = 0) -> npt.NDArray:
        """Generates a starting design with small random variations.

           Creates a design that is 0.5 within the design region, plus small
           normal random noise (0.001 * randn). Returns 0 as the index placeholder.

        Args:
            noise (float): The amount of noise to add to the uniform field.
            blur (float): The amount of blurring to apply to random field. if >0 can produce
                          different local optima for Adam, and thus can be useful for
                          exploring multiple local optima in the problem. Disabled by default.

        Returns:
            tuple[npt.NDArray, int]: The starting design array (rho) and an integer (0).
        """
        space = self._pml_space
        num_elems_pml = self._num_elems_pml

        design_region = np.zeros((self.num_elems_x, self.num_elems_y))
        design_region[
            num_elems_pml + space : self.num_elems_x - num_elems_pml - space,
            num_elems_pml + space : self.num_elems_y - num_elems_pml - space,
        ] = 1

        # Ensure np_random is initialized
        if self.np_random is None:
            self.reset()
        # Generate random numbers using the problem's RNG
        # Use randomized initialization -- for now keep
        random_noise = noise * self.np_random.standard_normal((self.num_elems_x, self.num_elems_y))
        rho_start = design_region * (0.5 + random_noise)
        if blur > 0.0:
            rho_start = operator_blur(rho_start, blur)

        return rho_start.astype(np.float32)

    def random_design(self, noise: float | None = None, blur: int = 0) -> tuple[npt.NDArray, int]:
        """Generates a random initial design.

        Can return a design with small random variations or a uniform design, or can pull
        from the datasets (when available).

        Args:
            noise (float|None): If None, pull from dataset. If float, use that as the noise level.
            blur (int): The amount of pixel blurring to apply to random field. Only active if noise is used.

        Returns:
            tuple[npt.NDArray, int]: The starting design array (rho) and an integer (0).
        """
        # Ensure np_random is initialized
        if self.np_random is None:
            self.reset()

        if noise is not None:
            rho_start = self._randomized_noise_field_design(noise=noise, blur=blur)
            return rho_start, 0
        rnd = self.np_random.integers(low=0, high=len(self.dataset["train"]), dtype=int)
        return np.array(self.dataset["train"]["optimal_design"][rnd]), rnd

    def reset(self, seed: int | None = None, **kwargs) -> None:
        """Resets the problem, which in this case, is just the random seed."""
        return super().reset(seed, **kwargs)


# --- Example Usage (main block) ---
if __name__ == "__main__":
    # Problem Configuration Example
    problem_config = {
        "lambda1": 1.07,
        "lambda2": 0.84,
        "blur_radius": 1,
    }
    problem = Photonics2D(config=problem_config, num_elems_x=120, num_elems_y=120)
    problem.reset(seed=42)  # Use a seed

    start_design, _ = problem.random_design(noise=0.001)  # Randomized design with noise
    fig_start = problem.render(start_design)

    # Simulation Example
    print("Simulating starting design...")
    # Simulate returns the raw objective = penalty - overlap1*overlap2
    obj_start_raw = problem.simulate(start_design)
    print(f"Starting Raw Objective ({problem.objectives_keys[0]}): {obj_start_raw[0]:.4f}")

    # Optimization Example
    # Advanced Usage: Modifying optimization parameters
    opt_config = {"num_optimization_steps": 100, "save_frame_interval": 2}
    print(f"Optimizing design with ({opt_config})...")
    # Optimize maximizes (normalized_overlap - penalty)
    optimized_design, opti_history = problem.optimize(start_design, config=opt_config)
    print(f"Optimization finished. History length: {len(opti_history)}")
    if opti_history:
        print(f"First step objective (normalized, minimized): {opti_history[0].obj_values[0]:.4f}")
        print(f"Last step objective (normalized, minimized): {opti_history[-1].obj_values[0]:.4f}")

    print("Rendering optimized design...")
    fig_opt = problem.render(optimized_design, open_window=True)
    frame_dir = "opt_frames"
    fig_opt.savefig(frame_dir + "/optimized_design.png", dpi=200)

    print("Simulating the final optimized design...")
    # Simulate returns the raw objective = penalty - overlap1*overlap2
    obj_opt_raw = problem.simulate(optimized_design)
    print(f"Optimized Raw Objective ({problem.objectives_keys[0]}): {obj_opt_raw[0]:.4f}")

    if plt.get_fignums():
        print("Close plot window(s) to exit.")
        plt.show()
