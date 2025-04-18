"""Photonics2D problem.

This is essentially re-factored from the code at
https://nbviewer.org/github/fancompute/workshop-invdesign/blob/master/04_Invdes_wdm_scheduling.ipynb

Author: Mark Fuge @markfuge
"""

from __future__ import annotations

# Need os import for makedirs for saving plots
import os
import pprint
from typing import Any

import autograd.numpy as npa

# Import ArrayBox type for checking
from autograd.numpy.numpy_boxes import ArrayBox
import ceviche
from ceviche import fdfd_ez
from ceviche import jacobian
from ceviche.optimizers import adam_optimize
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt_save
import numpy as np
import numpy.typing as npt

# --- EngiBench Core Imports ---
# Import necessary base classes and types directly from the library's core
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
from engibench.problems.photonics2d.backend import Slice
from engibench.problems.photonics2d.backend import wavelength_to_frequency


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
    - `beta`: Initial Projection strength parameter (default: 10.0, then scheduled during opt.).
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
    _num_optimization_steps_default = 300  # Default number of optimization steps
    _beta_default = 10.0  # Default projection strength parameter
    _step_size_default = 1e-1  # Default step size for Adam optimizer
    _eta_default = 0.5
    _num_projections_default = 1
    _penalty_weight_default = 1e-2  # Default weight for mass penalty term
    _num_blurs_default = 1

    conditions: tuple[tuple[str, Any], ...] = (
        ("lambda1", 1.5),  # First input wavelength in μm
        ("lambda2", 1.3),  # Second input wavelength in μm
        ("blur_radius", 2),  # Radius for the density blurring filter (pixels)
        ("num_elems_x", _num_elems_x_default),  # Number of grid cells in x (pixels)
        ("num_elems_y", _num_elems_y_default),  # Number of grid cells in y (pixels)
    )

    design_space = spaces.Box(low=0.0, high=1.0, shape=(_num_elems_x_default, _num_elems_y_default), dtype=np.float64)

    dataset_id = f"IDEALLab/photonicmultiplexer_2d_{_num_elems_x_default}_{_num_elems_y_default}_v0"
    container_id = None  # type: ignore
    _dataset = None

    def __init__(self, config: dict[str, Any], **kwargs) -> None:
        """Initializes the Photonics2D problem.

        Args:
            config (dict): A dictionary with configuration (e.g., boundary conditions) for the simulation.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)

        # Replace the conditions with any new configs passed in
        self.conditions = tuple((key, config.get(key, value)) for key, value in self.conditions)
        current_conditions = dict(self.conditions)
        print("Initializing Photonics Problem with configuration:")
        pprint.pp(current_conditions)
        num_elems_x = current_conditions.get("num_elems_x", self._num_elems_x_default)
        num_elems_y = current_conditions.get("num_elems_y", self._num_elems_y_default)
        self.design_space = spaces.Box(low=0.0, high=1.0, shape=(num_elems_x, num_elems_y), dtype=np.float32)
        self.dataset_id = f"IDEALLab/photonics_2d_{num_elems_x}_{num_elems_y}_v{self.version}"

        # Setup basic simulation parameters
        self.omega1 = wavelength_to_frequency(current_conditions["lambda1"])
        self.omega2 = wavelength_to_frequency(current_conditions["lambda2"])
        self._current_beta = current_conditions.get("beta", self._beta_default)

        # --- Private attributes for simulation state ---
        _bg_rho: npt.NDArray
        _design_region: npt.NDArray
        _input_slice: Slice
        _output_slice1: Slice
        _output_slice2: Slice
        _simulation1: fdfd_ez
        _simulation2: fdfd_ez
        _source1: npt.NDArray  # Used only during optimize
        _source2: npt.NDArray  # Used only during optimize
        _probe1: npt.NDArray  # Used only during optimize
        _probe2: npt.NDArray  # Used only during optimize
        _e01: float = 0.0  # Normalization constant (used only during optimize)
        _e02: float = 0.0  # Normalization constant (used only during optimize)
        _current_beta: float = 10

        # --- Attributes to store last simulation results ---
        _last_epsr: npt.NDArray
        _last_ez1: npt.NDArray
        _last_ez2: npt.NDArray

    def _setup_simulation(self, config: dict[str, Any]) -> dict[str, Any]:
        """Helper function to setup simulation parameters and domain."""
        # Merge config with default conditions
        current_conditions = dict(self.conditions)
        current_conditions.update(config)

        num_elems_x = current_conditions["num_elems_x"]
        num_elems_y = current_conditions["num_elems_y"]

        # Initialize domain geometry
        self._bg_rho, self._design_region, self._input_slice, self._output_slice1, self._output_slice2 = init_domain(
            num_elems_x=num_elems_x,
            num_elems_y=num_elems_y,
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
        # Use scheduled beta if optimizing, otherwise use beta from conditions
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
        self._simulation1 = fdfd_ez(omega1, self._dl, epsr, [self._num_elems_pml, self._num_elems_pml])
        self._simulation2 = fdfd_ez(omega2, self._dl, epsr, [self._num_elems_pml, self._num_elems_pml])

        # 4. Solve FDFD
        # Always solve as Ez fields are needed for return value or objective calc
        _, _, ez1 = self._simulation1.solve(source1)
        _, _, ez2 = self._simulation2.solve(source2)

        return epsr, ez1, ez2, source1, source2, probe1, probe2

    def simulate(self, design: npt.NDArray, config: dict[str, Any] = {}, **kwargs) -> npt.NDArray:  # noqa: ARG002
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
        num_elems_x = conditions["num_elems_x"]
        num_elems_y = conditions["num_elems_y"]
        if design.shape != (num_elems_x, num_elems_y):
            raise ValueError(f"Input design shape {design.shape} does not match conditions ({num_elems_x}, {num_elems_y})")  # noqa: TRY003

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
        config: dict[str, Any] = {},
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
        num_elems_x = conditions["num_elems_x"]
        num_elems_y = conditions["num_elems_y"]
        # Pull out optimization parameters from conditions
        # Parameters specific to optimization
        num_optimization_steps = conditions.get("num_optimization_steps", self._num_optimization_steps_default)
        step_size = conditions.get("step_size", self._step_size_default)
        penalty_weight = conditions.get("penalty_weight", self._penalty_weight_default)
        self._current_beta = conditions.get("beta", self._beta_default)
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

        # --- Define Objective Function for Ceviche Optimizer ---
        def objective_for_optimizer(rho_flat: npt.NDArray | ArrayBox) -> float | ArrayBox:
            """Calculates (normalized_overlap - penalty) for maximization."""
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
        # No need for of_list_for_beta

        def callback(iteration: int, objective_history_list: list, rho_flat: npt.NDArray | ArrayBox) -> None:
            """Callback for adam_optimize. Receives the history of objective values."""
            # Handle Empty History
            if not objective_history_list:
                return

            # Get the latest objective value
            last_scalar_obj_value = objective_history_list[-1]

            # Check if the latest value is valid
            if not isinstance(last_scalar_obj_value, (int, float, np.number, npa.numpy_boxes.ArrayBox)):
                print(
                    f"!!! WARNING: Last objective value in history is not numeric at iter {iteration}: Type={type(last_scalar_obj_value)}, Val={last_scalar_obj_value} !!! Skipping processing."
                )  # Keep warning
                return

            # --- Process Valid Scalar Objective Value ---

            # Beta Scheduling Logic
            iteration = len(objective_history_list)
            # Spend first half on low continuation
            beta_schedule = [10, 100, 200, max_beta]
            early_continuation = num_optimization_steps / 2
            mid_continuation = num_optimization_steps * 3 / 4
            if iteration < early_continuation:
                self._current_beta = beta_schedule[0]  # 10
            elif early_continuation <= iteration & iteration < mid_continuation:
                if self._current_beta == beta_schedule[0]:
                    print(f"Increasing beta to {beta_schedule[1]}...")
                self._current_beta = beta_schedule[1]
            elif mid_continuation <= iteration & iteration < num_optimization_steps - 1:
                if self._current_beta == beta_schedule[1]:
                    print(f"Increasing beta to {beta_schedule[2]}...")
                self._current_beta = beta_schedule[2]
            else:  # Final step continuation should be max_beta
                print(f"Final beta set to {beta_schedule[3]}...")
                self._current_beta = beta_schedule[3]
            # --- End Beta Logic ---

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

                # --- Call self.render to generate the plot ---
                # Pass open_window=False as we just want the figure object
                # Pass the current conditions dictionary in case render needs it
                # Note: This will re-run the simulation for the current_rho
                fig = self.render(current_rho, open_window=False, config=conditions)
                # ---------------------------------------------

                # Ensure directory exists
                frame_dir = "opt_frames"
                os.makedirs(frame_dir, exist_ok=True)
                save_path = os.path.join(frame_dir, f"frame_iter_{iteration:04d}.png")  # Renamed file for clarity

                # Save the figure returned by render
                fig.savefig(save_path, dpi=200)
                plt_save.close(fig)  # Close the figure to free memory
                print(f"Callback Iter {iteration}: Saved frame to {save_path}")
            # --- End Frame Saving ---

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
        rho_optimum = operator_proj(rho_optimum, self._eta, beta=max_beta, num_projections=1)

        return rho_optimum.astype(np.float32), opti_steps_history

    # --- render method remains the same as previous version ---
    def render(self, design: npt.NDArray, open_window: bool = False, config: dict[str, Any] = {}, **kwargs) -> plt.Figure:  # noqa: ARG002
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
        num_elems_x = conditions["num_elems_x"]
        num_elems_y = conditions["num_elems_y"]
        if design.shape != (num_elems_x, num_elems_y):
            raise ValueError(f"Input design shape {design.shape} != ({num_elems_x}, {num_elems_y})")  # noqa: TRY003

        print("Rendering design under the following conditions:")
        pprint.pp(conditions)
        # Run simulation for the given design to get fields for plotting
        conditions["beta"] = conditions.get("beta", self._beta_default)
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
        current_conditions = dict(self.conditions)
        num_elems_x = current_conditions.get("num_elems_x", self._num_elems_x_default)
        num_elems_y = current_conditions.get("num_elems_y", self._num_elems_y_default)
        space = self._pml_space
        num_elems_pml = self._num_elems_pml

        design_region = np.zeros((num_elems_x, num_elems_y))
        design_region[
            num_elems_pml + space : num_elems_x - num_elems_pml - space,
            num_elems_pml + space : num_elems_y - num_elems_pml - space,
        ] = 1

        # Ensure np_random is initialized
        if self.np_random is None:
            self.reset()
        # Generate random numbers using the problem's RNG
        # Use randomized initialization -- for now keep
        random_noise = noise * self.np_random.standard_normal((num_elems_x, num_elems_y))  # type: ignore
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
        elif self._dataset is not None:
            rnd = self.np_random.integers(low=0, high=len(self.dataset["train"]), dtype=int)  # type:ignore
            return np.array(self.dataset["train"]["optimal_design"][rnd]), rnd  # type:ignore
        else:
            # If noise is None, yet no dataset is available, raise an error.
            # This can be removed once HF dataset is created and live.
            raise NotImplementedError("Dataset not yet available. Please set noise to a float value.")

    def reset(self, seed: int | None = None, **kwargs) -> None:
        """Resets the problem, which in this case, is just the random seed."""
        return super().reset(seed, **kwargs)


# --- Example Usage (main block) ---
if __name__ == "__main__":
    # Problem Configuration Example
    problem_config = {
        "lambda1": 1.11,
        "lambda2": 0.99,
        "blur_radius": 1,
        "num_elems_x": 120,
        "num_elems_y": 120,
    }
    problem = Photonics2D(config=problem_config)
    problem.reset(seed=42)  # Use a seed

    start_design, _ = problem.random_design(noise=0.001)  # Randomized design with noise
    fig_start = problem.render(start_design)

    # Simulation Example
    print("Simulating starting design...")
    # Simulate returns the raw objective = penalty - overlap1*overlap2
    obj_start_raw = problem.simulate(start_design)
    print(f"Starting Raw Objective ({problem.objectives[0][0]}): {obj_start_raw[0]:.4f}")

    # Optimization Example
    # Advanced Usage: Modifying optimization parameters
    opt_config = {"num_optimization_steps": 10, "save_frame_interval": 2}
    print(f"Optimizing design with ({opt_config})...")
    # Optimize maximizes (normalized_overlap - penalty)
    optimized_design, opti_history = problem.optimize(start_design, config=opt_config)
    print(f"Optimization finished. History length: {len(opti_history)}")
    if opti_history:
        print(f"First step objective (normalized, minimized): {opti_history[0].obj_values[0]:.4f}")
        print(f"Last step objective (normalized, minimized): {opti_history[-1].obj_values[0]:.4f}")

    print("Rendering optimized design...")
    fig_opt = problem.render(optimized_design, open_window=True)

    print("Simulating the final optimized design...")
    # Simulate returns the raw objective = penalty - overlap1*overlap2
    obj_opt_raw = problem.simulate(optimized_design)
    print(f"Optimized Raw Objective ({problem.objectives[0][0]}): {obj_opt_raw[0]:.4f}")

    if plt.get_fignums():
        print("Close plot window(s) to exit.")
        plt.show()
