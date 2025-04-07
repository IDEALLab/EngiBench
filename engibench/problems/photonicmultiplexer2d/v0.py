"""PhotonicsMultiplexer2D problem.

This is essentially re-factored from the code at
https://nbviewer.org/github/fancompute/workshop-invdesign/blob/master/04_Invdes_wdm_scheduling.ipynb

Author: Mark Fuge @markfuge
"""
from __future__ import annotations

# Need os import for makedirs for saving plots
import os
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
from engibench.problems.photonicmultiplexer2d.backend import epsr_parameterization

# --- EngiBench Problem-Specific Backend ---
from engibench.problems.photonicmultiplexer2d.backend import init_domain
from engibench.problems.photonicmultiplexer2d.backend import insert_mode
from engibench.problems.photonicmultiplexer2d.backend import mode_overlap
from engibench.problems.photonicmultiplexer2d.backend import Slice


class PhotonicMultiplexer2D(Problem[npt.NDArray]):
    r"""Photonic Inverse Design 2D Problem (Wavelength Demultiplexer).

    ## Problem Description
    Optimize a 2D material distribution (`rho`) to function as a wavelength
    demultiplexer, routing `omega1` to output 1 and `omega2` to output 2. The
    design variables represent material density which is converted to permittivity
    using filtering and projection.

    ## Design space
    2D tensor `rho` (num_elems_x, num_elems_y) with values in [0, 1], representing material density.
    Stored as `design_space` (gymnasium.spaces.Box).

    ## Objectives
    0. `raw_objective`: Combined objective to minimize, defined as
       `penalty - overlap1 * overlap2`. Lower is better. This is the value
       returned by `simulate`. Note that `optimize` internally works with a
       normalized version for stability (`penalty - normalized_overlap`) and
       reports history (`OptiStep`) corresponding to that normalized version.

    ## Conditions (Configurable Parameters)
    Default problem parameters that can be overridden via the `config` dict:
    - `omega1`: Angular frequency 1 (default: 2*pi*200e12).
    - `omega2`: Angular frequency 2 (default: 2*pi*230e12).
    - `blur_radius`: Radius for the density blurring filter (default: 2).
    - `num_elems_x`: Number of grid cells in x (default: 120).
    - `num_elems_y`: Number of grid cells in y (default: 120).
    - `space`: Space between PML and design region (pixels) (default: 10).
    - `wg_width`: Width of waveguides (pixels) (default: 12).
    - `wg_shift`: Lateral shift for output waveguides (pixels) (default: 9).
    - `num_optimization_steps`: Number of optimization steps (default: 100).
    - `step_size`: Adam optimizer step size (default: 1e-2).
    - `penalty_weight`: Weight for the L2 penalty term (default: 1e-2).
    - `beta`: Projection strength parameter (default: 10.0, can be scheduled).
    - `eta`: Projection center parameter (default: 0.5).
    - `N_proj`: Number of projection applications (default: 1).
    - `N_blur`: Number of blur applications (default: 1).
    - `save_frame_interval`: Interval for saving intermediate design frames during optimization.
                             If > 0, saves a frame every `save_frame_interval` iterations
                             to the `opt_frames/` directory. Default is 0 (disabled).

    ## Internal Constants (Not typically changed via config)
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
        title={Forward-Mode Differentiation of Maxwell’s Equations},
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
    objectives: tuple[tuple[str, ObjectiveDirection]] = (
        ("raw_objective", ObjectiveDirection.MINIMIZE),
    )
    # Note: optimize internally uses a normalized objective, see OptiStep values.
    # We keep a single objective name for simplicity in the list.

    # --- Default Conditions, Internal Constants, Design Space (same as before) ---
    _omega1_default = 2 * np.pi * 200e12
    _omega2_default = 2 * np.pi * 230e12
    _blur_radius_default = 2
    _num_elems_x_default = 120
    _num_elems_y_default = 120
    _space_default = 10
    _wg_width_default = 12
    _wg_shift_default = 9
    _max_opt_steps_default = 100
    _step_size_default = 1e-2
    _penalty_weight_default = 1e-7
    _beta_default = 10.0
    _eta_default = 0.5
    _num_projections_default = 1
    _num_blurs_default = 1
    _save_frame_interval_default = 0 # 0 means disabled by default

    conditions: tuple[tuple[str, Any], ...] = (
        ("omega1", _omega1_default), ("omega2", _omega2_default),
        ("blur_radius", _blur_radius_default), ("num_elems_x", _num_elems_x_default), ("num_elems_y", _num_elems_y_default),
        ("space", _space_default), ("wg_width", _wg_width_default), ("wg_shift", _wg_shift_default),
        ("max_opt_steps", _max_opt_steps_default), ("step_size", _step_size_default),
        ("penalty_weight", _penalty_weight_default), ("beta", _beta_default),
        ("eta", _eta_default), ("num_projections", _num_projections_default), ("num_blurs", _num_blurs_default),
        ("save_frame_interval", _save_frame_interval_default), 
    )

    dl = 40e-9
    num_elems_pml = 20
    epsr_min = 1.0
    epsr_max = 12.0
    space_slice = 8

    design_space = spaces.Box(
        low=0.0, high=1.0, shape=(_num_elems_x_default, _num_elems_y_default), dtype=np.float32
    )

    # TODO: until I have this part working with the dataset generation
    dataset_id = None
    container_id = None
    _dataset = None

    # --- Private attributes for simulation state ---
    _bg_rho: npt.NDArray | None = None
    _design_region: npt.NDArray | None = None
    _input_slice: Slice | None = None
    _output_slice1: Slice | None = None
    _output_slice2: Slice | None = None
    _simulation1: fdfd_ez | None = None
    _simulation2: fdfd_ez | None = None
    _source1: npt.NDArray | None = None # Used only during optimize
    _source2: npt.NDArray | None = None # Used only during optimize
    _probe1: npt.NDArray | None = None # Used only during optimize
    _probe2: npt.NDArray | None = None # Used only during optimize
    _E01: float = 0.0 # Normalization constant (used only during optimize)
    _E02: float = 0.0 # Normalization constant (used only during optimize)
    _current_beta: float = _beta_default

    # --- Attributes to store last simulation results ---
    _last_epsr: npt.NDArray | None = None
    _last_Ez1: npt.NDArray | None = None
    _last_Ez2: npt.NDArray | None = None


    def __init__(self, **kwargs) -> None:
        """Initializes the Photonics2D problem."""
        super().__init__(**kwargs)
        current_conditions = dict(self.conditions)
        num_elems_x = current_conditions.get("num_elems_x", self._num_elems_x_default)
        num_elems_y = current_conditions.get("num_elems_y", self._num_elems_y_default)
        self.design_space = spaces.Box(
            low=0.0, high=1.0, shape=(num_elems_x, num_elems_y), dtype=np.float32
        )
        self._reset_simulation_state()


    def _reset_simulation_state(self) -> None:
        """Helper to clear simulation-specific state."""
        self._bg_rho = None
        self._design_region = None
        self._input_slice = None
        self._output_slice1 = None
        self._output_slice2 = None
        self._simulation1 = None
        self._simulation2 = None
        self._source1 = None
        self._source2 = None
        self._probe1 = None
        self._probe2 = None
        self._E01 = 0.0
        self._E02 = 0.0
        self._last_epsr = None
        self._last_Ez1 = None
        self._last_Ez2 = None
        self._current_beta = dict(self.conditions).get("beta", self._beta_default)

    def reset(self, seed: int | None = None, **kwargs) -> None:
        """Resets the simulator state and numpy random seed."""
        super().reset(seed=seed, **kwargs)
        if self.np_random is None: self.np_random = np.random.default_rng(seed)
        # Seed legacy numpy random if needed, using the generator from base class
        np.random.seed(self.np_random.integers(0, 2**32 - 1))
        self._reset_simulation_state()

    # --- _setup_simulation, _run_fdfd helpers remain the same as previous version ---
    def _setup_simulation(self, config: dict[str, Any]) -> dict[str, Any]:
        """Helper function to setup simulation parameters and domain."""
        # Merge config with default conditions
        current_conditions = dict(self.conditions)
        current_conditions.update(config)

        num_elems_x = current_conditions["num_elems_x"]
        num_elems_y = current_conditions["num_elems_y"]

        # Initialize domain geometry (only if not already done or if size changed)
        if self._design_region is None or self._design_region.shape != (num_elems_x, num_elems_y):
             # --- This is the corrected unpacking line (expects 5 items) ---
             self._bg_rho, self._design_region, self._input_slice, \
             self._output_slice1, self._output_slice2 = init_domain(
                 num_elems_x=num_elems_x, num_elems_y=num_elems_y, num_elems_pml=self.num_elems_pml, space=current_conditions["space"],
                 wg_width=current_conditions["wg_width"],
                 space_slice=self.space_slice, wg_shift=current_conditions["wg_shift"]
             )
             # -------------------------------------------------------------

        return current_conditions

    def _run_fdfd(self, design: npt.NDArray, conditions: dict[str, Any]) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """Helper to run FDFD and return key components (epsr, fields, sources, probes)."""
        omega1 = conditions["omega1"]
        omega2 = conditions["omega2"]
        # Use scheduled beta if optimizing, otherwise use beta from conditions
        beta = self._current_beta if self._simulation1 is not None else conditions["beta"]

        # 1. Parameterize
        epsr = epsr_parameterization(
            rho=design, bg_rho=self._bg_rho, design_region=self._design_region,
            radius=conditions["blur_radius"], num_blurs=conditions["num_blurs"],
            beta=beta, eta=conditions["eta"], num_projections=conditions["num_projections"],
            epsr_min=self.epsr_min, epsr_max=self.epsr_max
        )

        # 2. Setup Sources and Probes (depend on epsr)
        source1 = insert_mode(omega1, self.dl, self._input_slice.x, self._input_slice.y, epsr, m=1)
        source2 = insert_mode(omega2, self.dl, self._input_slice.x, self._input_slice.y, epsr, m=1)
        probe1 = insert_mode(omega1, self.dl, self._output_slice1.x, self._output_slice1.y, epsr, m=1)
        probe2 = insert_mode(omega2, self.dl, self._output_slice2.x, self._output_slice2.y, epsr, m=1)

        # 3. Setup FDFD Simulations
        # TODO: Not really sure if I need this check
        if self._simulation1 is None or self._simulation1.omega != omega1:
            self._simulation1 = fdfd_ez(omega1, self.dl, epsr, [self.num_elems_pml, self.num_elems_pml])
        # Update eps_r only if it actually changes to potentially avoid cache misses inside solve if any exist
        elif not np.array_equal(self._simulation1.eps_r, epsr): 
             self._simulation1.eps_r = epsr

        # TODO: Not really sure if I need this check
        if self._simulation2 is None or self._simulation2.omega != omega2:
            self._simulation2 = fdfd_ez(omega2, self.dl, epsr, [self.num_elems_pml, self.num_elems_pml])
        elif not np.array_equal(self._simulation2.eps_r, epsr):
             self._simulation2.eps_r = epsr
        # 4. Solve FDFD
        # Always solve as Ez fields are needed for return value or objective calc
        _, _, Ez1 = self._simulation1.solve(source1)
        _, _, Ez2 = self._simulation2.solve(source2)

        return epsr, Ez1, Ez2, source1, source2, probe1, probe2

    def simulate(self, design: npt.NDArray, config: dict[str, Any] = {}, **kwargs) -> npt.NDArray:
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
             raise ValueError(f"Input design shape {design.shape} does not match conditions ({num_elems_x}, {num_elems_y})")

        # --- Run Simulation ---
        # We don't need source returns here
        epsr, Ez1, Ez2, _, _, probe1, probe2 = self._run_fdfd(design, conditions)

        # --- Store Results Internally ---
        self._last_epsr = epsr.copy()
        self._last_Ez1 = Ez1.copy()
        self._last_Ez2 = Ez2.copy()

        # --- Calculate Raw Objective (No Normalization) ---
        # Use standard numpy here, no gradients needed
        overlap1 = np.abs(np.sum(np.conj(Ez1) * probe1)) * 1e6
        overlap2 = np.abs(np.sum(np.conj(Ez2) * probe2)) * 1e6
        penalty = conditions["penalty_weight"] * np.linalg.norm(design)

        raw_objective = penalty - (overlap1 * overlap2) # Minimize this

        return np.array([raw_objective], dtype=np.float64)


    def optimize(self, starting_point: npt.NDArray, config: dict[str, Any] = {}, **kwargs) -> tuple[npt.NDArray, list[OptiStep]]:
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

        num_elems_x = conditions["num_elems_x"]
        num_elems_y = conditions["num_elems_y"]
        num_optimization_steps = conditions["num_optimization_steps"]
        step_size = conditions["step_size"]
        penalty_weight = conditions["penalty_weight"]
        self._current_beta = conditions["beta"]
        # --- Get the frame saving interval from conditions ---
        save_frame_interval = conditions["save_frame_interval"]
        # --- Initial Simulation for Normalization Constants E01, E02 ---
        print("Optimize: Calculating E01/E02 using starting_point...") # Keep this info message
        epsr_init, Ez1_init, Ez2_init, source1_init, source2_init, probe1_init, probe2_init = self._run_fdfd(starting_point, conditions)

        self._E01 = npa.abs(npa.sum(npa.conj(Ez1_init) * probe1_init)) * 1e6
        self._E02 = npa.abs(npa.sum(npa.conj(Ez2_init) * probe2_init)) * 1e6

        if self._E01 == 0 or self._E02 == 0:
             print(f"Warning: Initial overlap zero (E01={self._E01:.3e}, E02={self._E02:.3e}). Using fallback.") # Keep this warning
             self._E01 = self._E01 if self._E01 != 0 else 1e-9
             self._E02 = self._E02 if self._E02 != 0 else 1e-9

        self._source1 = source1_init
        self._source2 = source2_init
        self._probe1 = probe1_init
        self._probe2 = probe2_init

        # --- Define Objective Function for Ceviche Optimizer ---
        # Memoization is disabled
        def objective_for_optimizer(rho_flat):
            """Calculates (normalized_overlap - penalty) for maximization."""
            rho = rho_flat.reshape((num_elems_x, num_elems_y))
            conditions["beta"] = self._current_beta # Use scheduled beta

            # --- Parameterization and Simulation ---
            epsr = epsr_parameterization(
                 rho=rho, bg_rho=self._bg_rho, design_region=self._design_region,
                 radius=conditions["blur_radius"], num_blurs=conditions["num_blurs"],
                 beta=self._current_beta, eta=conditions["eta"], num_projections=conditions["num_projections"],
                 epsr_min=self.epsr_min, epsr_max=self.epsr_max
            )
            if self._simulation1 is None or self._simulation2 is None: raise RuntimeError("Sim objects not set")
            self._simulation1.eps_r = epsr
            self._simulation2.eps_r = epsr
            _, _, Ez1 = self._simulation1.solve(self._source1)
            _, _, Ez2 = self._simulation2.solve(self._source2)

            # Calculate overlaps
            overlap1 = mode_overlap(Ez1, self._probe1)
            overlap2 = mode_overlap(Ez2, self._probe2)
            current_E01 = self._E01 # Assume already handled zero case
            current_E02 = self._E02
            normalized_overlap = (overlap1 / current_E01) * (overlap2 / current_E02)

            penalty = penalty_weight * npa.linalg.norm(rho)
            # print(f"input {type(rho_flat)}")
            # print(f"output {type(normalized_overlap - penalty)}")
            return normalized_overlap - penalty # Value to MAXIMIZE

        # --- Define Gradient ---
        objective_jac = jacobian(objective_for_optimizer, mode='reverse')

        # --- Define Callback ---
        opti_steps_history: list[OptiStep] = []
        # No need for of_list_for_beta

        def callback(iteration, objective_history_list, rho_flat):
            """Callback for adam_optimize. Receives the history of objective values."""
            # Handle Empty History
            if not objective_history_list:
                return

            # Get the latest objective value
            last_scalar_obj_value = objective_history_list[-1]

            # Check if the latest value is valid
            if not isinstance(last_scalar_obj_value, (int, float, np.number, npa.numpy_boxes.ArrayBox)):
                 print(f"!!! WARNING: Last objective value in history is not numeric at iter {iteration}: Type={type(last_scalar_obj_value)}, Val={last_scalar_obj_value} !!! Skipping processing.") # Keep warning
                 return

            # --- Process Valid Scalar Objective Value ---

            max_beta = 300
            # Beta Scheduling Logic
            if len(objective_history_list) >= 2:
                # --- Start Beta Logic ---
                differences = np.diff(objective_history_list)
                prev_values = np.array(objective_history_list[:-1])
                valid_indices = prev_values != 0
                percentage_changes = np.zeros_like(differences)
                len_diff = len(differences)
                if len(valid_indices) >= len_diff > 0:
                    valid_indices_aligned = valid_indices[:len_diff]
                    diff_indices = valid_indices_aligned < len(differences)
                    prev_indices = valid_indices_aligned < len(prev_values)
                    aligned_indices = diff_indices & prev_indices & valid_indices_aligned
                    if np.any(aligned_indices):
                         percentage_changes[aligned_indices] = np.abs(differences[aligned_indices] / prev_values[aligned_indices]) * 100
                if len(percentage_changes) > 0:
                    last_change = percentage_changes[-1]
                    percentile_5 = np.percentile(percentage_changes, 5) if len(percentage_changes) > 1 else last_change
                    if ((not np.isnan(last_change) and not np.isnan(percentile_5))
                        and (last_change <= percentile_5 and self._current_beta < max_beta)):
                        self._current_beta += 5
                        print(f"Callback Iter {iteration}: Increasing beta to {self._current_beta}") # Commented out
            # --- End Beta Logic ---

            # Store OptiStep info
            neg_norm_objective_value = -last_scalar_obj_value
            step_info = OptiStep(
                 obj_values=np.array([neg_norm_objective_value], dtype=np.float64),
                 step=iteration
            )
            opti_steps_history.append(step_info)

            # --- Configurable Intermediate Frame Saving ---
            # Check if saving is enabled and if current iteration is a multiple of the interval
            # Also check iteration > 0 to avoid saving the initial state redundantly
            if save_frame_interval is not None and save_frame_interval > 0 and iteration > 0 and iteration % save_frame_interval == 0:
                try:
                    print(f"Callback Iter {iteration}: Generating render frame...") # Info message

                    # Reshape the current design parameters
                    current_rho = rho_flat.reshape((num_elems_x, num_elems_y))

                    # --- Call self.render to generate the plot ---
                    # Pass open_window=False as we just want the figure object
                    # Pass the current conditions dictionary in case render needs it
                    # Note: This will re-run the simulation for the current_rho
                    fig = self.render(current_rho, open_window=False, config=conditions)
                    # ---------------------------------------------

                    # Ensure directory exists
                    frame_dir = 'opt_frames'
                    os.makedirs(frame_dir, exist_ok=True)
                    save_path = os.path.join(frame_dir, f'frame_iter_{iteration:04d}.png') # Renamed file for clarity

                    # Save the figure returned by render
                    fig.savefig(save_path, dpi=100)
                    plt_save.close(fig) # Close the figure to free memory
                    print(f"Callback Iter {iteration}: Saved frame to {save_path}")

                except Exception as e:
                    # Print traceback for detailed debugging if rendering fails
                    import traceback
                    print(f"Callback Iter {iteration}: Failed to render/save frame: {e}")
                    traceback.print_exc() # Print full traceback for render errors
            # --- End Frame Saving ---

        # --- Run Optimization ---
        print(f"\nStarting optimization with num_optimization_steps={num_optimization_steps}, step_size={step_size}") # Keep start message
        (rho_optimum_flat, _) = adam_optimize(
             objective_for_optimizer, starting_point.flatten(), objective_jac,
             Nsteps=num_optimization_steps, direction='max', step_size=step_size, callback=callback
        )

        # --- Final Result ---
        rho_optimum = rho_optimum_flat.reshape((num_elems_x, num_elems_y))

        return rho_optimum.astype(np.float32), opti_steps_history

    # --- render method remains the same as previous version ---
    def render(self, design: npt.NDArray, open_window: bool = False, config: dict[str, Any] = {}, **kwargs) -> plt.Figure:
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
            raise ValueError(f"Input design shape {design.shape} != ({num_elems_x}, {num_elems_y})")

        # Run simulation for the given design to get fields for plotting
        conditions["beta"] = conditions.get("beta", self._beta_default)
        # Use run_fdfd but ignore most outputs, just need epsr, Ez1, Ez2
        epsr, Ez1, Ez2, _, _, _, _ = self._run_fdfd(design, conditions) 

        # Store these fields as the "last" simulated ones as well
        self._last_epsr = epsr.copy()
        self._last_Ez1 = Ez1.copy()
        self._last_Ez2 = Ez2.copy()

        # --- Plotting (same as before) ---
        fig, ax = plt.subplots(1, 3, constrained_layout=True, figsize=(9, 3))
        ceviche.viz.abs(Ez1, outline=epsr, ax=ax[0], cbar=False, cmap='magma', outline_alpha=0.9, outline_val=np.sqrt(self.epsr_min/self.epsr_max))
        ceviche.viz.abs(Ez2, outline=epsr, ax=ax[1], cbar=False, cmap='magma', outline_alpha=0.9, outline_val=np.sqrt(self.epsr_min/self.epsr_max))
        ceviche.viz.real(epsr, ax=ax[2], cmap='Greys')
        slices_to_plot = [self._input_slice, self._output_slice1, self._output_slice2]
        for sl in slices_to_plot:
            if sl:
                for axis in ax[:2]: # Plot on field plots
                    axis.plot(sl.x * np.ones(len(sl.y)), sl.y, 'w-', alpha=0.5, linewidth=1)
        omega1 = conditions["omega1"]
        omega2 = conditions["omega2"]
        lambda1_um = 299792458 / (omega1 / (2 * np.pi)) / 1e-6
        lambda2_um = 299792458 / (omega2 / (2 * np.pi)) / 1e-6
        ax[0].set_title(f'|Ez| at $\\lambda_1$ = {lambda1_um:.2f} $\\mu$m')
        ax[1].set_title(f'|Ez| at $\\lambda_2$ = {lambda2_um:.2f} $\\mu$m')
        ax[2].set_title(f'Permittivity $\epsilon_r$')
        for axis in ax:
            axis.set_xlabel('')
            axis.set_ylabel('')
            axis.set_xticks([])
            axis.set_yticks([])

        plt.tight_layout()
        if open_window:
            plt.show(block=False)
        return fig

    def random_design(self) -> tuple[npt.NDArray, int]:
        """Generates a starting design with small random variations.

           Creates a design that is 0.5 within the design region, plus small
           uniform random noise (0.001 * rand). Returns 0 as the index placeholder.

        Returns:
            tuple[npt.NDArray, int]: The starting design array (rho) and an integer (0).
        """
        current_conditions = dict(self.conditions)
        num_elems_x = current_conditions.get("num_elems_x", self._num_elems_x_default)
        num_elems_y = current_conditions.get("num_elems_y", self._num_elems_y_default)
        space = current_conditions.get("space", self._space_default)
        num_elems_pml = self.num_elems_pml

        design_region = np.zeros((num_elems_x, num_elems_y))
        design_region[num_elems_pml + space:num_elems_x - num_elems_pml - space, num_elems_pml + space:num_elems_y - num_elems_pml - space] = 1

        # Use randomized initialization
        # TODO: Will below ever happen?
        # Ensure np_random is initialized
        if self.np_random is None:
            self.reset()
        # Generate random numbers using the problem's RNG
        random_noise = 0.001 * self.np_random.random((num_elems_x, num_elems_y))
        rho_start = design_region * (0.5 + random_noise)

        return rho_start.astype(np.float32), 0


# --- Example Usage (main block) ---
if __name__ == "__main__":
    problem = PhotonicMultiplexer2D()
    problem.reset(seed=42) # Use a seed

    start_design, _ = problem.random_design()
    fig_start = problem.render(start_design)

    # Simulation Example
    print("Simulating starting design...")
    # Simulate returns the raw objective = penalty - overlap1*overlap2
    obj_start_raw = problem.simulate(start_design)
    print(f"Starting Raw Objective ({problem.objectives[0][0]}): {obj_start_raw[0]:.4f}")

    # Optimization Example
    opt_config = {
        "num_optimization_steps": 20,
        "step_size": 1e-1,
        "penalty_weight": 1e-1,
        "save_frame_interval": 2
    }
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
