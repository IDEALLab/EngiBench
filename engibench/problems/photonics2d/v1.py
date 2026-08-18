"""Photonics2D problem - Version 1 (v1).

## v1

v1 makes ``simulate`` and ``optimize`` return consistent values for the same design.

Background: a design is a continuous density field ``rho`` (values in [0, 1]) that is mapped to a
permittivity image and scored with an FDFD solve. During optimization two operations shape ``rho``:
a *blur* (which enforces a minimum feature size) and a *projection* (a tanh "soft step" that pushes
values toward 0 or 1 so the final design is nearly binary; its sharpness is controlled by ``beta``,
which is increased over the run -- a schedule known as "continuation").

In v0 the blur and projection were applied inside **both** ``simulate`` and ``optimize``, at
different strengths, so an optimized design did not reproduce its reported score when passed back to
``simulate``. v1 confines blur and projection to ``optimize`` and evaluates designs as-is everywhere
else:

* A *design* is a physical density ``rho`` in [0, 1] (already blurred/projected).
* ``simulate(design)`` checks ``rho`` is in [0, 1] (via :meth:`check_constraints`) and maps it to
  permittivity by scaling only (:func:`design_to_epsr`) -- no blur, no projection -- then runs FDFD.
* ``optimize(starting_point)`` records step 0 as the raw starting point (so it equals
  ``simulate(starting_point)``), then for steps 1..N applies blur + projection with ``beta`` ramping
  from ``initial_beta`` to ``max_beta`` inside an explicit Adam loop (clipping ``rho`` to [0, 1]
  each step). It returns the projected density at ``max_beta``, so ``simulate(optimize(x)[0])``
  reproduces ``history[-1]``.

The projection (:func:`operator_proj`) is the standard tanh operator, which already maps
[0, 1] -> [0, 1]; keeping ``rho`` clipped to [0, 1] is why a sigmoid reparameterization is not needed.

Based on the v0 implementation by Mark Fuge @markfuge.
"""

import dataclasses
from dataclasses import dataclass
import os
import pprint
from typing import Annotated, Any

import autograd.numpy as npa
import ceviche
from ceviche import fdfd_ez
from ceviche import jacobian
from gymnasium import spaces
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from engibench.constraint import bounded
from engibench.constraint import Criticality
from engibench.constraint import IMPL
from engibench.constraint import THEORY
from engibench.core import OptiStep
from engibench.core import SimulationResult
from engibench.problems.photonics2d.backend import design_to_epsr
from engibench.problems.photonics2d.backend import filter_and_project
from engibench.problems.photonics2d.backend import insert_mode
from engibench.problems.photonics2d.backend import mode_overlap
from engibench.problems.photonics2d.backend import poly_ramp
from engibench.problems.photonics2d.v0 import Photonics2D as Photonics2D_v0


class Photonics2D(Photonics2D_v0):
    r"""Photonic Inverse Design 2D Problem (Wavelength Demultiplexer) - v1.

    See the module docstring for the v1 ``simulate`` / ``optimize`` contract. All physical
    constants and geometry are inherited from v0; v1 overrides only the evaluation /
    optimization logic and promotes the optimizer knobs to validated ``Config`` fields.
    """

    version = 1

    @dataclass
    class Config(Photonics2D_v0.Config):
        """v1 configuration: the v0 fields plus the optimizer knobs as validated ``Config`` fields.

        These are declared on ``Config`` only -- never on ``Conditions`` -- because ``Conditions``
        fields must exist as dataset columns. They control optimization only and do not change the
        meaning of a stored design.
        """

        penalty_weight: Annotated[float, bounded(lower=0.0).category(THEORY | IMPL)] = 1e-3
        """Weight of the material-usage penalty added to the objective."""
        step_size: Annotated[float, bounded(lower=0.0).category(IMPL)] = 1e-1
        """Adam step size."""
        eta: Annotated[float, bounded(lower=0.0, upper=1.0).category(IMPL)] = 0.5
        """Projection threshold (center) for the tanh Heaviside."""
        initial_beta: Annotated[float, bounded(lower=0.0).category(IMPL)] = 1.0
        """Starting projection sharpness for the beta continuation."""
        max_beta: Annotated[float, bounded(lower=1.0).category(IMPL)] = 300.0
        """Final projection sharpness reached at the last optimization step."""

    # float64 design space: a contains-check on a float32 box would reject ordinary float64
    # designs, so use the more permissive float64 (it still rejects out-of-[0,1] designs).
    design_space = spaces.Box(low=0.0, high=1.0, shape=(Config.num_elems_x, Config.num_elems_y), dtype=np.float64)
    dataset_id = f"IDEALLab/photonics_2d_{Config.num_elems_x}_{Config.num_elems_y}_v1"
    config: Config

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize like v0, then widen the instance design space to float64."""
        super().__init__(*args, **kwargs)
        self.design_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_elems_x, self.num_elems_y), dtype=np.float64)

    # ------------------------------------------------------------------ helpers

    def _check_design(self, design: npt.NDArray, config: dict[str, Any] | None) -> None:
        """Validate ``design`` (and any overridden config) against the declared constraints.

        Raises ``ValueError`` on any Error-level violation (e.g. ``rho`` outside [0, 1]). Only
        keys that are real ``Config`` fields are forwarded, so runtime-only knobs such as
        ``save_frame_interval`` do not trip the dataclass construction inside ``check_constraints``.
        """
        valid_keys = {f.name for f in dataclasses.fields(self.Config)}
        cfg = {k: v for k, v in (config or {}).items() if k in valid_keys}
        violations = self.check_constraints(design, cfg)
        errors = violations.by_criticality(Criticality.Error)
        if errors:
            raise ValueError(f"Design/config violates constraints:\n{errors}")

    def _run_fdfd(  # type: ignore[override]
        self, epsr: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """Run both FDFD solves for a *given* permittivity field (no parameterization here).

        Sources and probes live in the fixed waveguide background, so they are rebuilt from
        ``epsr`` consistently regardless of the design.

        Returns:
            (ez1, ez2, source1, source2, probe1, probe2).
        """
        omega1, omega2 = self.omega1, self.omega2
        source1 = insert_mode(omega1, self._dl, self._input_slice.x, self._input_slice.y, epsr, m=1)
        source2 = insert_mode(omega2, self._dl, self._input_slice.x, self._input_slice.y, epsr, m=1)
        probe1 = insert_mode(omega1, self._dl, self._output_slice1.x, self._output_slice1.y, epsr, m=1)
        probe2 = insert_mode(omega2, self._dl, self._output_slice2.x, self._output_slice2.y, epsr, m=1)

        self._simulation1 = fdfd_ez(omega1, self._dl, epsr, [self._num_elems_pml, self._num_elems_pml])
        self._simulation2 = fdfd_ez(omega2, self._dl, epsr, [self._num_elems_pml, self._num_elems_pml])
        _, _, ez1 = self._simulation1.solve(source1)
        _, _, ez2 = self._simulation2.solve(source2)
        return ez1, ez2, source1, source2, probe1, probe2

    def _objective_from_fields(
        self,
        *,
        ez1: Any,
        ez2: Any,
        probe1: npt.NDArray,
        probe2: npt.NDArray,
        rho_phys: Any,
        penalty_weight: float,
    ) -> Any:
        """Compute the objective ``total_overlap - penalty`` shared by ``simulate`` and ``optimize``.

        Both paths call this one function so they cannot drift apart numerically. It is written with
        ``autograd.numpy`` so it also works on the traced values that flow through the optimizer's
        gradient (hence the ``Any`` field types). The penalty uses the *physical* density, which is
        what makes ``simulate(optimize(x)[0]) == history[-1]`` hold.
        """
        overlap1 = mode_overlap(ez1, probe1)
        overlap2 = mode_overlap(ez2, probe2)
        total_overlap = overlap1 * overlap2
        penalty = penalty_weight * npa.linalg.norm(rho_phys)
        return total_overlap - penalty  # value to MAXIMIZE

    # ------------------------------------------------------------------ simulate

    def simulate_verbose(
        self, design: npt.NDArray, config: dict[str, Any] | None = None, **kwargs: Any
    ) -> SimulationResult:
        """Simulate a design **as-is** (scale -> permittivity, no blur, no projection).

        Args:
            design: The density field ``rho`` (shape num_elems_x, num_elems_y), expected in [0, 1].
            config: Optional overrides for the conditions (e.g. ``lambda1``, ``penalty_weight``).
            **kwargs: Ignored.

        Returns:
            SimulationResult with a 1-element array ``[total_overlap - penalty]`` (higher is better).
        """
        del kwargs
        conditions = self._setup_simulation(config)
        self._check_design(design, config)

        penalty_weight = conditions["penalty_weight"]
        epsr = design_to_epsr(design, self._bg_rho, self._design_region, self._epsr_min, self._epsr_max)
        ez1, ez2, _, _, probe1, probe2 = self._run_fdfd(epsr)

        # Store the most recent fields for render().
        self._last_epsr = np.array(epsr).copy()
        self._last_Ez1 = ez1.copy()
        self._last_Ez2 = ez2.copy()

        obj = self._objective_from_fields(
            ez1=ez1, ez2=ez2, probe1=probe1, probe2=probe2, rho_phys=design, penalty_weight=penalty_weight
        )
        return SimulationResult(np.array([float(obj)], dtype=np.float64))

    # ------------------------------------------------------------------ optimize

    def optimize(  # noqa: PLR0915
        self, starting_point: npt.NDArray, config: dict[str, Any] | None = None, **kwargs: Any
    ) -> tuple[npt.NDArray, list[OptiStep]]:
        """Optimize ``rho`` from ``starting_point`` with an explicit Adam loop + beta continuation.

        Step 0 evaluates the raw starting point (no projection), so ``history[0]`` equals
        ``simulate(starting_point)``. Steps 1..N apply blur + tanh projection with ``beta`` ramping
        ``initial_beta`` -> ``max_beta``; ``rho`` is clipped to [0, 1] after every step. The returned
        design is the physical (projected) density at ``max_beta``, so feeding it back into
        ``simulate`` reproduces ``history[-1]``.

        Args:
            starting_point: The starting density ``rho`` (shape num_elems_x, num_elems_y).
            config: Optional overrides (``num_optimization_steps``, ``step_size``, ``penalty_weight``,
                ``initial_beta``, ``max_beta``, ``eta``, ``save_frame_interval``, ...).
            **kwargs: Ignored.

        Returns:
            (optimized physical design, list of OptiStep history). ``OptiStep.obj_values`` holds
            ``[total_overlap - penalty]``; ``step`` is the optimizer iteration (0 == starting point).
        """
        del kwargs
        conditions = self._setup_simulation(config)
        self._check_design(starting_point, config)

        print("Optimizing Photonics2D (v1) under the following conditions:")
        pprint.pp(conditions)

        nx, ny = self.num_elems_x, self.num_elems_y
        n_steps = int(conditions["num_optimization_steps"])
        step_size = conditions["step_size"]
        eta = conditions["eta"]
        initial_beta = conditions["initial_beta"]
        max_beta = conditions["max_beta"]
        penalty_weight = conditions["penalty_weight"]
        blur_radius = conditions["blur_radius"]
        save_frame_interval = conditions.get("save_frame_interval", 0)

        bg_rho, design_region = self._bg_rho, self._design_region
        epsr_min, epsr_max = self._epsr_min, self._epsr_max

        history: list[OptiStep] = []

        # --- Step 0: run the starting point AS-IS (no projection) == simulate(starting_point) ---
        epsr0 = design_to_epsr(starting_point, bg_rho, design_region, epsr_min, epsr_max)
        ez1, ez2, source1, source2, probe1, probe2 = self._run_fdfd(epsr0)
        # Sources/probes live in the fixed waveguides; compute once and reuse for all steps.
        self._source1, self._source2 = source1, source2
        self._probe1, self._probe2 = probe1, probe2
        obj0 = float(
            self._objective_from_fields(
                ez1=ez1, ez2=ez2, probe1=probe1, probe2=probe2, rho_phys=starting_point, penalty_weight=penalty_weight
            )
        )
        history.append(OptiStep(obj_values=np.array([obj0], dtype=np.float64), step=0))

        if save_frame_interval and save_frame_interval > 0:
            os.makedirs("opt_frames", exist_ok=True)

        # --- Autograd objective for steps 1..N (reads the current beta from the closure) ---
        beta_state = {"beta": initial_beta}

        def objective(rho_flat: Any) -> Any:
            rho = rho_flat.reshape((nx, ny))
            rho_phys = filter_and_project(
                rho=rho, bg_rho=bg_rho, design_region=design_region, radius=blur_radius, beta=beta_state["beta"], eta=eta
            )
            epsr = epsr_min + (epsr_max - epsr_min) * rho_phys
            self._simulation1.eps_r = epsr
            self._simulation2.eps_r = epsr
            _, _, ez1 = self._simulation1.solve(self._source1)
            _, _, ez2 = self._simulation2.solve(self._source2)
            return self._objective_from_fields(
                ez1=ez1, ez2=ez2, probe1=self._probe1, probe2=self._probe2, rho_phys=rho_phys, penalty_weight=penalty_weight
            )

        objective_grad = jacobian(objective, mode="reverse")

        # --- Explicit Adam ascent (maximization), rho clipped to [0, 1] each step ---
        rho = starting_point.flatten().astype(np.float64)
        m = np.zeros_like(rho)
        v = np.zeros_like(rho)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        returned_design = starting_point.astype(self.design_space.dtype)

        for t in range(1, n_steps + 1):
            # Quadratic ramp from initial_beta (t=0) to max_beta (t=n_steps).
            beta_state["beta"] = poly_ramp(t, max_iter=n_steps, b0=initial_beta, bmax=max_beta, degree=2)

            # Physical design evaluated at this step (matches what `objective` uses internally).
            rho_phys = np.asarray(
                filter_and_project(
                    rho=rho.reshape((nx, ny)),
                    bg_rho=bg_rho,
                    design_region=design_region,
                    radius=blur_radius,
                    beta=beta_state["beta"],
                    eta=eta,
                )
            )
            obj_t = float(objective(rho))
            grad = np.asarray(objective_grad(rho)).flatten()

            history.append(
                OptiStep(
                    obj_values=np.array([obj_t], dtype=np.float64),
                    step=t,
                    x=rho_phys.copy(),
                    x_sensitivities=grad.reshape((nx, ny)).copy(),
                )
            )
            # The returned design is the physical density at the LAST evaluated point, so
            # simulate(returned_design) == history[-1].obj_values (last step uses beta = max_beta).
            returned_design = rho_phys.astype(self.design_space.dtype)

            # Adam update (ascent) followed by box projection onto [0, 1].
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad * grad)
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            rho = np.clip(rho + step_size * m_hat / (np.sqrt(v_hat) + eps), 0.0, 1.0)

            if save_frame_interval and save_frame_interval > 0 and t % save_frame_interval == 0:
                fig = self.render(rho_phys, config=config, open_window=False)
                fig.savefig(os.path.join("opt_frames", f"frame_iter_{t:04d}.png"), dpi=200)
                plt.close(fig)
                print(f"Iter {t}: objective {obj_t:.3e} (saved frame)")

        return returned_design, history

    # ------------------------------------------------------------------ render

    def render(self, design: npt.NDArray, config: dict[str, Any] | None = None, *, open_window: bool = False) -> Figure:
        """Render the design (as-is) and the resulting E-field magnitudes.

        Like ``simulate``, this runs the design verbatim (scale -> permittivity, no projection)
        so the rendered structure matches the simulated objective.

        Args:
            design: The density ``rho`` to render.
            config: Optional condition overrides.
            open_window: If True, open a window with the rendered plot.

        Returns:
            A matplotlib Figure with |Ez| at omega1, |Ez| at omega2, and the permittivity.
        """
        conditions = self._setup_simulation(config)
        epsr = design_to_epsr(design, self._bg_rho, self._design_region, self._epsr_min, self._epsr_max)
        ez1, ez2, _, _, probe1, probe2 = self._run_fdfd(epsr)

        overlap1 = mode_overlap(ez1, probe1)
        overlap2 = mode_overlap(ez2, probe2)
        total_overlap = overlap1 * overlap2

        self._last_epsr = np.array(epsr).copy()
        self._last_Ez1 = ez1.copy()
        self._last_Ez2 = ez2.copy()

        fig, ax = plt.subplots(1, 3, constrained_layout=True, figsize=(9, 3))
        ceviche.viz.abs(ez1, outline=epsr, ax=ax[0], cbar=False, outline_alpha=0.25)
        ceviche.viz.abs(ez2, outline=epsr, ax=ax[1], cbar=False, outline_alpha=0.25)
        ceviche.viz.real(epsr, ax=ax[2], cmap="Greys")
        for sl in (self._input_slice, self._output_slice1, self._output_slice2):
            if sl:
                for axis in ax[:2]:
                    axis.plot(sl.x * np.ones(len(sl.y)), sl.y, "w-", alpha=0.5, linewidth=1)

        lambda1_um = conditions["lambda1"]
        lambda2_um = conditions["lambda2"]
        blur_radius = conditions["blur_radius"]
        fig.suptitle(
            f"Total Overlap: {total_overlap:.2f} "
            f"($\\lambda_1$={lambda1_um:.2f} $\\mu$m, $\\lambda_2$={lambda2_um:.2f} $\\mu$m, blur={blur_radius}, as-is)"
        )
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
