# ruff: noqa: N806
# Disabled variable name conventions

"""Beams 2D problem."""

from copy import deepcopy
import dataclasses
from typing import Any

import numpy as np
import numpy.typing as npt

from engibench.problems.beams2d.backend import calc_sensitivity
from engibench.problems.beams2d.backend import design_to_image
from engibench.problems.beams2d.backend import image_to_design
from engibench.problems.beams2d.backend import inner_opt
from engibench.problems.beams2d.backend import overhang_filter_d
from engibench.problems.beams2d.backend import overhang_filter_x
from engibench.problems.beams2d.backend import State
from engibench.problems.beams2d.v0 import Beams2D as Beams2D_v0
from engibench.problems.beams2d.v0 import ExtendedOptiStep
from engibench.problems.beams2d.v0 import main
from engibench.utils.upcast import upcast


class Beams2D(Beams2D_v0):
    r"""Beam 2D topology optimization problem - Version 1 (v1).

    ## v1
    This version augments v0 by fixing a minor detail in the v0 warm-start optimization process.
    Specifically, when warm-starting from a provided design, a small epsilon value is added to
    avoid zero-density values that could lead to gradient issues. The datasets themselves remain unchanged.

    All other behavior is identical to v0.
    """

    version = 1

    def optimize(
        self, starting_point: npt.NDArray | None = None, config: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, list[ExtendedOptiStep]]:
        """Optimizes the design of a beam.

        Args:
            starting_point (npt.NDArray or None): The design to begin warm-start optimization from (optional).
            config (dict): A dictionary with configuration (e.g., boundary conditions) for the optimization.

        Returns:
            Tuple[np.ndarray, dict]: The optimized design and its performance.
        """
        base_config = self.Config(**{**dataclasses.asdict(self.simulate_config), **(config or {})})

        self.__st = State.new(base_config.nelx, base_config.nely, base_config.rmin, base_config.forcedist)

        # Returns the full history of the optimization instead of just the last step
        optisteps_history = []

        if starting_point is None:
            xPhys = base_config.volfrac * np.ones((base_config.nelx, base_config.nely), dtype=float)
            x = xPhys.ravel()
        else:
            starting_point = image_to_design(starting_point)
            assert starting_point is not None
            eps = 1e-4
            x = (
                (1 - eps) * starting_point + eps * base_config.volfrac
            )  # add tiny non-zero values to avoid warm-start gradient issues for zero values
            xPhys = x.reshape((base_config.nelx, base_config.nely))

        xPrint = overhang_filter_x(xPhys) if base_config.overhang_constraint else xPhys.ravel()
        loop, change = (0, 1.0)

        while change > self.__st.min_change and loop < base_config.max_iter:
            ce = calc_sensitivity(xPrint, st=self.__st, cfg=dataclasses.asdict(base_config))
            simulate_config = upcast(base_config)
            self.reset_called = True  # override for multiple reset calls in optimize
            c = self.simulate(xPrint, ce=ce, config=dataclasses.asdict(simulate_config))

            # The design this step was evaluated at, taken before the overhang
            # filter below rebinds xPrint to the next one.
            design = np.array(xPrint)

            dc = (-base_config.penal * xPrint ** (base_config.penal - 1) * (self.__st.Emax - self.__st.Emin)) * ce
            dv = np.ones(base_config.nely * base_config.nelx)
            # MATLAB implementation:
            if base_config.overhang_constraint:
                xPrint, dc, dv = overhang_filter_d(xPhys, dc, dv)
            else:
                xPrint = xPhys.ravel()

            dc = np.asarray(self.__st.H * (dc[np.newaxis].T / self.__st.Hs))[:, 0]
            dv = np.asarray(self.__st.H * (dv[np.newaxis].T / self.__st.Hs))[:, 0]
            # Ensure dc remains nonpositive
            dc = np.clip(dc, None, 0.0)

            xnew, xPhys, xPrint = inner_opt(x, self.__st, dc, dv, dataclasses.asdict(base_config))
            # Compute the change by the inf. norm
            change = np.linalg.norm(
                xnew.reshape(base_config.nelx * base_config.nely, 1) - x.reshape(base_config.nelx * base_config.nely, 1),
                np.inf,
            )

            # Record the current state in optisteps_history, now that the move
            # this design led to is known.
            #
            # x_sensitivities is the filtered objective sensitivity alone, one
            # value per design variable, so that it has the same shape as the
            # design it belongs to. The volume sensitivity dv is deliberately not
            # stacked alongside it: consumers flatten this field, so an extra
            # channel doubles its length with nothing recording that it did, and
            # photonics2d -- the other 2D problem that reports sensitivities --
            # reports the objective gradient by itself.
            #
            # The move is measured in printed density, the space the recorded
            # design is in, so that design + update is the next step's design;
            # inner_opt also returns the raw density field, and differencing that
            # instead would mix the two spaces. The objective delta needs the
            # *next* step's objective, so it is filled in on the following pass,
            # and the last step keeps None rather than costing an extra solve.
            obj_values = np.array(c)
            if optisteps_history:
                previous = optisteps_history[-1]
                previous.obj_values_update = obj_values - previous.obj_values
            current_step = ExtendedOptiStep(
                obj_values=obj_values,
                step=loop,
                x=design,
                x_sensitivities=dc.copy(),
                x_update=xPrint - design,
            )
            current_step.design = design
            optisteps_history.append(current_step)

            loop += 1

            x = deepcopy(xnew)

        return design_to_image(xPrint, base_config.nelx, base_config.nely), optisteps_history


if __name__ == "__main__":
    main(Beams2D)
