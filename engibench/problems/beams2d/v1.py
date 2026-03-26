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

            # Record the current state in optisteps_history
            current_step = ExtendedOptiStep(obj_values=np.array(c), step=loop)
            current_step.design = np.array(xPrint)
            optisteps_history.append(current_step)

            loop += 1

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
            change = np.linalg.norm(  # type: ignore[assignment]
                xnew.reshape(base_config.nelx * base_config.nely, 1) - x.reshape(base_config.nelx * base_config.nely, 1),
                np.inf,
            )
            x = deepcopy(xnew)

        return design_to_image(xPrint, base_config.nelx, base_config.nely), optisteps_history


if __name__ == "__main__":
    main(Beams2D)
