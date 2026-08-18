"""This module contains the Python implementation of the thermoelastic 2D problem."""

from math import ceil
from math import hypot
import time
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from engibench.core import OptiStep
from engibench.problems.thermoelastic2d.model.fem_matrix_builder import fe_melthm
from engibench.problems.thermoelastic2d.model.fem_setup import fe_mthm_bc
from engibench.problems.thermoelastic2d.model.mma_subroutine import MMAInputs
from engibench.problems.thermoelastic2d.model.mma_subroutine import mmasub
from engibench.problems.thermoelastic2d.model.opti_dataclass import OptiStepUpdate
from engibench.problems.thermoelastic2d.utils import get_res_bounds
from engibench.problems.thermoelastic2d.utils import plot_multi_physics

SECOND_ITERATION_THRESHOLD = 2
FIRST_ITERATION_THRESHOLD = 1
MIN_ITERATIONS = 10
MAX_ITERATIONS = 120
UPDATE_THRESHOLD = 0.01


class FeaModel:
    """Finite Element Analysis (FEA) model for coupled 2D thermoelastic topology optimization."""

    def __init__(self, *, plot: bool = False, eval_only: bool | None = False, max_iter: int = MAX_ITERATIONS) -> None:
        """Instantiates a new model for the thermoelastic 2D problem.

        Args:
            plot: (bool, optional): If True, the updated design will be plotted at each iteration.
            eval_only: (bool, optional): If True, the model will only evaluate the design and return the objective values.
            max_iter: Maximal number of iterations for the `run` method.
        """
        self.plot = plot
        self.eval_only = eval_only
        self.max_iter = max_iter

    def get_initial_design(self, volume_fraction: float, nelx: int, nely: int) -> np.ndarray:
        """Generates the initial design variable field for the optimization process.

        Args:
            volume_fraction (float): The initial volume fraction for the material distribution.
            nelx (int): Number of elements in the x-direction.
            nely (int): Number of elements in the y-direction.

        Returns:
            np.ndarray: A 2D NumPy array of shape (nely, nelx) initialized with the given volume fraction.
        """
        return volume_fraction * np.ones((nely, nelx))

    def get_matricies(self, nu: float, e: float, k: float, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes and returns the element matrices required for the structural-thermal analysis.

        Args:
            nu (float): Poisson's ratio.
            e (float): Young's modulus (modulus of elasticity).
            k (float): Thermal conductivity.
            alpha (float): Coefficient of thermal expansion.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - The stiffness matrix for mechanical analysis.
                - The thermal stiffness matrix.
                - The coupling matrix for thermal expansion effects.
        """
        return fe_melthm(nu, e, k, alpha)

    def get_filter(self, nelx: int, nely: int, rmin: float) -> tuple[coo_matrix, np.ndarray]:
        """Constructs a sensitivity filtering matrix to smoothen the design variables.

        The filter helps mitigate checkerboarding issues in topology optimization by averaging
        sensitivities over neighboring elements.

        Args:
            nelx (int): Number of elements in the x-direction.
            nely (int): Number of elements in the y-direction.
            rmin (float): Minimum filter radius.

        Returns:
            Tuple[csr_matrix, np.ndarray]: A tuple containing:
                - `h` (csr_matrix): A sparse matrix that represents the filtering operation.
                - `hs` (np.ndarray): A normalization factor for the filtering.
        """
        i_h = []
        j_h = []
        s_h = []

        for i1 in range(nelx):
            i2_min = max(i1 - (ceil(rmin) - 1), 0)
            for j1 in range(nely):
                e1 = i1 * nely + j1
                i2_max = min(i1 + (ceil(rmin) - 1), nelx - 1)
                for i2 in range(i2_min, i2_max + 1):
                    j2_min = max(j1 - (ceil(rmin) - 1), 0)
                    j2_max = min(j1 + (ceil(rmin) - 1), nely - 1)
                    for j2 in range(j2_min, j2_max + 1):
                        e2 = i2 * nely + j2
                        i_h.append(e1)
                        j_h.append(e2)
                        s_h.append(max(0, rmin - hypot(i1 - i2, j1 - j2)))

        h = coo_matrix((s_h, (i_h, j_h)), shape=(nelx * nely, nelx * nely)).tocsr()
        hs = np.array(h.sum(axis=1)).flatten()

        return h, hs

    def has_converged(self, change: float, iterr: int) -> bool:
        """Determines whether the optimization process has converged based on the change in design variables and iteration count.

        Args:
            change (float): The maximum change in design variables from the previous iteration.
            iterr (int): The current iteration number.

        Returns:
            bool: True if the optimization has converged, False otherwise. Convergence is defined as either:
                - The change in design variables is below a predefined threshold and a minimum number of iterations have been completed.
                - The maximum number of iterations has been reached.
        """
        if iterr >= self.max_iter:
            return True
        return change < UPDATE_THRESHOLD and iterr >= MIN_ITERATIONS

    def record_step(self, opti_steps: list[OptiStep], opti_step_update: OptiStepUpdate):
        """Helper to handle OptiStep creation and updates.

        Args:
            opti_steps (list): The list of OptiStep objects to be updated in place with the new step information.
            opti_step_update (OptiStepUpdate): A dataclass encapsulating all input parameters.

        Returns:
            None. This function updates the opti_steps list in place.
        """
        obj_values = opti_step_update.obj_values
        iterr = opti_step_update.iterr
        x_curr = opti_step_update.x_curr
        x_sensitivities = opti_step_update.x_sensitivities
        x_update = opti_step_update.x_update
        extra_iter = opti_step_update.extra_iter
        if extra_iter is False:
            step = OptiStep(obj_values=obj_values, step=iterr, x=x_curr, x_sensitivities=x_sensitivities, x_update=x_update)
            opti_steps.append(step)

        if len(opti_steps) > 1:
            # Targeting the most recent step to update its 'delta'
            target_idx = -2 if not extra_iter else -1
            opti_steps[target_idx].obj_values_update = obj_values.copy() - opti_steps[target_idx].obj_values

    def run(self, bcs: dict[str, Any], x_init: np.ndarray | None = None) -> dict[str, Any]:  # noqa: PLR0915
        """Run the optimization algorithm for the coupled structural-thermal problem.

        This method performs an iterative optimization procedure that adjusts the design
        variables for a coupled structural-thermal problem until convergence criteria are met.
        The algorithm utilizes finite element analysis (FEA), sensitivity analysis, and the Method
        of Moving Asymptotes (MMA) to update the design.

        Args:
            bcs (dict[str, any]): A dictionary containing boundary conditions and problem parameters.
                Expected keys include:
                    - 'volume_fraction_target' (float): Target volume fraction.
                    - 'fixed_elements' (np.ndarray): NxN binary array encoding the location of fixed elements.
                    - 'force_elements_x' (np.ndarray): NxN binary array encoding the location of loaded elements in the x direction.
                    - 'force_elements_y' (np.ndarray): NxN binary array encoding the location of loaded elements in the y direction.
                    - 'heatsink_elements' (np.ndarray): NxN binary array encoding the location of heatsink elements.
                    - 'weight' (float, optional): Weighting factor between structural and thermal objectives.
            x_init (Optional[np.ndarray]): Initial design variable array. If None, the design is generated
                using the get_initial_design method.

        Returns:
            Dict[str, Any]: A dictionary containing the optimization results. The dictionary includes:
                - 'design' (np.ndarray): Final design layout.
                - 'bcs' (Dict[str, Any]): The input boundary conditions.
                - 'sc' (float): Structural cost component.
                - 'tc' (float): Thermal cost component.
                - 'vf' (float): Volume fraction error.
            If self.eval_only is True, returns a dictionary with keys 'sc', 'tc', and 'vf' only.
        """
        # WEIGHTING
        w1 = bcs.get("weight", 0.5)
        w2 = 1.0 - w1

        fixed_elements = bcs["fixed_elements"]
        fe_h, fe_w = fixed_elements.shape
        nelx = fe_h - 1
        nely = fe_w - 1

        volfrac = bcs["volume_fraction_target"]
        n = nely * nelx  # Total number of elements

        # OptiSteps records
        opti_steps: list[OptiStep] = []

        # 1. Initial Design
        x = self.get_initial_design(volfrac, nelx, nely) if x_init is None else x_init

        # 2. Parameters
        penal = 3  # Penalty term
        rmin = 1.5  # Filter's radius
        e = 1.0  # Modulus of elasticity
        nu = 0.3  # Poisson's ratio
        k = 1.0  # Conductivity
        alpha = 5e-4  # Coefficient of thermal expansion (CTE)
        tref = 9.267e-4  # Reference Temperature
        change = 1.0  # Density change criterion
        m = 1  # Number of constraints (volume constraints)
        iterr = 0  # Number of iterations
        xmin = 1e-3  # Densities' Lower bound
        xmax = 1.0  # Densities' Upper bound
        low = xmin
        upp = xmax
        xold1 = x.reshape(n, 1)
        xold2 = x.reshape(n, 1)
        a0 = 1
        a = np.zeros((m, 1))
        c = 10000 * np.ones((m, 1))
        d = np.zeros((m, 1))

        # Convergence / Iteration Criteria
        extra_iter = False  # This flag denotes if we are on the final extra iteration (for purpose of gathering gradient information)

        # 3. Matrices
        ke, k_eth, c_ethm = self.get_matricies(nu, e, k, alpha)

        # 4. Filter
        h, hs = self.get_filter(nelx, nely, rmin)

        # 5. Optimization Loop
        change_evol = []
        obj = []

        f0valm = 0.0
        f0valt = 0.0

        while not self.has_converged(change, iterr) or extra_iter is True:
            iterr += 1
            s_time = time.time()
            curr_time = time.time()

            # FE-ANALYSIS
            results = fe_mthm_bc(nely, nelx, penal=penal, x=x, ke=ke, k_eth=k_eth, c_ethm=c_ethm, tref=tref, bcs=bcs)
            km = results.km
            kth = results.kth
            um = results.um
            uth = results.uth
            fm = results.fm
            fth = results.fth
            d_cthm = results.d_cthm
            fixeddofsm = results.fixeddofsm
            freedofsm = results.freedofsm
            fixeddofsth = results.fixeddofsth
            fp = results.fp

            t_forward = time.time() - curr_time
            curr_time = time.time()

            # Plot design update
            if self.plot is True:
                plot_multi_physics(x, fixeddofsm, fixeddofsth, nelx, nely, fp, um, uth, open_plot=True)

            ndofm = 2 * (nely + 1) * (nelx + 1)

            # Flatten the force matrix once
            flam = fm.flatten()

            # --- Solve for free degrees of freedom ---
            km_ff = km[freedofsm, :][:, freedofsm].tocsc()
            flam_freedofs = flam[freedofsm]
            lamm_freedofs = -spsolve(km_ff, flam_freedofs)
            lamm = np.zeros(ndofm)
            lamm[freedofsm] = lamm_freedofs

            # --- Sensitivity analysis or second solve ---
            temp = lamm.T @ d_cthm - um.T @ d_cthm - fth.T
            temp = np.asarray(temp).flatten()
            kth_sparse = kth.tocsc()
            lamth = spsolve(kth_sparse, temp)

            # ------- END OPTIMIZED CODE ------- #

            # PREPARE SENSITIVITY ANALYSIS
            t_sensitivity = time.time() - curr_time
            curr_time = time.time()
            f0val = 0
            f0valm = 0
            f0valt = 0
            df0dx_mat = np.zeros((nely, nelx))

            df0dx_m = np.zeros((nely, nelx))
            df0dx_t = np.zeros((nely, nelx))

            xval = x.reshape(n, 1)
            # DEFINE CONSTRAINTS
            volconst = np.sum(x) / (volfrac * n) - 1  # shape ()
            fval = volconst  # Column vector of size (1xm)
            dfdx = np.ones((1, n)) / (volfrac * n)  # shape (1, n)

            # CALCULATE SENSITIVITIES
            for elx in range(nelx):
                for ely in range(nely):
                    n1 = (nely + 1) * elx + ely
                    n2 = (nely + 1) * (elx + 1) + ely
                    edof4 = np.array([n1 + 1, n2 + 1, n2, n1], dtype=int)
                    edof8 = np.array(
                        [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1], dtype=int
                    )

                    ume = um[edof8].flatten()
                    uthe = uth[edof4].flatten()
                    lamm[edof8].flatten()
                    lamthe = lamth[edof4].flatten()
                    x_p = x[ely, elx] ** penal
                    x_p_minus1 = penal * x[ely, elx] ** (penal - 1)

                    # Separate
                    f0valm += x_p * ume.T @ ke @ ume
                    f0valt += x_p * uthe.T @ k_eth @ uthe

                    df0dx_m[ely, elx] = -x_p_minus1 * ume.T @ ke @ ume
                    df0dx_t[ely, elx] = lamthe.T @ (x_p_minus1 * k_eth @ uthe)
                    df0dx_mat[ely, elx] = (df0dx_m[ely, elx] * w1) + (df0dx_t[ely, elx] * w2)  # Weighted sensitivity

            f0val = (f0valm * w1) + (f0valt * w2)

            if self.eval_only is True:
                vf_error = np.abs(np.mean(x) - volfrac)
                return {
                    "structural_compliance": f0valm,
                    "thermal_compliance": f0valt,
                    "volume_fraction_error": vf_error,
                }

            # OptiStep Information
            vf_error = np.abs(np.mean(x) - volfrac)
            obj_values = np.array([f0valm, f0valt, vf_error])
            x_curr = x.copy()  # Design variables before the gradient update (nely, nelx)

            df0dx = df0dx_mat.reshape(nely * nelx, 1)
            df0dx = (h @ (xval * df0dx)) / hs[:, None] / np.maximum(1e-3, xval)  # Filtered sensitivity

            t_sensitivity_calc = time.time() - curr_time
            curr_time = time.time()

            # UPDATE DESIGN VARIABLES USING MMA
            upp_vec = np.ones((n,)) * upp
            low_vec = np.ones((n,)) * low
            mmainputs = MMAInputs(
                m=m,
                n=n,
                iterr=iterr,
                xval=xval[:, 0],  # selecting appropriate column
                xmin=xmin,
                xmax=xmax,
                xold1=xold1,
                xold2=xold2,
                df0dx=df0dx[:, 0],  # selecting appropriate column
                fval=fval,  # Constraint values
                dfdx=dfdx,
                low=low_vec,
                upp=upp_vec,
                a0=a0,
                a=a[0],
                c=c[0],
                d=d[0],
                f0val=f0val,
            )
            xmma = mmasub(mmainputs)
            t_mma = time.time() - curr_time

            # Store previous density fields
            if iterr > SECOND_ITERATION_THRESHOLD:
                xold2 = xold1
                xold1 = xval
            elif iterr > FIRST_ITERATION_THRESHOLD:
                xold1 = xval

            x = xmma.reshape(nely, nelx)

            # Extract the exact gradient update step for OptiStep
            x_update = x.copy() - x_curr

            # Record the OptiStep
            df0dx_all = np.stack([df0dx_m, df0dx_t, df0dx_mat, dfdx.reshape(nely, nelx)], axis=0)
            opti_step_update = OptiStepUpdate(
                obj_values=obj_values,
                iterr=iterr,
                x_curr=x_curr,
                x_sensitivities=df0dx_all,
                x_update=x_update,
                extra_iter=extra_iter,
            )
            self.record_step(opti_steps, opti_step_update)

            # Print results
            change = np.max(np.abs(xmma - xold1))
            change_evol.append(change)
            obj.append(f0val)
            t_total = time.time() - s_time
            print(
                f" It.: {iterr:4d} Obj.: {f0val:10.4f} Vol.: {np.sum(x) / (nelx * nely):6.3f} ch.: {change:6.3f} || t_forward:{t_forward:6.3f} + t_sensitivity:{t_sensitivity:6.3f} + t_sens_calc:{t_sensitivity_calc:6.3f} + t_mma: {t_mma:6.3f} = {t_total:6.3f}"
            )

            # If extra_iter is True, we just did our last iteration and want to break
            if extra_iter is True:
                x = xold1.reshape(nely, nelx)  # Revert to design before the last update (for accurate gradient information)
                break  # We technically don't have to break here, as the logic is built into the loop condition

            # We know we are not on the extra iteration
            # Check to see if we have converged. If so, flag our extra iteration
            if self.has_converged(change, iterr):
                extra_iter = True

        print("Optimization finished...")
        vf_error = np.abs(np.mean(x) - volfrac)

        return {
            "design": x,
            "bcs": bcs,
            "structural_compliance": f0valm,
            "thermal_compliance": f0valt,
            "volume_fraction_error": vf_error,
            "opti_steps": opti_steps,
        }


if __name__ == "__main__":
    nelx = 64
    nely = 64

    client = FeaModel(plot=True)

    lci, tri, rci, bri = get_res_bounds(nelx + 1, nely + 1)

    bcs = {
        "nelx": nelx,
        "nely": nely,
        "fixed_elements": [lci[21], lci[32], lci[43]],
        "force_elements_y": [bri[31]],
        "heatsink_elements": [lci[31], lci[32], lci[33]],
        "volume_fraction_target": 0.2,
        "rmin": 1.1,
        "weight": 1.0,  # 1.0 for pure structural, 0.0 for pure thermal
    }

    result = client.run(bcs)
