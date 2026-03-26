"""This module contains the Python implementation of the thermoelastic 3D problem."""

from math import ceil
import time
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix

from engibench.core import OptiStep
from engibench.problems.thermoelastic3d.model.fem_matrix_builder import fe_melthm_3d
from engibench.problems.thermoelastic3d.model.fem_plotting import plot_fem_3d
from engibench.problems.thermoelastic3d.model.fem_setup import fe_mthm_bc_3d
from engibench.problems.thermoelastic3d.model.linear_solver import solve_spd_with_amg
from engibench.problems.thermoelastic3d.model.mma_subroutine import MMAInputs
from engibench.problems.thermoelastic3d.model.mma_subroutine import mmasub

SECOND_ITERATION_THRESHOLD = 2
FIRST_ITERATION_THRESHOLD = 1
MIN_ITERATIONS = 10
MAX_ITERATIONS = 200
UPDATE_THRESHOLD = 0.01


class FeaModel3D:
    """Finite Element Analysis (FEA) model for coupled 3D thermoelastic topology optimization."""

    def __init__(self, *, plot: bool = False, eval_only: bool = False, max_iter: int = MAX_ITERATIONS) -> None:
        """Instantiates a new 3D thermoelastic model.

        Args:
            plot: If True, you can hook in your own plotting / volume rendering each iteration.
            eval_only: If True, evaluate the given design once and return objective components only.
            max_iter: Maximal number of iterations for the `run` method.
        """
        self.plot = plot
        self.eval_only = eval_only
        self.max_iter = max_iter

    def get_initial_design(self, volume_fraction: float, nelx: int, nely: int, nelz: int) -> np.ndarray:
        """Generates the initial design variable field for the optimization process.

        Args:
            volume_fraction (float): The initial volume fraction for the material distribution.
            nelx (int): Number of elements in the x-direction.
            nely (int): Number of elements in the y-direction.
            nelz (int): Number of elements in the z-direction.

        Returns:
            np.ndarray: A 3D NumPy array of shape (nely, nelx, nelz) initialized with the given volume fraction.
        """
        return volume_fraction * np.ones((nely, nelx, nelz), dtype=float)

    def get_matrices(self, nu: float, e: float, k: float, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        return fe_melthm_3d(nu, e, k, alpha)

    def get_filter(self, nelx: int, nely: int, nelz: int, rmin: float) -> tuple[coo_matrix, np.ndarray]:
        """Constructs a sensitivity filtering matrix to smoothen the design variables.

        The filter helps mitigate checkerboarding issues in topology optimization by averaging
        sensitivities over neighboring elements.

        Args:
            nelx (int): Number of elements in the x-direction.
            nely (int): Number of elements in the y-direction.
            nelz (int): Number of elements in the z-direction.
            rmin (float): Minimum filter radius.

        Returns:
            Tuple[csr_matrix, np.ndarray]: A tuple containing:
                - `h` (csr_matrix): A sparse matrix that represents the filtering operation.
                - `hs` (np.ndarray): A normalization factor for the filtering.
        """

        def e_index(ix: int, iy: int, iz: int) -> int:
            """Returns the global index of an element given its coordinates.

            Args:
                ix (int): The global x-index of an element.
                iy (int): The global y-index of an element.
                iz (int): The global z-index of an element.

            Returns:
                g_idx (int): The global index of an element.
            """
            return (nely * nelz) * ix + (nelz) * iy + iz

        i_h: list[int] = []
        j_h: list[int] = []
        s_h: list[float] = []

        rceil = ceil(rmin) - 1  # integer neighborhood radius

        for ix in range(nelx):
            ix_min = max(ix - rceil, 0)
            ix_max = min(ix + rceil, nelx - 1)
            for iy in range(nely):
                iy_min = max(iy - rceil, 0)
                iy_max = min(iy + rceil, nely - 1)
                for iz in range(nelz):
                    iz_min = max(iz - rceil, 0)
                    iz_max = min(iz + rceil, nelz - 1)
                    e1 = e_index(ix, iy, iz)
                    for jx in range(ix_min, ix_max + 1):
                        for jy in range(iy_min, iy_max + 1):
                            for jz in range(iz_min, iz_max + 1):
                                e2 = e_index(jx, jy, jz)
                                dx = ix - jx
                                dy = iy - jy
                                dz = iz - jz
                                dist = (dx * dx + dy * dy + dz * dz) ** 0.5
                                w = max(0.0, rmin - dist)
                                if w > 0.0:
                                    i_h.append(e1)
                                    j_h.append(e2)
                                    s_h.append(w)

        n = nelx * nely * nelz
        h = coo_matrix((s_h, (i_h, j_h)), shape=(n, n)).tocsr()
        hs = np.array(h.sum(axis=1)).ravel()
        return h, hs

    def run(self, bcs: dict[str, Any], x_init: np.ndarray | None = None) -> dict[str, Any]:  # noqa: PLR0915, C901
        """Run the optimization algorithm for the coupled structural-thermal problem.

        This method performs an iterative optimization procedure that adjusts the design
        variables for a coupled structural-thermal problem until convergence criteria are met.
        The algorithm utilizes finite element analysis (FEA), sensitivity analysis, and the Method
        of Moving Asymptotes (MMA) to update the design.

        Args:
            bcs (dict[str, any]): A dictionary containing boundary conditions and problem parameters.
                Expected keys include:
                    - 'volfrac' (float): Target volume fraction.
                    - 'fixed_elements' (np.ndarray): NxN binary array encoding the location of fixed elements.
                    - 'force_elements_x' (np.ndarray): NxN binary array encoding the location of loaded elements in the x direction.
                    - 'force_elements_y' (np.ndarray): NxN binary array encoding the location of loaded elements in the y direction.
                    - 'force_elements_z' (np.ndarray): NxN binary array encoding the location of loaded elements in the z direction.
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
        # Weighting
        w1 = bcs.get("weight", 0.5)  # structural
        w2 = 1.0 - w1  # thermal

        # Derive mesh sizes from fixed_elements mask (shape: (nelx+1, nely+1, nelz+1))
        fixed_nodes_mask = np.asarray(bcs["fixed_elements"], dtype=bool)

        nxp, nyp, nzp = fixed_nodes_mask.shape  # nodes-per-direction
        nelx, nely, nelz = nxp - 1, nyp - 1, nzp - 1
        n = nelx * nely * nelz  # number of elements

        volfrac = bcs["volfrac"]

        # OptiSteps records
        opti_steps = []

        # 1. Initial design
        x = self.get_initial_design(volfrac, nelx, nely, nelz) if x_init is None else x_init.copy()

        # 2. Parameters
        penal = bcs.get("penal", 3.0)  # SIMP Penalty
        rmin = bcs.get("rmin", 1.1)  # Minimum feature size
        e = 1.0  # Young's modulus
        nu = 0.3  # Poisson's ratio
        k = 1.0  # Thermal conductivity
        alpha = 5e-4  # Thermal strain
        tref = 9.267e-4  # Reference temperature
        change = 1.0
        iterr = 0
        xmin, xmax = 1e-3, 1.0
        xold1 = x.reshape(n, 1)
        xold2 = x.reshape(n, 1)
        m = 1  # volume constraint
        a0 = 1.0
        a = np.zeros((m, 1))
        c = 10000.0 * np.ones((m, 1))
        d = np.zeros((m, 1))
        low = xmin
        upp = xmax

        low_vec = None
        upp_vec = None

        # 3. Element matrices
        ke, k_eth, c_ethm = self.get_matrices(nu, e, k, alpha)

        # 4. 3D filter
        h, hs = self.get_filter(nelx, nely, nelz, rmin)

        # 5. Optimization Loop
        change_evol = []
        obj_evol = []

        f0valm = 0.0
        f0valt = 0.0

        while change > UPDATE_THRESHOLD or iterr < MIN_ITERATIONS:
            iterr += 1
            t0 = time.time()
            tcur = t0

            """
            ABAQUS HOOK: Solve the linear systems using the Abaqus solver
            """
            # Forward FEA with BCs & assembly (3D)
            res = fe_mthm_bc_3d(nely, nelx, nelz, penal, x, ke, k_eth, c_ethm, tref, bcs)

            kth = res.kth
            um = res.um
            uth = res.uth
            fth = res.fth
            d_cthm = res.d_cthm

            if self.plot is True and (iterr % 50 == 0):
                plot_fem_3d(bcs, x)

            t_forward = time.time() - tcur
            tcur = time.time()

            # Mechanical adjoint: self-adjoint, so the adjoint vector is simply the displacements
            lamm = -um

            # Thermal adjoint: K_th * lambda_th = (lamm^T - um^T) * d_cthm - f_th
            rhs_th = d_cthm.T @ (lamm - um) - fth
            lamth = solve_spd_with_amg(kth.tocsr(), rhs_th)

            t_adjoints = time.time() - tcur
            tcur = time.time()

            # Sensitivities & objective
            f0valm = 0.0
            f0valt = 0.0

            df0dx_m = np.zeros_like(x)  # (nely, nelx, nelz)
            df0dx_t = np.zeros_like(x)
            df0dx_mat = np.zeros_like(x)

            # Element DOF helper consistent with fe_mthm_bc_3d
            def elem_dofs(elx: int, ely: int, elz: int) -> tuple[np.ndarray, np.ndarray]:
                """Returns bookkeeping matrices mapping local element degrees of freedom to global degrees of freedom.

                Args:
                    elx (int): local element x-index.
                    ely (int): local element y-index.
                    elz (int): local element z-index.

                Returns:
                    edof8 (np.ndarray): mapping local element to global thermal degrees of freedom.
                    edof24 (np.ndarray): mapping local element to global mechanical degrees of freedom.
                """

                def node_id(ix: int, iy: int, iz: int) -> int:
                    """Returns the global index of an element given its coordinates.

                    Args:
                        ix (int): The global x-index of an element.
                        iy (int): The global y-index of an element.
                        iz (int): The global z-index of an element.

                    Returns:
                        g_idx (int): The global index of an element.
                    """
                    return (nely + 1) * (nelz + 1) * ix + (nelz + 1) * iy + iz

                n000 = node_id(elx, ely, elz)
                n100 = node_id(elx + 1, ely, elz)
                n110 = node_id(elx + 1, ely + 1, elz)
                n010 = node_id(elx, ely + 1, elz)
                n001 = node_id(elx, ely, elz + 1)
                n101 = node_id(elx + 1, ely, elz + 1)
                n111 = node_id(elx + 1, ely + 1, elz + 1)
                n011 = node_id(elx, ely + 1, elz + 1)

                edof8 = np.array([n000, n100, n110, n010, n001, n101, n111, n011], dtype=int)
                edof24 = np.empty(24, dtype=int)
                edof24[0::3] = 3 * edof8 + 0
                edof24[1::3] = 3 * edof8 + 1
                edof24[2::3] = 3 * edof8 + 2
                return edof8, edof24

            # Loop elements (clear & readable; vectorization is possible later)
            for elx in range(nelx):
                for ely in range(nely):
                    for elz in range(nelz):
                        edof8, edof24 = elem_dofs(elx, ely, elz)

                        um_e = um[edof24]  # (24,)
                        the = uth[edof8]  # (8,)

                        lamthe = lamth[edof8]  # (8,)

                        x_e = x[ely, elx, elz]
                        x_p = x_e**penal
                        x_p_minus1 = penal * (x_e ** (penal - 1))

                        f0valm += x_p * (um_e @ (ke @ um_e))
                        f0valt += x_p * (the @ (k_eth @ the))

                        df0dx_m[ely, elx, elz] = -x_p_minus1 * um_e.T @ ke @ um_e
                        df0dx_t[ely, elx, elz] = lamthe.T @ (x_p_minus1 * k_eth @ the)
                        df0dx_mat[ely, elx, elz] = (df0dx_m[ely, elx, elz] * w1) + (df0dx_t[ely, elx, elz] * w2)

            f0val = (f0valm * w1) + (f0valt * w2)

            if self.eval_only:
                vf_error = abs(np.mean(x) - volfrac)
                return {
                    "structural_compliance": float(f0valm),
                    "thermal_compliance": float(f0valt),
                    "volume_fraction": vf_error,
                }
            vf_error = np.abs(np.mean(x) - volfrac)
            obj_values = np.array([f0valm, f0valt, vf_error])
            opti_step = OptiStep(obj_values=obj_values, step=iterr)
            opti_steps.append(opti_step)

            xval = x.reshape(n, 1)
            volconst = np.sum(x) / (volfrac * n) - 1.0
            fval = volconst  # scalar
            dfdx = np.ones((1, n), dtype=float) / (volfrac * n)

            # Apply 3D filter to sensitivities
            df0dx_vec = df0dx_mat.reshape(n, 1)
            df0dx_filt = (h @ (xval * df0dx_vec)) / hs[:, None] / np.maximum(1e-3, xval)

            t_sens = time.time() - tcur
            tcur = time.time()

            # MMA update
            if low_vec is None or upp_vec is None:
                upp_vec = np.ones((n,), dtype=float) * upp
                low_vec = np.ones((n,), dtype=float) * low

            mmainputs = MMAInputs(
                m=1,
                n=n,
                iterr=iterr,
                xval=xval[:, 0],
                xmin=xmin,
                xmax=xmax,
                xold1=xold1,
                xold2=xold2,
                df0dx=df0dx_filt[:, 0],
                fval=fval,
                dfdx=dfdx,
                low=low_vec,
                upp=upp_vec,
                a0=a0,
                a=a[0],
                c=c[0],
                d=d[0],
                f0val=f0val,
            )
            xmma, low_vec, upp_vec = mmasub(mmainputs)

            # reshape low_vec to (-1,)
            low_vec = np.squeeze(low_vec)
            upp_vec = np.squeeze(upp_vec)

            # Shift history
            if iterr > SECOND_ITERATION_THRESHOLD:
                xold2 = xold1
                xold1 = xval
            elif iterr > FIRST_ITERATION_THRESHOLD:
                xold1 = xval

            # Update design
            x = xmma.reshape(nely, nelx, nelz)

            # Progress
            change = np.max(np.abs(xmma - xold1))
            change_evol.append(change)
            obj_evol.append(f0val)

            t_mma = time.time() - tcur
            t_total = time.time() - t0
            print(
                f" It.: {iterr:4d} Obj.: {f0val:10.4f} "
                f"Vol.: {np.sum(x) / (nelx * nely * nelz):6.3f} ch.: {change:6.3f} "
                f"|| t_forward:{t_forward:6.3f} + t_adj:{t_adjoints:6.3f} + t_sens:{t_sens:6.3f} + t_mma:{t_mma:6.3f} = {t_total:6.3f}"
            )

            if iterr > self.max_iter:
                break

        print("3D optimization finished.")
        vf_error = abs(np.mean(x) - volfrac)

        return {
            "design": x,
            "bcs": bcs,
            "structural_compliance": float(f0valm),
            "thermal_compliance": float(f0valt),
            "volume_fraction": vf_error,
            "opti_steps": opti_steps,
        }
