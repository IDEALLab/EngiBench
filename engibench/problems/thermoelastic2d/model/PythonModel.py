"""This module contains the Python implementation of the thermoelastic 2D problem."""

from math import ceil
from math import sqrt
import time

import cvxopt
from cvxopt import matrix
import cvxopt.cholmod
import numpy as np
from scipy.sparse import coo_matrix

from engibench.problems.thermoelastic2d.model.fem_matrix_builder import fe_melthm
from engibench.problems.thermoelastic2d.model.fem_setup import fe_mthm_bc
from engibench.problems.thermoelastic2d.model.mma_subroutine import MMAInputs
from engibench.problems.thermoelastic2d.model.mma_subroutine import mmasub
from engibench.problems.thermoelastic2d.utils import get_res_bounds
from engibench.problems.thermoelastic2d.utils import plot_multi_physics

# ruff: noqa: PLR2004, ANN001, PLR0915, ANN201, D102, D107, D100, N999, D101


class PythonModel:
    def __init__(self, plot=False, eval_only=False) -> None:
        self.plot = plot
        self.eval_only = eval_only

    def get_initial_design(self, volume_fraction, nelx, nely):
        return volume_fraction * np.ones((nely, nelx))

    def get_matricies(self, nu, e, k, alpha):
        return fe_melthm([nu, e, k, alpha])

    def get_filter(self, nelx, nely, rmin):
        int(nelx * nely * ((2 * (ceil(rmin) - 1) + 1) ** 2))
        i_h = []
        j_h = []
        s_h = []
        for i1 in range(nelx):
            for j1 in range(nely):
                e1 = i1 * nely + j1
                i2_min = max(i1 - (ceil(rmin) - 1), 0)
                i2_max = min(i1 + (ceil(rmin) - 1), nelx - 1)
                for i2 in range(i2_min, i2_max + 1):
                    j2_min = max(j1 - (ceil(rmin) - 1), 0)
                    j2_max = min(j1 + (ceil(rmin) - 1), nely - 1)
                    for j2 in range(j2_min, j2_max + 1):
                        e2 = i2 * nely + j2
                        i_h.append(e1)
                        j_h.append(e2)
                        s_h.append(max(0, rmin - sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2)))

        h = coo_matrix((s_h, (i_h, j_h)), shape=(nelx * nely, nelx * nely)).tocsr()
        hs = np.array(h.sum(axis=1)).flatten()
        return h, hs

    def run(self, bcs, x_init=None):
        """This function runs the optimization algorithm for the coupled structural-thermal problem.

        - bcs : a dictionary containing the boundary conditions for the problem.

        bcs = {
            'fixed_elements': [...], # List of fixed elements
            'load_elements': [...],  # List of force elements
        }
        """
        nelx = bcs["nelx"]
        nely = bcs["nely"]
        volfrac = bcs["volfrac"]
        n = nely * nelx  # Total number of elements

        # 1. Initial Design
        x = self.get_initial_design(volfrac, nelx, nely) if x_init is None else x_init

        # 2. Parameters
        penal = 3  # Penalty term
        rmin = 1.1  # Filter's radius
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

        # 3. Matrices
        ke, k_eth, c_ethm = self.get_matricies(nu, e, k, alpha)

        # 4. Filter
        h, hs = self.get_filter(nelx, nely, rmin)

        # 5. Optimization Loop
        change_evol = []
        obj = []

        while change > 0.01 or iterr < 10:
            iterr += 1
            s_time = time.time()
            curr_time = time.time()

            # FE-ANALYSIS
            results = fe_mthm_bc(nely, nelx, penal, x, ke, k_eth, c_ethm, tref, bcs)
            km, kth, um, uth, fm, fth, d_cthm, fixeddofsm, alldofsm, freedofsm, fixeddofsth, alldofsth, freedofsth, fp = (
                results
            )
            t_forward = time.time() - curr_time
            curr_time = time.time()

            # Plot physics
            if self.plot is True:
                plot_multi_physics(x, fixeddofsm, fixeddofsth, nelx, nely, fp, um, uth, open_plot=True)

            # OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
            ndofm = 2 * (nely + 1) * (nelx + 1)

            # Initialize lamm as a 1D array
            lamm = np.zeros(ndofm)
            flam = fm.flatten()

            # Extract submatrix and vector for free degrees of freedom
            km_ff = km[np.ix_(freedofsm, freedofsm)].toarray()
            flam_freedofs = flam[freedofsm]
            km_ff_sparse = coo_matrix(km_ff)  # Convert to Compressed Sparse Column (CSC) format first
            km_ff_cvxopt = cvxopt.spmatrix(km_ff_sparse.data, km_ff_sparse.row, km_ff_sparse.col)
            flam_freedofs_cvxopt = cvxopt.matrix(flam_freedofs)
            cvxopt.cholmod.linsolve(km_ff_cvxopt, flam_freedofs_cvxopt)
            lamm_freedofs = -np.array(flam_freedofs_cvxopt).flatten()  # Convert back to NumPy array

            # Assign the results back to lamm
            lamm[freedofsm] = lamm_freedofs
            lamm[fixeddofsm] = 0  # This line may be optional since lamm is initialized to zeros
            temp = lamm.T @ d_cthm - um.T @ d_cthm - fth.T
            kth_coo = kth.tocoo()
            kth_cvx = cvxopt.spmatrix(kth_coo.data, kth_coo.row, kth_coo.col, size=kth_coo.shape)
            temp_cvx = matrix(temp.T)
            cvxopt.cholmod.linsolve(kth_cvx, temp_cvx)
            lamth = np.array(temp_cvx).reshape(-1)  # Convert back to a NumPy array if needed

            # PREPARE SENSITIVITY ANALYSIS
            f0val = 0
            f0valm = 0
            f0valt = 0
            df0dx_mat = np.zeros((nely, nelx))

            df0dx_m = np.zeros((nely, nelx))
            df0dx_t = np.zeros((nely, nelx))

            xval = x.reshape(n, 1)
            # DEFINE CONSTRAINTS
            volconst = np.sum(x) / (volfrac * n) - 1
            fval = volconst  # Column vector of size (1xm)
            dfdx = np.ones((1, n)) / (volfrac * n)
            t_sensitivity = time.time() - curr_time
            curr_time = time.time()

            # WEIGHTING
            w1 = bcs.get("weight", 0.5)
            w2 = 1.0 - w1

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
                    f0val += (f0valm * w1) + (f0valt * w2)

                    df0dx_m[ely, elx] = -x_p_minus1 * ume.T @ ke @ ume
                    df0dx_t[ely, elx] = lamthe.T @ (x_p_minus1 * k_eth @ uthe)
                    df0dx_mat[ely, elx] = (df0dx_m[ely, elx] * w1) + (df0dx_t[ely, elx] * w2)

            if self.eval_only is True:
                vf_error = np.abs(np.mean(x) - volfrac)
                return {
                    "sc": f0valm,
                    "tc": f0valt,
                    "vf": vf_error,
                }

            df0dx = df0dx_mat.reshape(nely * nelx, 1)
            df0dx = (h @ (xval * df0dx)) / hs[:, None] / np.maximum(1e-3, xval)

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
                fval=fval,
                dfdx=dfdx,
                low=low_vec,
                upp=upp_vec,
                a0=a0,
                a=a[0],
                c=c[0],
                d=d[0],
            )
            xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low_, upp_ = mmasub(mmainputs)
            t_mma = time.time() - curr_time

            # Store previous density fields
            if iterr > 2:
                xold2 = xold1
                xold1 = xval
            elif iterr > 1:
                xold1 = xval

            x = xmma.reshape(nely, nelx)

            # Print results
            change = np.max(np.abs(xmma - xold1))
            change_evol.append(change)
            obj.append(f0val)
            t_total = time.time() - s_time
            print(
                f" It.: {iterr:4d} Obj.: {f0val:10.4f} Vol.: {np.sum(x) / (nelx * nely):6.3f} ch.: {change:6.3f} || t_forward:{t_forward:6.3f} + t_sensitivity:{t_sensitivity:6.3f} + t_sens_calc:{t_sensitivity_calc:6.3f} + t_mma: {t_mma:6.3f} = {t_total:6.3f}"
            )

        print("Optimization finished...")
        vf_error = np.abs(np.mean(x) - volfrac)

        result = {
            "design": x,
            "bcs": bcs,
            "sc": f0valm,
            "tc": f0valt,
            "vf": vf_error,
        }
        return result


if __name__ == "__main__":
    nelx = 64
    nely = 64

    client = PythonModel(plot=True)

    lci, tri, rci, bri = get_res_bounds(nelx + 1, nely + 1)

    bcs = {
        "nelx": nelx,
        "nely": nely,
        "fixed_elements": [lci[21], lci[32], lci[43]],
        "force_elements_y": [bri[31]],
        "heatsink_elements": [lci[31], lci[32], lci[33]],
        "volfrac": 0.2,
        "rmin": 1.1,
        "weight": 1.0,  # 1.0 for pure structural, 0.0 for pure thermal
    }

    result = client.run(bcs)
