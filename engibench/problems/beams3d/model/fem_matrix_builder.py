"""Local structural stiffness matrices for the Beams 3D problem."""

import numpy as np
import numpy.typing as npt

SHAPE_NORM = 0.125  # Hex8 shape function normalization factor


def fe_mech_3d(nu: float, e: float) -> npt.NDArray[np.float64]:
    """Build the structural 3D Hex8 element stiffness matrix."""
    gp = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
    w = np.array([1.0, 1.0])

    xi_nodes = np.array([-1, 1, 1, -1, -1, 1, 1, -1])
    et_nodes = np.array([-1, -1, 1, 1, -1, -1, 1, 1])
    ze_nodes = np.array([-1, -1, -1, -1, 1, 1, 1, 1])

    def shape_derivs(xi: float, eta: float, zeta: float) -> np.ndarray:
        """Evaluate Hex8 shape-function derivatives at one quadrature point."""
        d_n_dxi = np.zeros((8, 3))
        d_n_dxi[:, 0] = SHAPE_NORM * xi_nodes * (1 + et_nodes * eta) * (1 + ze_nodes * zeta)
        d_n_dxi[:, 1] = SHAPE_NORM * et_nodes * (1 + xi_nodes * xi) * (1 + ze_nodes * zeta)
        d_n_dxi[:, 2] = SHAPE_NORM * ze_nodes * (1 + xi_nodes * xi) * (1 + et_nodes * eta)
        return d_n_dxi

    j = np.diag([0.5, 0.5, 0.5])
    det_j = np.linalg.det(j)
    inv_j = np.linalg.inv(j)

    lam = e * nu / ((1 + nu) * (1 - 2 * nu))
    mu = e / (2 * (1 + nu))
    d = np.array(
        [
            [lam + 2 * mu, lam, lam, 0, 0, 0],
            [lam, lam + 2 * mu, lam, 0, 0, 0],
            [lam, lam, lam + 2 * mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu],
        ],
        dtype=float,
    )

    ke = np.zeros((24, 24))

    for i, xi in enumerate(gp):
        for j_idx, eta in enumerate(gp):
            for kq, zeta in enumerate(gp):
                d_n_dx = shape_derivs(xi, eta, zeta) @ inv_j

                b = np.zeros((6, 24))
                for a in range(8):
                    ix = 3 * a
                    dy, dx_, dz = d_n_dx[a, 1], d_n_dx[a, 0], d_n_dx[a, 2]
                    b[0, ix + 0] = dx_
                    b[1, ix + 1] = dy
                    b[2, ix + 2] = dz
                    b[3, ix + 0] = dy
                    b[3, ix + 1] = dx_
                    b[4, ix + 1] = dz
                    b[4, ix + 2] = dy
                    b[5, ix + 0] = dz
                    b[5, ix + 2] = dx_

                wt = w[i] * w[j_idx] * w[kq] * det_j
                ke += (b.T @ d @ b) * wt

    return ke
