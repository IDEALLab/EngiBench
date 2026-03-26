"""This module assembles the local stiffness matrices the elastic, thermal, and thermoelastic domains."""

import numpy as np
from numpy import typing as npt

SHAPE_NORM = 0.125  # Hex8 shape function normalization factor


def fe_melthm_3d(
    nu: float, e: float, k: float, alpha: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Build 3D Hex8 element matrices for thermo-elasticity.

    Args:
        nu (float): Poisson's ratio
        e (float): Young's modulus
        k (float): Thermal conductivity (isotropic)
        alpha (float): Coefficient of thermal expansion

    Returns:
        ke    : 24x24 mechanical stiffness (Hex8, 3 dof/node)
        k_eth :  8x8 thermal conductivity (Hex8, 1 dof/node)
        c_ethm: 24x8 coupling mapping nodal temperatures to equivalent mechanical forces
    """
    # --- Gauss points (2x2x2) and weights ---
    gp = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
    w = np.array([1.0, 1.0])

    # --- Hex8 shape functions and derivatives in natural coordinates ---
    # Node order: (-,-,-), (+,-,-), (+,+,-), (-,+,-), (-,-,+), (+,-,+), (+,+,+), (-,+,+)
    xi_nodes = np.array([-1, 1, 1, -1, -1, 1, 1, -1])
    et_nodes = np.array([-1, -1, 1, 1, -1, -1, 1, 1])
    ze_nodes = np.array([-1, -1, -1, -1, 1, 1, 1, 1])

    def shape_fun_and_derivs(xi: float, eta: float, zeta: float) -> tuple[np.ndarray, np.ndarray]:
        """Builds the shape function for the local elements.

        Args:
            xi (float): Gaussian x coordinate
            eta (float): Gaussian y coordinate
            zeta (float): Gaussian z coordinate

        Returns:
            n: (8,) shape functions
            d_n_dxi: (8,3) derivatives w.r.t. [xi, eta, zeta]
        """
        n = SHAPE_NORM * (1 + xi_nodes * xi) * (1 + et_nodes * eta) * (1 + ze_nodes * zeta)

        # Derivatives with respect to natural coords
        d_n_dxi = np.zeros((8, 3))
        d_n_dxi[:, 0] = SHAPE_NORM * xi_nodes * (1 + et_nodes * eta) * (1 + ze_nodes * zeta)  # ∂N/∂xi
        d_n_dxi[:, 1] = SHAPE_NORM * et_nodes * (1 + xi_nodes * xi) * (1 + ze_nodes * zeta)  # ∂N/∂eta
        d_n_dxi[:, 2] = SHAPE_NORM * ze_nodes * (1 + xi_nodes * xi) * (1 + et_nodes * eta)  # ∂N/∂zeta
        return n, d_n_dxi

    # --- Geometry & Jacobian ---
    # For a unit cube in physical space mapped from [-1,1]^3:
    # x = (xi+1)/2, y = (eta+1)/2, z = (zeta+1)/2 -> J = diag(0.5, 0.5, 0.5)
    j = np.diag([0.5, 0.5, 0.5])
    det_j = np.linalg.det(j)  # = 0.125
    inv_j = np.linalg.inv(j)  # = diag(2,2,2)

    # --- Elasticity matrix (Voigt 6x6) for 3D isotropic ---
    lam = e * nu / ((1 + nu) * (1 - 2 * nu))
    mu = e / (2 * (1 + nu))
    d = np.array(
        [
            [lam + 2 * mu, lam, lam, 0, 0, 0],
            [lam, lam + 2 * mu, lam, 0, 0, 0],
            [lam, lam + 2 * mu, lam + 2 * mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu],
        ],
        dtype=float,
    )
    d[2, 1] = lam

    # Thermal "volumetric" strain direction in Voigt
    e_th = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    # --- Allocate element matrices ---
    ke = np.zeros((24, 24))
    k_eth = np.zeros((8, 8))
    c_ethm = np.zeros((24, 8))

    # --- Integration loop ---
    for i, xi in enumerate(gp):
        for j_idx, eta in enumerate(gp):
            for kq, zeta in enumerate(gp):
                n, d_n_dxi = shape_fun_and_derivs(xi, eta, zeta)

                # Gradients in physical coords: dN_dx = dN_dxi * invJ
                d_n_dx = d_n_dxi @ inv_j  # (8,3)

                # Build B-matrix (6 x 24) for Hex8, 3 dof/node (u,v,w)
                b = np.zeros((6, 24))
                for a in range(8):
                    ix = 3 * a
                    dy, dx_, dz = d_n_dx[a, 1], d_n_dx[a, 0], d_n_dx[a, 2]  # clarity
                    # normal strains
                    b[0, ix + 0] = dx_
                    b[1, ix + 1] = dy
                    b[2, ix + 2] = dz
                    # shear strains (engineering)
                    b[3, ix + 0] = dy
                    b[3, ix + 1] = dx_
                    b[4, ix + 1] = dz
                    b[4, ix + 2] = dy
                    b[5, ix + 0] = dz
                    b[5, ix + 2] = dx_

                # Thermal gradient matrix for conduction: G = grad(N) (3 x 8)
                g = d_n_dx.T  # rows: [dN/dx; dN/dy; dN/dz]

                # Weight factor
                wt = w[i] * w[j_idx] * w[kq] * det_j

                # --- Accumulate ---
                # Mechanical stiffness
                ke += (b.T @ d @ b) * wt

                # Thermal conductivity (isotropic k * grad N^T grad N)
                k_eth += (g.T @ g) * (k * wt)

                # Thermo-mech coupling: columns correspond to nodal temperatures (via N_j)
                de_th = d @ (alpha * e_th)  # (6,)
                c_ethm += (b.T @ de_th[:, None] @ n[None, :]) * wt

    return ke, k_eth, c_ethm
