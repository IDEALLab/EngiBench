"""This module contains the function to build the element stiffness matrix, conductivity matrix, and coupling matrix for thermal expansion."""

import numpy as np


def fe_melthm(nu: float, e: float, k: float, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function builds the element stiffness matrix, conductivity matrix, and coupling matrix for thermal expansion.

    Args:
        nu (float): Poisson's ratio.
        e (float): Young's modulus (modulus of elasticity).
        k (float): Thermal conductivity.
        alpha (float): Coefficient of thermal expansion.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]
            - KE (np.ndarray): Element stiffness matrix
            - KEth (np.ndarray): Element conductivity matrix
            - CEthm (np.ndarray): Element coupling matrix (thermal expansion)
    """
    # Construct element stiffness matrix
    kel = np.array(
        [
            1 / 2 - nu / 6,
            1 / 8 + nu / 8,
            -1 / 4 - nu / 12,
            -1 / 8 + 3 * nu / 8,
            -1 / 4 + nu / 12,
            -1 / 8 - nu / 8,
            nu / 6,
            1 / 8 - 3 * nu / 8,
        ]
    )

    indices = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 0, 7, 6, 5, 4, 3, 2],
        [2, 7, 0, 5, 6, 3, 4, 1],
        [3, 6, 5, 0, 7, 2, 1, 4],
        [4, 5, 6, 7, 0, 1, 2, 3],
        [5, 4, 3, 2, 1, 0, 7, 6],
        [6, 3, 4, 1, 2, 7, 0, 5],
        [7, 2, 1, 4, 3, 6, 5, 0],
    ]

    ke = (e / (1 - nu**2)) * kel[indices]

    # Construct element conductivity matrix
    k_eth = (k / 6) * np.array([[4, -1, -2, -1], [-1, 4, -1, -2], [-2, -1, 4, -1], [-1, -2, -1, 4]])

    # Element coupling matrix (thermal expansion)
    c_ethm = (e * alpha / (6 * (1 - nu))) * np.array(
        [
            [-2, -2, -1, -1],
            [-2, -1, -1, -2],
            [2, 2, 1, 1],
            [-1, -2, -2, -1],
            [1, 1, 2, 2],
            [1, 2, 2, 1],
            [-1, -1, -2, -2],
            [2, 1, 1, 2],
        ]
    )

    return ke, k_eth, c_ethm
