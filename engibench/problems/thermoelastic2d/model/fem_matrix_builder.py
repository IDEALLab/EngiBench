"""This module contains the function to build the element stiffness matrix, conductivity matrix, and coupling matrix for thermal expansion."""

import numpy as np


def fe_melthm(param: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function builds the element stiffness matrix, conductivity matrix, and coupling matrix for thermal expansion.

    Args:
        param : list[float]
            A list containing the material properties [nu, E, k, alpha]
            - nu (float): Poisson's ratio
            - E (float): Young's modulus
            - k (float): Thermal conductivity
            - alpha (float): Coefficient of thermal expansion

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]
            - KE (np.ndarray): Element stiffness matrix
            - KEth (np.ndarray): Element conductivity matrix
            - CEthm (np.ndarray): Element coupling matrix (thermal expansion)
    """
    nu = param[0]  # Poisson's ratio
    e = param[1]  # Young's modulus
    k = param[2]  # Thermal conductivity
    alpha = param[3]  # Coefficient of thermal expansion

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

    ke = (e / (1 - nu**2)) * np.array(
        [
            [kel[0], kel[1], kel[2], kel[3], kel[4], kel[5], kel[6], kel[7]],
            [kel[1], kel[0], kel[7], kel[6], kel[5], kel[4], kel[3], kel[2]],
            [kel[2], kel[7], kel[0], kel[5], kel[6], kel[3], kel[4], kel[1]],
            [kel[3], kel[6], kel[5], kel[0], kel[7], kel[2], kel[1], kel[4]],
            [kel[4], kel[5], kel[6], kel[7], kel[0], kel[1], kel[2], kel[3]],
            [kel[5], kel[4], kel[3], kel[2], kel[1], kel[0], kel[7], kel[6]],
            [kel[6], kel[3], kel[4], kel[1], kel[2], kel[7], kel[0], kel[5]],
            [kel[7], kel[2], kel[1], kel[4], kel[3], kel[6], kel[5], kel[0]],
        ]
    )

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
