"""Finite Element Model Setup for Thermoelastic 2D Problem."""

from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# ruff: noqa: PLR0915, PLR0913


def fe_mthm_bc(
    nely: int,
    nelx: int,
    penal: float,
    x: np.ndarray,
    ke: np.ndarray,
    k_eth: np.ndarray,
    c_ethm: np.ndarray,
    tref: float,
    bcs: dict[str, Any],
) -> tuple[
    csr_matrix,  # km: Global mechanical stiffness matrix
    csr_matrix,  # kth: Global thermal conductivity matrix
    np.ndarray,  # um: Displacement vector
    np.ndarray,  # uth: Temperature vector
    np.ndarray,  # fm: Mechanical loading vector
    np.ndarray,  # fth: Thermal loading vector
    coo_matrix,  # d_cthm: Derivative coupling matrix with respect to temperature
    np.ndarray,  # fixeddofsm: Fixed mechanical degrees of freedom
    np.ndarray,  # alldofsm: All mechanical degrees of freedom
    np.ndarray,  # freedofsm: Free mechanical degrees of freedom
    np.ndarray,  # fixeddofsth: Fixed thermal degrees of freedom
    np.ndarray,  # alldofsth: All thermal degrees of freedom
    np.ndarray,  # freedofsth: Free thermal degrees of freedom
    np.ndarray,  # fp: Force vector used for mechanical loading
]:
    """Constructs the finite element model matrices for coupled structural-thermal topology optimization.

    This function assembles the global mechanical and thermal matrices for a coupled
    structural-thermal topology optimization problem. It builds the global stiffness (mechanical)
    and conductivity (thermal) matrices, applies the prescribed boundary conditions and loads,
    and solves the governing equations for both the displacement and temperature fields.

    Args:
        nely (int): Number of vertical elements.
        nelx (int): Number of horizontal elements.
        penal (Union[int, float]): SIMP penalty factor used to penalize intermediate densities.
        x (np.ndarray): 2D array of design variables (densities) with shape (nely, nelx).
        ke (np.ndarray): Element stiffness matrix.
        k_eth (np.ndarray): Element conductivity matrix.
        c_ethm (np.ndarray): Element coupling matrix between the thermal and mechanical fields.
        tref (float): Reference temperature.
        bcs (Dict[str, Any]): Dictionary specifying boundary conditions. Expected keys include:
            - "heatsink_elements": Indices for fixed thermal degrees of freedom.
            - "fixed_elements": Indices for fixed mechanical degrees of freedom.
            - "force_elements_x" (optional): Indices for x-direction force elements.
            - "force_elements_y" (optional): Indices for y-direction force elements.

    Returns:
        Tuple[csr_matrix, csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray, coo_matrix,
              np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - km (csr_matrix): Global mechanical stiffness matrix.
            - kth (csr_matrix): Global thermal conductivity matrix.
            - um (np.ndarray): Displacement vector.
            - uth (np.ndarray): Temperature vector.
            - fm (np.ndarray): Mechanical loading vector.
            - fth (np.ndarray): Thermal loading vector.
            - d_cthm (coo_matrix): Derivative of the coupling matrix with respect to temperature.
            - fixeddofsm (np.ndarray): Array of fixed mechanical degrees of freedom.
            - alldofsm (np.ndarray): Array of all mechanical degrees of freedom.
            - freedofsm (np.ndarray): Array of free mechanical degrees of freedom.
            - fixeddofsth (np.ndarray): Array of fixed thermal degrees of freedom.
            - alldofsth (np.ndarray): Array of all thermal degrees of freedom.
            - freedofsth (np.ndarray): Array of free thermal degrees of freedom.
            - fp (np.ndarray): Force vector used for mechanical loading.
    """
    # ---------------------------
    # THERMAL GOVERNING EQUATIONS
    # ---------------------------
    nn = (nelx + 1) * (nely + 1)  # Total number of nodes

    # Create node numbering grid (not used later)
    np.arange(nn).reshape((nelx + 1, nely + 1))

    # Thermal BCs
    alldofsth = np.arange(nn)  # All thermal degrees of freedom
    fixeddofsth = np.array(bcs["heatsink_elements"])
    freedofsth = np.setdiff1d(alldofsth, fixeddofsth)

    # ---------------------------
    # Conductivity Matrix (Kth)
    # ---------------------------
    row = []
    col = []
    data = []

    for elx in range(nelx):
        for ely in range(nely):
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edof = np.array([n1 + 1, n2 + 1, n2, n1], dtype=int)
            dof_pairs = np.array(np.meshgrid(edof, edof)).T.reshape(-1, 2)
            local_data = (x[ely, elx] ** penal * k_eth).flatten()
            row.extend(dof_pairs[:, 0])
            col.extend(dof_pairs[:, 1])
            data.extend(local_data)

    kth = coo_matrix((data, (row, col)), shape=(nn, nn))
    kth = (kth + kth.T) / 2.0
    kth = kth.tolil()

    # Thermal Loading (Fth)
    fth = np.ones(nn) * tref
    tsink = 0  # Sink Temperature

    for dof in fixeddofsth:
        fth[int(dof)] = tsink
        kth.rows[int(dof)] = [int(dof)]
        kth.data[int(dof)] = [1.0]

    kth = kth.tocsr()
    uth = spsolve(kth, fth)

    # ---------------------------
    # ASSEMBLE MECHANICAL SYSTEM (Vectorized Version)
    # ---------------------------
    dof_per_node = 2
    ndofsm = dof_per_node * (nelx + 1) * (nely + 1)
    um = np.zeros(ndofsm)

    # Fixed degrees of freedom (dof)
    fixeddofsm_x = np.array(bcs["fixed_elements"]) * 2
    fixeddofsm_y = np.array(bcs["fixed_elements"]) * 2 + 1
    fixeddofsm = np.concatenate((fixeddofsm_x, fixeddofsm_y))
    alldofsm = np.arange(ndofsm)
    freedofsm = np.setdiff1d(alldofsm, fixeddofsm)

    # --- Create element connectivity in a fully vectorized way ---
    x, y = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="xy")
    n1 = (nely + 1) * x + y  # shape: (nely, nelx)
    n2 = (nely + 1) * (x + 1) + y  # shape: (nely, nelx)

    n1_flat = n1.flatten()  # shape: (E,)
    n2_flat = n2.flatten()  # shape: (E,)

    # Build connectivity arrays:
    edof4 = np.column_stack((n1_flat + 1, n2_flat + 1, n2_flat, n1_flat))  # shape: (E, 4)
    edof8 = np.column_stack(
        (
            2 * n1_flat + 2,
            2 * n1_flat + 3,
            2 * n2_flat + 2,
            2 * n2_flat + 3,
            2 * n2_flat,
            2 * n2_flat + 1,
            2 * n1_flat,
            2 * n1_flat + 1,
        )
    )  # shape: (E, 8)

    # Compute the penalized stiffness factor per element (vectorized for all E)
    penalized_x = (x.flatten()) ** penal  # shape: (E,)

    # --- Assemble the global stiffness matrix (km) ---
    block_idx = np.arange(8)
    block_rows_8 = np.repeat(block_idx, 8)  # shape: (64,)
    block_cols_8 = np.tile(block_idx, 8)  # shape: (64,)

    km_rows = edof8[:, block_rows_8]  # shape: (E, 64)
    km_cols = edof8[:, block_cols_8]  # shape: (E, 64)
    km_data = penalized_x[:, None] * ke.flatten()[None, :]  # shape: (E, 64)

    # Flatten the arrays so that they can be passed to coo_matrix
    km_rows = km_rows.ravel()
    km_cols = km_cols.ravel()
    km_data = km_data.ravel()

    # --- Assemble the coupling matrix (d_cthm) ---
    block_rows_8x4 = np.repeat(np.arange(8), 4)  # shape: (32,)
    block_cols_8x4 = np.tile(np.arange(4), 8)  # shape: (32,)

    d_cthm_rows = edof8[:, block_rows_8x4]  # shape: (E, 32)
    d_cthm_cols = edof4[:, block_cols_8x4]  # shape: (E, 32)
    d_cthm_data = penalized_x[:, None] * c_ethm.flatten()[None, :]  # shape: (E, 32)

    d_cthm_rows = d_cthm_rows.ravel()
    d_cthm_cols = d_cthm_cols.ravel()
    d_cthm_data = d_cthm_data.ravel()

    # --- Assemble the global feps vector ---
    uthe = uth[edof4]  # shape: (E, 4)
    diff = uthe - tref  # shape: (E, 4)
    feps_contrib = penalized_x[:, None] * (diff @ c_ethm.T)  # shape: (E, 8)
    feps = np.zeros(ndofsm)
    np.add.at(feps, edof8.ravel(), feps_contrib.ravel())

    # --- Create sparse matrices ---
    km = coo_matrix((km_data, (km_rows, km_cols)), shape=(ndofsm, ndofsm))
    d_cthm = coo_matrix((d_cthm_data, (d_cthm_rows, d_cthm_cols)), shape=(ndofsm, nn))

    # ----- END OPTIMIZED VERSION -----

    # DEFINE LOADS
    fp = np.zeros(ndofsm)

    if "force_elements_x" in bcs:
        load_elements_x = np.array(bcs["force_elements_x"]) * 2
        fp[load_elements_x] = 0.5

    if "force_elements_y" in bcs:
        load_elements_y = np.array(bcs["force_elements_y"]) * 2 + 1
        fp[load_elements_y] = 0.5

    fm = fp + feps

    km = km.tocsr()
    km = (km + km.T) / 2.0

    um[freedofsm] = spsolve(km[np.ix_(freedofsm, freedofsm)], fm[freedofsm])
    um[fixeddofsm] = 0

    return km, kth, um, uth, fm, fth, d_cthm, fixeddofsm, alldofsm, freedofsm, fixeddofsth, alldofsth, freedofsth, fp
