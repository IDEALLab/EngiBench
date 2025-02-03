"""Finite Element Model Setup for Thermoelastic 2D Problem."""

import time
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

# ruff: noqa: ERA001, PLR0915, PLR0913


def fe_mthm_bc(
    nely: Any,
    nelx: Any,
    penal: Any,
    x: Any,
    ke: Any,
    k_eth: Any,
    c_ethm: Any,
    tref: Any,
    bcs: Any,
) -> Any:
    """By Gabriel Apaza.

    Last Update: February 2024.

    DESCRIPTION:
    This function completes the formulation of the Finite Element Model for
    topology optimization by completing the different tasks:
    - Builds the global stiffness/conductivity matrix K by summing the
      element matrices KE, with respect to the densities x and the penalty penal.
    - Defines the boundary conditions (fixeddofs) and the loading on the
      domain (F).
    - Solves the governing equations (KU = F)

    INPUTS:
    nely : Number of vertical elements
    nelx : Number of horizontal elements
    penal : Penalty term
    x : Design variables (or densities)
    KE : Element stiffness matrix
    KEth : Element conductivity matrix
    CEthm : Element coupling matrix
    Tref : Reference Temperature
    bcs: Dictionary specifying the boundary conditions

    OUTPUTS:
    Km : Global stiffness matrix
    Kth : Global conductivity matrix
    Um : Displacement vector
    Uth : Temperature vector
    Fm : Mechanical Loading vector
    Fth : Thermal Loading vector
    dCthm : Derivatives of Cthm with respect to Uth
    fixeddofsm : Array of constrained with prescribed mechanical
                 boundary conditions
    alldofsm : Array of all the mechanical degrees of freedoms in
               the domain
    freedofsm : Array of the "free" mechanical degrees of freedoms in
                the structure
    fixeddofsth : Array of constrained with prescribed thermal
                  boundary conditions
    alldofsth : Array of all the thermal degrees of freedoms in
                the domain
    freedofsth : Array of the "free" thermal degrees of freedoms in
                 the structure
    """
    time.time()

    # --------------------------------
    # THERMAL GOVERNING EQUATIONS
    # --------------------------------

    # Nodes at the boundary of the domain
    nn = (nelx + 1) * (nely + 1)  # Total number of nodes

    # Create node numbering grid
    np.arange(nn).reshape((nelx + 1, nely + 1))

    # Thermal BCs
    alldofsth = np.arange(nn)  # All thermal degrees of freedom
    fixeddofsth = np.array(bcs["heatsink_elements"])  # Fixed thermal degrees of freedom
    freedofsth = np.setdiff1d(alldofsth, fixeddofsth)  # Free thermal degrees of freedom

    # --------------------------------
    # Conductivity Matrix (Kth)
    # --------------------------------

    # Initialize lists to store row, col, and data for coo_matrix ------------
    time.time()
    row = []
    col = []
    data = []

    # Loop over each element and compute the corresponding stiffness contributions
    for elx in range(nelx):
        for ely in range(nely):
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edof = np.array([n1 + 1, n2 + 1, n2, n1], dtype=int)

            # Get the indices for the rows and columns in the global stiffness matrix
            dof_pairs = np.array(np.meshgrid(edof, edof)).T.reshape(-1, 2)

            # Compute the values to add to the stiffness matrix
            local_data = (x[ely, elx] ** penal * k_eth).flatten()

            # Append the row, col, and data to the lists for later assembly into coo_matrix
            row.extend(dof_pairs[:, 0])
            col.extend(dof_pairs[:, 1])
            data.extend(local_data)

    # Convert to coo_matrix
    kth = coo_matrix((data, (row, col)), shape=(nn, nn))
    kth = (kth + kth.T) / 2.0
    kth = kth.tolil()

    # --------------------------------
    # Thermal Loading (Fth)
    # --------------------------------
    time.time()
    fth = np.ones(nn) * tref

    tsink = 0  # Sink Temperature

    for dof in fixeddofsth:
        fth[int(dof)] = tsink
        kth.rows[int(dof)] = [int(dof)]
        kth.data[int(dof)] = [1.0]

    # --------------------------------
    # Solve
    # --------------------------------
    time.time()

    # Convert Kth to CSR format for solving
    kth = kth.tocsr()

    # Solve for Uth
    uth = spsolve(kth, fth)

    # --------------------------------
    # Assemble Mechanical System
    # --------------------------------
    time.time()

    # Define free and fixed degrees of freedoms
    dof_per_node = 2
    ndofsm = dof_per_node * (nelx + 1) * (nely + 1)

    fixeddofsm_x = np.array(bcs["fixed_elements"]) * 2
    fixeddofsm_y = np.array(bcs["fixed_elements"]) * 2 + 1

    fixeddofsm = np.concatenate((fixeddofsm_x, fixeddofsm_y))
    # print('fixeddofsm:', fixeddofsm)

    alldofsm = np.arange(ndofsm)
    freedofsm = np.setdiff1d(alldofsm, fixeddofsm)

    # Initialize lists to store row, col, and data for coo_matrix
    km_row = []
    km_col = []
    km_data = []

    feps = np.zeros(ndofsm)
    um = np.zeros(ndofsm)
    d_cthm_row = []
    d_cthm_col = []
    d_cthm_data = []
    time.time()
    for elx in range(nelx):
        for ely in range(nely):
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely

            # "Converted"
            edof4 = np.array([n1 + 1, n2 + 1, n2, n1], dtype=int)
            edof8 = np.array(
                [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1], dtype=int
            )

            # Precompute penalized x values
            penalized_x = x[ely, elx] ** penal

            # Append Km entries
            dof_pairs = np.array(np.meshgrid(edof8, edof8)).T.reshape(-1, 2)
            km_row.extend(dof_pairs[:, 0])
            km_col.extend(dof_pairs[:, 1])
            km_data.extend((penalized_x * ke).flatten())

            # Feps update
            uthe = uth[edof4]
            feps[edof8] += penalized_x * c_ethm @ (uthe - tref)

            # Append dCthm entries
            dof_pairs_d_cthm = np.array(np.meshgrid(edof8, edof4)).T.reshape(-1, 2)
            d_cthm_row.extend(dof_pairs_d_cthm[:, 0])
            d_cthm_col.extend(dof_pairs_d_cthm[:, 1])
            d_cthm_data.extend((penalized_x * c_ethm).flatten())

    # Assemble Km as a coo_matrix
    km = coo_matrix((km_data, (km_row, km_col)), shape=(ndofsm, ndofsm))

    # Assemble dCthm as a coo_matrix
    d_cthm = coo_matrix((d_cthm_data, (d_cthm_row, d_cthm_col)), shape=(ndofsm, nn))

    # DEFINE LOADS
    fp = np.zeros(ndofsm)

    if "force_elements_x" in bcs:
        load_elements_x = np.array(bcs["force_elements_x"]) * 2
        fp[load_elements_x] = 0.5

    if "force_elements_y" in bcs:
        load_elements_y = np.array(bcs["force_elements_y"]) * 2 + 1
        fp[load_elements_y] = 0.5

    fm = fp
    # Fm = Fp + Feps  # Add this line to consider weak coupling between structural and thermal conditions

    # Convert Km to CSR format for solving
    km = km.tocsr()
    km = (km + km.T) / 2.0

    # SOLVING
    um[freedofsm] = spsolve(km[np.ix_(freedofsm, freedofsm)], fm[freedofsm])
    um[fixeddofsm] = 0

    return km, kth, um, uth, fm, fth, d_cthm, fixeddofsm, alldofsm, freedofsm, fixeddofsth, alldofsth, freedofsth, fp
