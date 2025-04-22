"""Finite Element Model Setup for Thermoelastic 2D Problem."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from engibench.problems.thermoelastic2d.utils import binary_matrix_to_indices


@dataclass
class FEMthmBCResult:
    """Dataclass encapsulating all output parameters for the fem setup code."""

    km: csr_matrix  # Global mechanical stiffness matrix.
    kth: csr_matrix  # Global thermal conductivity matrix.
    um: np.ndarray  # Displacement vector.
    uth: np.ndarray  # Temperature vector.
    fm: np.ndarray  # Mechanical loading vector.
    fth: np.ndarray  # Thermal loading vector.
    d_cthm: coo_matrix  # Derivative of the coupling matrix with respect to temperature.
    fixeddofsm: np.ndarray  # Array of fixed mechanical degrees of freedom.
    alldofsm: np.ndarray  # Array of all mechanical degrees of freedom.
    freedofsm: np.ndarray  # Array of free mechanical degrees of freedom.
    fixeddofsth: np.ndarray  # Array of fixed thermal degrees of freedom.
    alldofsth: np.ndarray  # Array of all thermal degrees of freedom.
    freedofsth: np.ndarray  # Array of free thermal degrees of freedom.
    fp: np.ndarray  # Force vector used for mechanical loading.


def fe_mthm_bc(  # noqa: PLR0915, PLR0913
    nely: int,
    nelx: int,
    penal: float,
    x: np.ndarray,
    ke: np.ndarray,
    k_eth: np.ndarray,
    c_ethm: np.ndarray,
    tref: float,
    bcs: dict[str, Any],
) -> FEMthmBCResult:
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
        FEMthmBCResult: Dataclass containing the following fields:
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
    # Domain Weighting
    # - 0.0 for pure thermal
    # - 1.0 for pure structural
    weight = bcs.get("weight", 0.5)

    # ---------------------------
    # THERMAL GOVERNING EQUATIONS
    # ---------------------------
    nn = (nelx + 1) * (nely + 1)  # Total number of nodes

    # Create node numbering grid (not used later)
    np.arange(nn).reshape((nelx + 1, nely + 1))

    # Thermal BCs
    alldofsth = np.arange(nn)  # All thermal degrees of freedom
    fixeddofsth = np.array(binary_matrix_to_indices(bcs["heatsink_elements"]))
    freedofsth = np.setdiff1d(alldofsth, fixeddofsth)

    # ---------------------------
    # Conductivity Matrix (Kth)
    # ---------------------------
    row: list[float] = []
    col: list[float] = []
    data = []

    for elx in range(nelx):
        for ely in range(nely):
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edof = np.array([n1 + 1, n2 + 1, n2, n1], dtype=int)
            dof_pairs = np.array(np.meshgrid(edof, edof)).T.reshape(-1, 2)
            local_data = ((2e-2 + x[ely, elx]) ** penal * k_eth).flatten()
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
    # ASSEMBLE MECHANICAL SYSTEM
    # ---------------------------
    dof_per_node = 2
    ndofsm = dof_per_node * (nelx + 1) * (nely + 1)

    # Fixed degrees of freedom (dofs)
    fixeddofsm_x = np.array(binary_matrix_to_indices(bcs["fixed_elements"])) * 2
    fixeddofsm_y = np.array(binary_matrix_to_indices(bcs["fixed_elements"])) * 2 + 1
    fixeddofsm = np.concatenate((fixeddofsm_x, fixeddofsm_y))
    alldofsm = np.arange(ndofsm)
    freedofsm = np.setdiff1d(alldofsm, fixeddofsm)

    # Number of elements
    n_elements = nelx * nely
    elx_idx, ely_idx = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
    elx_idx = elx_idx.ravel()
    ely_idx = ely_idx.ravel()

    # Compute the base node numbers for each element
    n1 = (nely + 1) * elx_idx + ely_idx
    n2 = (nely + 1) * (elx_idx + 1) + ely_idx

    # Construct element degree-of-freedom arrays:
    edof4 = np.stack([n1 + 1, n2 + 1, n2, n1], axis=1)
    edof8 = np.stack([2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1], axis=1)

    # Compute penalized element stiffness factors.
    penalized = x[ely_idx, elx_idx] ** penal

    # --- Assemble stiffness matrix contributions ---
    km_block = penalized[:, None, None] * ke
    km_row = np.repeat(edof8, 8, axis=1)
    km_col = np.tile(edof8, 8)
    km_data = km_block.reshape(n_elements, 64)

    # Flatten arrays for sparse assembly
    km_row = km_row.ravel()
    km_col = km_col.ravel()
    km_data = km_data.ravel()

    # --- Assemble d_cthm contributions ---
    d_cthm_block = penalized[:, None, None] * c_ethm
    d_cthm_row = np.repeat(edof8, 4, axis=1)
    d_cthm_col = np.tile(edof4, 8)
    d_cthm_data = d_cthm_block.reshape(n_elements, 32)

    # Flatten arrays for sparse assembly
    d_cthm_row = d_cthm_row.ravel()
    d_cthm_col = d_cthm_col.ravel()
    d_cthm_data = d_cthm_data.ravel()

    # --- Assemble thermal force contributions ---
    uthe = uth[edof4]  # shape: (n_elements, 4)
    diff = uthe - tref  # shape: (n_elements, 4)
    thermal = penalized[:, None] * (c_ethm @ diff.T).T  # shape: (n_elements, 8)

    # Instead of looping to add contributions to feps, use np.bincount to accumulate.
    feps = np.bincount(edof8.ravel(), weights=thermal.ravel(), minlength=ndofsm)

    # --- Create sparse matrices ---
    km = coo_matrix((km_data, (km_row, km_col)), shape=(ndofsm, ndofsm))
    d_cthm = coo_matrix((d_cthm_data, (d_cthm_row, d_cthm_col)), shape=(ndofsm, nn))

    # ---------------------------
    # DEFINE LOADS
    # ---------------------------
    fp = np.zeros(ndofsm)
    if "force_elements_x" in bcs:
        load_elements_x = np.array(binary_matrix_to_indices(bcs["force_elements_x"])) * 2
        fp[load_elements_x] = 0.5
    if "force_elements_y" in bcs:
        load_elements_y = np.array(binary_matrix_to_indices(bcs["force_elements_y"])) * 2 + 1
        fp[load_elements_y] = 0.5

    # Total force vector includes thermal contributions
    if weight == 0.0:  # pure thermal
        fm = feps.astype(np.float64)
    elif weight == 1.0:  # pure structural
        fm = fp
    else:
        fm = fp + feps

    # Finalize the stiffness matrix by converting to CSR format and symmetrizing.
    km = km.tocsr()
    km = (km + km.T) / 2.0

    # ---------------------------
    # SOLVE THE SYSTEM
    # ---------------------------
    um = np.zeros(ndofsm)
    um[freedofsm] = spsolve(km[np.ix_(freedofsm, freedofsm)], fm[freedofsm])
    um[fixeddofsm] = 0

    return FEMthmBCResult(
        km=km,
        kth=kth,
        um=um,
        uth=uth,
        fm=fm,
        fth=fth,
        d_cthm=d_cthm,
        fixeddofsm=fixeddofsm,
        alldofsm=alldofsm,
        freedofsm=freedofsm,
        fixeddofsth=fixeddofsth,
        alldofsth=alldofsth,
        freedofsth=freedofsth,
        fp=fp,
    )
