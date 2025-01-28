# ruff: noqa: N806

# ADAPTED FROM: A 200 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN, JANUARY 2013
# Original code updated by Niels Aage February 2016 (see "topopt_fast.py")
# Adapted by Arthur Drake January 2025
# from __future__ import division

# import seaborn as sns
# import matplotlib.pyplot as plt
# from IPython.display import clear_output
# import torch
# import os, sys
import cvxopt
import cvxopt.cholmod
import numpy as np
from scipy.sparse import coo_matrix

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # # Boundary conditions? Didn't ask

# # # "Simulator" just being a simple function? Yes

# # # Is docker still necessary for this simpler problem?
# Not necessary

# # # Once done with implementation, Make sure to commit code that follows the pre-commit standards


def main(nelx, nely, volfrac, penal, rmin, ft, max_iter=100, overhang_constraint=False, display=False):
    if display:
        print("Minimum compliance problem with OC")
        print("ndes: " + str(nelx) + " x " + str(nely))
        print("volfrac: " + str(volfrac) + ", rmin: " + str(rmin) + ", penal: " + str(penal))
        print("Filter method: " + ["Sensitivity based", "Density based"][ft])

    # Max and min stiffness
    Emin = 1e-9  # 1e-9
    Emax = 1.0

    # dofs:
    ndof = 2 * (nelx + 1) * (nely + 1)

    # g=0 # must be initialized to use the NGuyen/Paulino OC approach

    # FE: Build the index vectors for the for coo matrix format.
    KE = lk()
    edofMat = np.zeros((nelx * nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edofMat[el, :] = np.array(
                [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1]
            )
    # Construct the index pointers for the coo format
    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()

    # Filter: Build (and assemble) the index+data vectors for the coo matrix format
    nfilter = int(nelx * nely * ((2 * (np.ceil(rmin) - 1) + 1) ** 2))
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    cc = 0
    for i in range(nelx):
        for j in range(nely):
            row = i * nely + j
            kk1 = int(np.maximum(i - (np.ceil(rmin) - 1), 0))
            kk2 = int(np.minimum(i + np.ceil(rmin), nelx))
            ll1 = int(np.maximum(j - (np.ceil(rmin) - 1), 0))
            ll2 = int(np.minimum(j + np.ceil(rmin), nely))
            for k in range(kk1, kk2):
                for l in range(ll1, ll2):
                    col = k * nely + l
                    fac = rmin - np.sqrt((i - k) * (i - k) + (j - l) * (j - l))
                    iH[cc] = row
                    jH[cc] = col
                    sH[cc] = np.maximum(0.0, fac)
                    cc = cc + 1
    # Finalize assembly and convert to csc format
    H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsc()
    Hs = H.sum(1)

    # BC's and support
    dofs = np.arange(2 * (nelx + 1) * (nely + 1))
    fixed = np.union1d(dofs[0 : 2 * (nely + 1) : 2], np.array([2 * (nelx + 1) * (nely + 1) - 1]))
    free = np.setdiff1d(dofs, fixed)

    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))

    # Set load
    f[1, 0] = -1

    # Allocate design variables (as array), initialize and allocate sens.
    x = volfrac * np.ones(nely * nelx, dtype=float)
    xPhys = x = volfrac * np.ones(nely * nelx, dtype=float)

    xPrint, _, _, _ = base_filter(xPhys, nelx, nely, None, None, overhang_constraint=overhang_constraint)

    loop = 0
    change = 1
    dv = np.ones(nely * nelx)
    dc = np.ones(nely * nelx)
    ce = np.ones(nely * nelx)

    while change > 0.025 and loop < max_iter:  # while change>0.01 and loop<max_iter:
        loop = loop + 1

        # TODO: Separate into its own function

        # Setup and solve FE problem
        sK = ((KE.flatten()[np.newaxis]).T * (Emin + (xPrint) ** penal * (Emax - Emin))).flatten(order="F")
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        # Remove constrained dofs from matrix and convert to coo
        K = deleterowcol(K, fixed, fixed).tocoo()
        # Solve system
        K = cvxopt.spmatrix(K.data, K.row.astype(int), K.col.astype(int))
        B = cvxopt.matrix(f[free, 0])
        cvxopt.cholmod.linsolve(K, B)
        u[free, 0] = np.array(B)[:, 0]

        ############################################################################################################
        # Objective and sensitivity
        ce = (np.dot(u[edofMat].reshape(nelx * nely, 8), KE) * u[edofMat].reshape(nelx * nely, 8)).sum(1)

        # COMPLIANCE (OBJECTIVE)
        c = ((Emin + xPrint**penal * (Emax - Emin)) * ce).sum()

        # Derivative of compliance wrt density
        dc = (-penal * xPrint ** (penal - 1) * (Emax - Emin)) * ce

        # wrt volume fraction (don't change this)
        dv = np.ones(nely * nelx)
        ############################################################################################################

        # Sensitivity filtering
        # print(dc.shape, np.min(dc), np.max(dc))
        # print(dv.shape, np.min(dv), np.max(dv))

        # xPrint, dx_m, dc, dv = filter(xPrint, nelx, nely, device, model, dc, dv, exp=exp)

        xPrint, dx_m, dc, dv = base_filter(xPhys, nelx, nely, dc, dv, overhang_constraint)  # MATLAB implementation

        # xPrint, dx_m, dc, dv = filter(xPhys, nelx, nely, device, model, dc, dv, exp=exp, L=1)
        # for _ in range(nely-1):
        #     xPrint, dx_m, dc, dv = filter(xPrint, nelx, nely, device, model, dc, dv, exp=exp, L=1)

        if ft == 0:
            dc = np.asarray((H * (x * dc))[np.newaxis].T / Hs)[:, 0] / np.maximum(0.001, x)  # type: ignore
        elif ft == 1:
            dc = np.asarray(H * (dc[np.newaxis].T / Hs))[:, 0]
            dv = np.asarray(H * (dv[np.newaxis].T / Hs))[:, 0]

        # Optimality criteria
        l1 = 0
        l2 = 1e9
        move = 0.2
        # reshape to perform vector operations
        xnew = np.zeros(nelx * nely)

        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            if lmid > 0:
                xnew = np.maximum(
                    0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid))))
                )  # type: ignore
            else:
                xnew = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, x + move)))

            # Filter design variables
            if ft == 0:
                xPhys = xnew
            elif ft == 1:
                xPhys = np.asarray(H * xnew[np.newaxis].T / Hs)[:, 0]

            xPrint, _, _, _ = base_filter(xPhys, nelx, nely, None, None, overhang_constraint=overhang_constraint)

            if xPrint.sum() > volfrac * nelx * nely:
                l1 = lmid
            else:
                l2 = lmid
            if l1 + l2 == 0:
                break

        # Compute the change by the inf. norm
        change = np.linalg.norm(xnew.reshape(nelx * nely, 1) - x.reshape(nelx * nely, 1), np.inf)
        x = [item for item in xnew]
        x = np.array(x)

        # xPrint_m = to_image(xPrint, nelx, nely)
        # xPhys_m = to_image(xPhys, nelx, nely)
        # dc_m = to_image(dc, nelx, nely)
        # dv_m = to_image(dv, nelx, nely)

        if display:
            # clear_output(wait=True)
            print(f"Iteration {loop}/{max_iter}")
            # titles = ['negative dc', 'dv', 'dx', 'xPhys', 'xPrint']
            # _, axs = plt.subplots(1,5,figsize=(30,3))
            # [ax.axes.xaxis.set_visible(False) for ax in axs]
            # [ax.axes.yaxis.set_visible(False) for ax in axs]
            # sns.heatmap(ax=axs[0], data=-dc_m[0][0], vmin=0, vmax=10)
            # sns.heatmap(ax=axs[1], data=dv_m[0][0], vmin=0, vmax=10)
            # sns.heatmap(ax=axs[2], data=dx_m[0][0], vmin=0, vmax=10)
            # sns.heatmap(ax=axs[3], data=xPhys_m[0][0], vmin=0, vmax=1)
            # sns.heatmap(ax=axs[4], data=xPrint_m[0][0], vmin=0, vmax=1)
            # [axs[i].title.set_text(titles[i]) for i in range(len(titles))]
            # plt.show()

    return xPrint, dc, dv, c


# element stiffness matrix
def lk():
    E = 1  # 1
    nu = 0.3  # 0.3
    k = np.array(
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
    KE = (
        E
        / (1 - nu**2)
        * np.array(
            [
                [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
            ]
        )
    )
    return KE


def deleterowcol(A, delrow, delcol):
    # Assumes that matrix is in symmetric csc form !
    m = A.shape[0]
    keep = np.delete(np.arange(0, m), delrow)
    A = A[keep, :]
    keep = np.delete(np.arange(0, m), delcol)
    A = A[:, keep]
    return A


def base_filter(x1, nelx, nely, dc, dv, overhang_constraint=False):
    x = to_image(x1, nelx, nely)[0][0]
    if overhang_constraint:
        if dc is None:
            dx_m = np.ones(x.shape)
        else:
            dc = to_image(dc, nelx, nely)[0][0]
            dv = to_image(dv, nelx, nely)[0][0]
        P = 40
        ep = 1e-4
        xi_0 = 0.5
        Ns = 3
        nSens = 2  # dc and dv (hard-coded)

        Q = P + np.log(Ns) / np.log(xi_0)
        SHIFT = 100 * (np.finfo(float).tiny) ** (1 / P)
        BACKSHIFT = 0.95 * (Ns ** (1 / Q)) * (SHIFT ** (P / Q))
        xi = np.zeros(x.shape)
        Xi = np.zeros(x.shape)
        keep = np.zeros(x.shape)
        sq = np.zeros(x.shape)

        xi[nely - 1, :] = [item for item in x[nely - 1, :]]
        for i in reversed(range(nely - 1)):
            cbr = np.array([0] + list(xi[i + 1, :]) + [0]) + SHIFT
            keep[i, :] = cbr[:nelx] ** P + cbr[1 : nelx + 1] ** P + cbr[2:] ** P
            Xi[i, :] = keep[i, :] ** (1 / Q) - BACKSHIFT
            sq[i, :] = np.sqrt((x[i, :] - Xi[i, :]) ** 2 + ep)
            xi[i, :] = 0.5 * ((x[i, :] + Xi[i, :]) - sq[i, :] + np.sqrt(ep))

        if dc is not None and dv is not None:
            varargin = [dc, dv]
            dc_copy = [[x for x in y] for y in dc]
            dv_copy = [[x for x in y] for y in dv]
            dfxi = [np.array(dc_copy), np.array(dv_copy)]
            dfx = [np.array(dc_copy), np.array(dv_copy)]
            lamb = np.zeros((nSens, nelx))
            for i in range(nely - 1):
                dsmindx = 0.5 * (1 - (x[i, :] - Xi[i, :]) / sq[i, :])
                dsmindXi = 1 - dsmindx
                cbr = np.array([0] + list(xi[i + 1, :]) + [0]) + SHIFT

                dmx = np.zeros((Ns, nelx))
                for j in range(Ns):
                    dmx[j, :] = (P / Q) * (keep[i, :] ** (1 / Q - 1)) * (cbr[j : nelx + j] ** (P - 1))

                qi = np.ravel([[i] * 3 for i in range(nelx)])
                qj = qi + [-1, 0, 1] * nelx
                qs = np.ravel(dmx.T)

                dsmaxdxi = coo_matrix((qs[1:-1], (qi[1:-1], qj[1:-1]))).tocsc()
                for k in range(nSens):
                    dfx[k][i, :] = dsmindx * (dfxi[k][i, :] + lamb[k, :])
                    lamb[k, :] = ((dfxi[k][i, :] + lamb[k, :]) * dsmindXi) @ dsmaxdxi

            i = nely - 1
            for k in range(nSens):
                dfx[k][i, :] = dfx[k][i, :] + lamb[k, :]

            dc = dfx[0]
            dv = dfx[1]
            dx_m = np.nan_to_num(dc / varargin[0], 1)  # type: ignore
            dc = to_array(dc)
            dv = to_array(dv)

        dx_m = np.expand_dims(dx_m, axis=(0, 1))
        xi = to_array(xi)
    else:
        dx_m = np.expand_dims(np.ones(x.shape), axis=(0, 1))
        xi = x1

    return xi, dx_m, dc, dv


# def npy(tensor):
#     return tensor.cpu().detach().numpy()


def to_image(data, nelx=100, nely=50):
    return np.swapaxes(data.reshape(nelx, nely), 0, 1)
    # if torch.is_tensor(data):
    #     return torch.swapaxes(data.view(1, 1, nelx, nely), 2, 3)
    # else:
    #     return np.swapaxes(data.reshape(1, 1, nelx, nely), 2, 3)


def to_array(data):
    shift = 0 if len(data.shape) == 4 else -2

    return np.swapaxes(data, int(2 + shift), int(3 + shift)).ravel()
    # if torch.is_tensor(data):
    #     return torch.flatten(torch.swapaxes(data, int(2+shift), int(3+shift)))
    # else:
    #     return np.swapaxes(data, int(2+shift), int(3+shift)).ravel()


if __name__ == "__main__":
    # Default input parameters
    nelx = 180
    nely = 60
    volfrac = 0.4
    rmin = 5.4
    penal = 3.0
    ft = 1  # ft==0 -> sens, ft==1 -> dens

    import sys

    if len(sys.argv) > 1:
        nelx = int(sys.argv[1])
    if len(sys.argv) > 2:
        nely = int(sys.argv[2])
    if len(sys.argv) > 3:
        volfrac = float(sys.argv[3])
    if len(sys.argv) > 4:
        rmin = float(sys.argv[4])
    if len(sys.argv) > 5:
        penal = float(sys.argv[5])
    if len(sys.argv) > 6:
        ft = int(sys.argv[6])

    main(nelx, nely, volfrac, penal, rmin, ft)
