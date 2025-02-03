"""This module contains the MMA subroutine used in the thermoelastic2d problem."""

from typing import Any

import numpy as np

# ruff: noqa: PLR2004, ERA001, PLR0915


def mmasub(list_inputs: Any) -> Any:
    """Perform one MMA iteration to solve a nonlinear programming problem.

    Parameters:
    m (int): Number of general constraints.
    n (int): Number of variables x_j.
    iter (int): Current iteration number (1 on the first call).
    xval (ndarray): Current values of the variables x_j.
    xmin (ndarray): Lower bounds for the variables x_j.
    xmax (ndarray): Upper bounds for the variables x_j.
    xold1 (ndarray): xval from one iteration ago.
    xold2 (ndarray): xval from two iterations ago.
    df0dx (ndarray): Derivatives of the objective function f_0 with respect to x_j.
    fval (ndarray): Values of the constraint functions f_i at xval.
    dfdx (ndarray): Derivatives of the constraint functions f_i with respect to x_j.
    low (ndarray): Lower asymptotes from the previous iteration.
    upp (ndarray): Upper asymptotes from the previous iteration.
    a0 (float): Constant a_0 in the term a_0 * z.
    a (ndarray): Constants a_i in the terms a_i * z.
    c (ndarray): Constants c_i in the terms c_i * y_i.
    d (ndarray): Constants d_i in the terms 0.5 * d_i * y_i^2.

    Returns:
    tuple: (xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp)
    """
    m, n, iterr, xval, xmin, xmax, xold1, xold2, df0dx, fval, dfdx, low, upp, a0, a, c, d = list_inputs

    # Initialize parameters
    epsimin = 1e-7
    raa0 = 1e-5
    move = 1.0
    albefa = 0.1
    asyinit = 0.01
    asyincr = 1.2
    asydecr = 0.7

    eeen = np.ones(n)
    eeem = np.ones(m)
    np.zeros(n)

    # Calculation of the asymptotes low and upp
    if iterr < 2.5:
        low = xval - asyinit * (xmax - xmin)
        upp = xval + asyinit * (xmax - xmin)
    else:
        xold1 = xold1.flatten()
        xold2 = xold2.flatten()
        zzz = (xval - xold1) * (xold1 - xold2)
        factor = np.ones(n)
        factor[zzz > 0] = asyincr
        factor[zzz < 0] = asydecr
        low = xval - factor * (xold1 - low)
        upp = xval + factor * (upp - xold1)
        lowmin = xval - 0.2 * (xmax - xmin)
        lowmax = xval - 0.01 * (xmax - xmin)
        uppmin = xval + 0.01 * (xmax - xmin)
        uppmax = xval + 0.2 * (xmax - xmin)
        low = np.maximum(low, lowmin)
        low = np.minimum(low, lowmax)
        upp = np.minimum(upp, uppmax)
        upp = np.maximum(upp, uppmin)

    # Calculation of the bounds alfa and beta
    zzz1 = low + albefa * (xval - low)
    zzz2 = xval - move * (xmax - xmin)
    zzz = np.maximum(zzz1, zzz2)
    alfa = np.maximum(zzz, xmin)
    zzz1 = upp - albefa * (upp - xval)
    zzz2 = xval + move * (xmax - xmin)
    zzz = np.minimum(zzz1, zzz2)
    beta = np.minimum(zzz, xmax)

    # Calculations of p0, q0, P, Q and b
    xmami = xmax - xmin
    xmamieps = 1e-5 * eeen
    xmami = np.maximum(xmami, xmamieps)
    xmamiinv = eeen / xmami
    ux1 = upp - xval
    ux2 = ux1 * ux1
    xl1 = xval - low
    xl2 = xl1 * xl1
    uxinv = eeen / ux1
    xlinv = eeen / xl1

    p0 = np.maximum(df0dx, 0)
    q0 = np.maximum(-df0dx, 0)
    pq0 = 0.001 * (p0 + q0) + raa0 * xmamiinv
    p0 += pq0
    q0 += pq0
    p0 *= ux2
    q0 *= xl2

    p = np.maximum(dfdx, 0)
    q = np.maximum(-dfdx, 0)
    pq = 0.001 * (p + q) + raa0 * np.outer(eeem, xmamiinv)  # shape (1, 4096)
    pq = pq.squeeze()
    p += pq
    q += pq
    p = p * ux2[np.newaxis, :]  # Multiply each row of P by ux2
    q = q * xl2[np.newaxis, :]  # Multiply each row of Q by xl2
    b = p @ uxinv + q @ xlinv - fval

    # Solving the subproblem by a primal-dual Newton method
    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = subsolv(
        [m, n, epsimin, low, upp, alfa, beta, p0, q0, p, q, a0, a, b, c, d]
    )

    return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp


def subsolv(list_inputs: Any) -> Any:
    """Solve the MMA subproblem."""
    m, n, epsimin, low, upp, alfa, beta, p0, q0, p, q, a0, a, b, c, d = list_inputs

    een = np.ones(n)
    eem = np.ones(m)
    epsi = 1.0
    epsvecn = epsi * een
    epsvecm = epsi * eem
    x = 0.5 * (alfa + beta)
    y = eem.copy()
    z = 1.0
    lam = eem.copy()
    xsi = een / (x - alfa)
    xsi = np.maximum(xsi, een)
    eta = een / (beta - x)
    eta = np.maximum(eta, een)
    mu = np.maximum(eem, 0.5 * c)
    zet = 1.0
    s = eem.copy()
    itera = 0

    while epsi > epsimin:
        epsvecn = epsi * een
        epsvecm = epsi * eem
        ux1 = upp - x
        xl1 = x - low
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        uxinv1 = een / ux1
        xlinv1 = een / xl1

        plam = p0 + p.T @ lam
        qlam = q0 + q.T @ lam
        gvec = p @ uxinv1 + q @ xlinv1
        dpsidx = plam / ux2 - qlam / xl2

        rex = dpsidx - xsi + eta
        rey = c + d * y - mu - lam
        rez = a0 - zet - np.dot(a, lam)
        relam = gvec - a * z - y + s - b
        rexsi = xsi * (x - alfa) - epsvecn
        reeta = eta * (beta - x) - epsvecn
        remu = mu * y - epsvecm
        rezet = zet * z - epsi
        res = lam * s - epsvecm

        residu1 = np.concatenate([rex, rey, [rez]])
        residu2 = np.concatenate([relam, rexsi, reeta, remu, [rezet], res])
        residu = np.concatenate([residu1, residu2])
        residunorm = np.sqrt(np.dot(residu, residu))
        residumax = np.max(np.abs(residu))
        ittt = 0
        while residumax > 0.9 * epsi and ittt < 500:
            ittt += 1
            itera += 1
            ux1 = upp - x
            xl1 = x - low
            ux2 = ux1 * ux1
            xl2 = xl1 * xl1
            ux3 = ux1 * ux2
            xl3 = xl1 * xl2
            uxinv1 = een / ux1
            xlinv1 = een / xl1
            uxinv2 = een / ux2
            xlinv2 = een / xl2

            plam = p0 + p.T @ lam
            qlam = q0 + q.T @ lam
            gvec = p @ uxinv1 + q @ xlinv1
            gg = p @ np.diag(uxinv2) - q @ np.diag(xlinv2)
            dpsidx = plam / ux2 - qlam / xl2
            delx = dpsidx - epsvecn / (x - alfa) + epsvecn / (beta - x)
            dely = c + d * y - lam - epsvecm / y
            delz = a0 - np.dot(a, lam) - epsi / z
            dellam = gvec - a * z - y - b + epsvecm / lam
            diagx = plam / ux3 + qlam / xl3
            diagx = 2 * diagx + xsi / (x - alfa) + eta / (beta - x)
            diagxinv = een / diagx
            diagy = d + mu / y
            diagyinv = eem / diagy
            diaglam = s / lam
            diaglamyi = diaglam + diagyinv

            if m < n:
                blam = dellam + dely / diagy - gg @ (delx / diagx)
                bb = np.concatenate([blam, [delz]])
                alam = np.diag(diaglamyi) + gg @ np.diag(diagxinv) @ gg.T
                aa = np.vstack([np.hstack([alam, a.reshape(-1, 1)]), np.hstack([a.reshape(1, -1), np.array([[-zet / z]])])])
                solut = np.linalg.solve(aa, bb)
                dlam = solut[:m]
                dz = solut[m]
                dx = -delx / diagx - (gg.T @ dlam) / diagx
            else:
                diaglamyiinv = eem / diaglamyi
                dellamyi = dellam + dely / diagy
                axx = np.diag(diagx) + gg.T @ np.diag(diaglamyiinv) @ gg
                azz = zet / z + np.dot(a, a / diaglamyi)
                axz = -gg.T @ (a / diaglamyi)
                bx = delx + gg.T @ (dellamyi / diaglamyi)
                bz = delz - np.dot(a, dellamyi / diaglamyi)
                aa = np.vstack([np.hstack([axx, axz.reshape(-1, 1)]), np.hstack([axz.reshape(1, -1), np.array([[azz]])])])
                bb = -np.concatenate([bx, [bz]])
                solut = np.linalg.solve(aa, bb)
                dx = solut[:n]
                dz = solut[n]
                dlam = (gg @ dx) / diaglamyi - dz * (a / diaglamyi) + dellamyi / diaglamyi

            dy = -dely / diagy + dlam / diagy
            dxsi = -xsi + epsvecn / (x - alfa) - (xsi * dx) / (x - alfa)
            deta = -eta + epsvecn / (beta - x) + (eta * dx) / (beta - x)
            dmu = -mu + epsvecm / y - (mu * dy) / y
            dzet = -zet + epsi / z - zet * dz / z
            ds = -s + epsvecm / lam - (s * dlam) / lam
            xx = np.concatenate([y, [z], lam, xsi, eta, mu, [zet], s])
            dxx = np.concatenate([dy, [dz], dlam, dxsi, deta, dmu, [dzet], ds])

            stepxx = -1.01 * dxx / xx
            stmxx = np.max(stepxx)
            stepalfa = -1.01 * dx / (x - alfa)
            stmalfa = np.max(stepalfa)
            stepbeta = 1.01 * dx / (beta - x)
            stmbeta = np.max(stepbeta)
            stmalbe = np.maximum(stmalfa, stmbeta)
            stmalbexx = np.maximum(stmalbe, stmxx)
            stminv = np.maximum(stmalbexx, 1.0)
            steg = 1.0 / stminv

            xold = x.copy()
            yold = y.copy()
            zold = z
            lamold = lam.copy()
            xsiold = xsi.copy()
            etaold = eta.copy()
            muold = mu.copy()
            zetold = zet
            sold = s.copy()

            itto = 0
            resinew = 2 * residunorm
            while resinew > residunorm and itto < 50:
                itto += 1
                x = xold + steg * dx
                y = yold + steg * dy
                z = zold + steg * dz
                lam = lamold + steg * dlam
                xsi = xsiold + steg * dxsi
                eta = etaold + steg * deta
                mu = muold + steg * dmu
                zet = zetold + steg * dzet
                s = sold + steg * ds
                ux1 = upp - x
                xl1 = x - low
                ux2 = ux1 * ux1
                xl2 = xl1 * xl1
                uxinv1 = een / ux1
                xlinv1 = een / xl1

                plam = p0 + p.T @ lam
                qlam = q0 + q.T @ lam
                gvec = p @ uxinv1 + q @ xlinv1
                dpsidx = plam / ux2 - qlam / xl2

                rex = dpsidx - xsi + eta
                rey = c + d * y - mu - lam
                rez = a0 - zet - np.dot(a, lam)
                relam = gvec - a * z - y + s - b
                rexsi = xsi * (x - alfa) - epsvecn
                reeta = eta * (beta - x) - epsvecn
                remu = mu * y - epsvecm
                rezet = zet * z - epsi
                res = lam * s - epsvecm

                residu1 = np.concatenate([rex, rey, [rez]])
                residu2 = np.concatenate([relam, rexsi, reeta, remu, [rezet], res])
                residu = np.concatenate([residu1, residu2])
                resinew = np.sqrt(np.dot(residu, residu))
                steg = steg / 2.0

            residunorm = resinew
            residumax = np.max(np.abs(residu))
            steg = 2.0 * steg

        epsi *= 0.1

    xmma = x
    ymma = y
    zmma = z
    lamma = lam
    xsimma = xsi
    etamma = eta
    mumma = mu
    zetmma = zet
    smma = s

    return xmma, ymma, zmma, lamma, xsimma, etamma, mumma, zetmma, smma
