"""This module contains the MMA subroutine used in the thermoelastic2d problem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# ruff: noqa: PLR2004, PLR0915


@dataclass(frozen=True)
class MMAInputs:
    """Dataclass encapsulating all input parameters for the MMA subroutine."""

    m: int
    n: int
    iterr: int
    xval: NDArray[np.float64]
    xmin: NDArray[np.float64]
    xmax: NDArray[np.float64]
    xold1: NDArray[np.float64]
    xold2: NDArray[np.float64]
    df0dx: NDArray[np.float64]
    fval: NDArray[np.float64]
    dfdx: NDArray[np.float64]
    low: NDArray[np.float64]  # Lower asymptotes from the previous iteration.
    upp: NDArray[np.float64]  # Upper asymptotes from the previous iteration.
    a0: float
    a: NDArray[np.float64]  # Coefficients a_i.
    c: NDArray[np.float64]  # Coefficients c_i.
    d: NDArray[np.float64]  # Coefficients d_i.


# Updated mmasub with strict type annotations.
def mmasub(
    inputs: MMAInputs,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Perform one MMA iteration to solve a nonlinear programming problem.

    Parameters:
        inputs (MMAInputs): A dataclass encapsulating all input parameters.

    Returns:
        tuple: (xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp)
               where each element is an NDArray of type float64.
    """
    # Unpack parameters from the dataclass.
    m = inputs.m
    n = inputs.n
    iterr = inputs.iterr
    xval = inputs.xval
    xmin = inputs.xmin
    xmax = inputs.xmax
    xold1 = inputs.xold1
    xold2 = inputs.xold2
    df0dx = inputs.df0dx
    fval = inputs.fval
    dfdx = inputs.dfdx
    low = inputs.low
    upp = inputs.upp
    a0 = inputs.a0
    a = inputs.a
    c = inputs.c
    d = inputs.d

    # Initialize parameters
    epsimin: float = 1e-7
    raa0: float = 1e-5
    move: float = 1.0
    albefa: float = 0.1
    asyinit: float = 0.01
    asyincr: float = 1.2
    asydecr: float = 0.7

    eeen: NDArray[np.float64] = np.ones(n)
    eeem: NDArray[np.float64] = np.ones(m)

    # Calculation of the asymptotes low and upp
    if iterr < 2.5:
        low = xval - asyinit * (xmax - xmin)
        upp = xval + asyinit * (xmax - xmin)
    else:
        xold1 = xold1.flatten()
        xold2 = xold2.flatten()
        zzz = (xval - xold1) * (xold1 - xold2)
        factor: NDArray[np.float64] = np.ones(n)
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
    pq = 0.001 * (p + q) + raa0 * np.outer(eeem, xmamiinv)
    pq = pq.squeeze()
    p += pq
    q += pq
    p = p * ux2[np.newaxis, :]  # Multiply each row by ux2
    q = q * xl2[np.newaxis, :]  # Multiply each row by xl2
    b = p @ uxinv + q @ xlinv - fval

    subsolv_inputs = SubsolvInputs(
        m=m, n=n, epsimin=epsimin, low=low, upp=upp, alfa=alfa, beta=beta, p0=p0, q0=q0, p=p, q=q, a0=a0, a=a, b=b, c=c, d=d
    )
    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = subsolv(subsolv_inputs)

    return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp


@dataclass(frozen=True)
class SubsolvInputs:
    """Dataclass encapsulating all input parameters for the MMA subsolv subroutine."""

    m: int
    n: int
    epsimin: float
    low: NDArray[np.float64]
    upp: NDArray[np.float64]
    alfa: NDArray[np.float64]
    beta: NDArray[np.float64]
    p0: NDArray[np.float64]
    q0: NDArray[np.float64]
    p: NDArray[np.float64]
    q: NDArray[np.float64]
    a0: float
    a: NDArray[np.float64]
    b: NDArray[np.float64]
    c: NDArray[np.float64]
    d: NDArray[np.float64]


def subsolv(
    inputs: SubsolvInputs,
) -> tuple[
    NDArray[np.float64],  # xmma
    NDArray[np.float64],  # ymma
    float,  # zmma
    NDArray[np.float64],  # lam
    NDArray[np.float64],  # xsi
    NDArray[np.float64],  # eta
    NDArray[np.float64],  # mu
    float,  # zet
    NDArray[np.float64],  # s
]:
    """Solve the MMA subproblem.

    Parameters:
        inputs (SubsolvInputs): Dataclass encapsulating all subproblem parameters.

    Returns:
        tuple: (xmma, ymma, zmma, lam, xsi, eta, mu, zet, s)
               where xmma, ymma, lam, xsi, eta, mu, and s are NDArray[np.float64],
               and zmma, zet are floats.
    """
    # Unpack inputs from the dataclass.
    m = inputs.m
    n = inputs.n
    epsimin = inputs.epsimin
    low = inputs.low.copy()
    upp = inputs.upp.copy()
    alfa = inputs.alfa.copy()
    beta = inputs.beta.copy()
    p0 = inputs.p0.copy()
    q0 = inputs.q0.copy()
    p = inputs.p.copy()
    q = inputs.q.copy()
    a0 = inputs.a0
    a = inputs.a.copy()
    b = inputs.b.copy()
    c = inputs.c.copy()
    d = inputs.d.copy()

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
