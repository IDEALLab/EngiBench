# ruff: noqa
"""This file is largely based on the MACHAero tutorials.

https://github.com/mdolab/MACH-Aero/blob/main/tutorial/

TEMPLATED VARS:
- $design_fname: Path to the design file.
- $tmp_xyz_fname: Path to the temporary xyz file.
- $mesh_fname: Path to the generated mesh file.
- $ffd_fname: Path to the generated FFD file.

"""

import numpy as np
from pyhyp import pyHyp


def _getupper(design: np.ndarray, npts: int, xtemp):
    myairfoil = np.ones(npts)
    for i in range((npts + 1) // 2):
        myairfoil[i] = abs(design[i, 0] - xtemp)
    myi = np.argmin(myairfoil)
    return design[myi, 1]


def _getlower(design: np.ndarray, npts: int, xtemp):
    myairfoil = np.ones(npts)
    for i in range((npts + 1) // 2, npts):
        myairfoil[i] = abs(design[i, 0] - xtemp)
    myi = np.argmin(myairfoil)
    return design[myi, 1]


if __name__ == "__main__":

    # TODO check with Cashen if we go for .dat or npy format ?
    # with open(args.input_fname, "rb") as f:
    #     design = np.load(f)
    design = np.loadtxt($design_fname)
    npts = design.shape[0]
    nmid = (npts + 1) // 2

    ######## STEP 1: Generate Mesh
    x = design[:, 0].copy()
    y = design[:, 1].copy()
    ndim = x.shape[0]

    airfoil3d = np.zeros((ndim, 2, 3))
    for j in range(2):
        airfoil3d[:, j, 0] = x[:]
        airfoil3d[:, j, 1] = y[:]
    # set the z value on two sides to 0 and 1
    airfoil3d[:, 0, 2] = 0.0
    airfoil3d[:, 1, 2] = 1.0
    # write out plot3d - this is used by pyHyp
    P3D_fname = $tmp_xyz_fname
    with open(P3D_fname, "w") as p3d:
        p3d.write(str(1) + "\n")
        p3d.write(str(ndim) + " " + str(2) + " " + str(1) + "\n")
        for ell in range(3):
            for j in range(2):
                for i in range(ndim):
                    p3d.write("%.15f\n" % (airfoil3d[i, j, ell]))

    # GenOptions
    options = {
        # ---------------------------
        #        Input Parameters
        # ---------------------------
        "inputFile": P3D_fname,
        "unattachedEdgesAreSymmetry": False,
        "outerFaceBC": "farfield",
        "autoConnect": True,
        "BC": {1: {"jLow": "zSymm", "jHigh": "zSymm"}},
        "families": "wall",
        # ---------------------------
        #        Grid Parameters
        # ---------------------------
        "N": 129,
        "s0": 3e-6,
        "marchDist": 100.0,
    }

    hyp = pyHyp(options=options)
    hyp.run()
    hyp.writeCGNS($mesh_fname)

    ######## STEP 2: Generate FFD
    # FFDBox1
    nffd = 10

    ffd_box = np.zeros((nffd, 2, 2, 3))

    xslice = np.zeros(nffd)
    yupper = np.zeros(nffd)
    ylower = np.zeros(nffd)

    xmargin = 0.001
    ymargin1 = 0.02
    ymargin2 = 0.005
    npts = design.shape[0]

    for i in range(nffd):
        xtemp = i * 1.0 / (nffd - 1.0)
        xslice[i] = -1.0 * xmargin + (1 + 2.0 * xmargin) * xtemp
        ymargin = ymargin1 + (ymargin2 - ymargin1) * xslice[i]
        yupper[i] = _getupper(design, npts, xslice[i]) + ymargin
        ylower[i] = _getlower(design, npts, xslice[i]) - ymargin

    # FDBox2
    # X
    ffd_box[:, 0, 0, 0] = xslice[:].copy()
    ffd_box[:, 1, 0, 0] = xslice[:].copy()
    # Y
    # lower
    ffd_box[:, 0, 0, 1] = ylower[:].copy()
    # upper
    ffd_box[:, 1, 0, 1] = yupper[:].copy()
    # copy
    ffd_box[:, :, 1, :] = ffd_box[:, :, 0, :].copy()
    # Z
    ffd_box[:, :, 0, 2] = 0.0
    # Z
    ffd_box[:, :, 1, 2] = 1.0

    with open($ffd_fname, "w") as f:
        f.write("1\n")
        f.write(str(nffd) + " 2 2\n")
        for ell in range(3):
            for k in range(2):
                for j in range(2):
                    for i in range(nffd):
                        f.write("%.15f " % (ffd_box[i, j, k, ell]))
                    f.write("\n")

    print("Generated files FFD and mesh")
