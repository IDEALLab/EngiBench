# mypy: ignore-errors
"""This file is largely based on the MACHAero tutorials.

https://github.com/mdolab/MACH-Aero/blob/main/tutorial/

TODO: Add the automatic grid spacing calculation.
"""

import json
import sys

from cli_interface import PreprocessParameters
import prefoil
from pyhyp import pyHyp

if __name__ == "__main__":
    args = PreprocessParameters(**json.loads(sys.argv[1]))

    coords = prefoil.utils.readCoordFile(args.design_fname)
    airfoil = prefoil.Airfoil(coords)
    print("Running pre-process.py")
    input_blunted = args.input_blunted
    if not input_blunted:
        airfoil.normalizeAirfoil()
        airfoil.makeBluntTE(xCut=args.x_cut)

    N_sample = args.N_sample
    n_tept_s = args.n_tept_s

    coords = airfoil.getSampledPts(
        N_sample,
        spacingFunc=prefoil.sampling.conical,
        func_args={"coeff": 1.2},
        nTEPts=n_tept_s,
    )

    # Write a fitted FFD with 10 chordwise points
    ffd_ymarginu = args.ffd_ymarginu
    ffd_ymarginl = args.ffd_ymarginl
    ffd_fname = args.ffd_fname
    ffd_pts = args.ffd_pts
    airfoil.generateFFD(ffd_pts, ffd_fname, ymarginu=ffd_ymarginu, ymarginl=ffd_ymarginl)

    # write out plot3d
    airfoil.writeCoords(args.tmp_xyz_fname, file_format="plot3d")

    # GenOptions
    options = {
        # ---------------------------
        #        Input Parameters
        # ---------------------------
        "inputFile": args.tmp_xyz_fname + ".xyz",
        "unattachedEdgesAreSymmetry": False,
        "outerFaceBC": "farfield",
        "autoConnect": True,
        "BC": {1: {"jLow": "zSymm", "jHigh": "zSymm"}},
        "families": "wall",
        # ---------------------------
        #        Grid Parameters
        # ---------------------------
        "N": args.N_grid,
        "nConstantStart": 8,
        "s0": args.s0,
        "marchDist": args.march_dist,
        # Smoothing parameters
        "volSmoothIter": 150,
        "volCoef": 0.25,
        "volBlend": 0.001,
    }

    hyp = pyHyp(options=options)
    hyp.run()
    hyp.writeCGNS(args.mesh_fname)

    print(f"Generated files FFD and mesh in {ffd_fname}, {args.mesh_fname}")
