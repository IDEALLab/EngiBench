# mypy: ignore-errors
"""This file is largely based on the MACHAero tutorials.

https://github.com/mdolab/MACH-Aero/blob/main/tutorial/

TEMPLATED VARS:
- $design_fname: Path to the design file.
- $N_sample: Number of points to sample on the airfoil surface. Defines part of the mesh resolution.
- $nTEPts: Number of points on the trailing edge.
- $xCut: Blunt edge dimensionless cut location.
- $tmp_xyz_fname: Path to the temporary xyz file.
- $mesh_fname: Path to the generated mesh file.
- $ffd_fname: Path to the generated FFD file.
- $ffd_ymarginu: Upper (y-axis) margin for the fitted FFD cage.
- $ffd_ymarginl: Lower (y-axis) margin for the fitted FFD cage.
- $ffd_pts: Number of FFD points.
- $N_grid: Number of grid levels to march from the airfoil surface. Defines part of the mesh resolution.
- $s0: Off-the-wall spacing for the purpose of modeling the boundary layer. # TODO: Add the automatic grid spacing calculation.
- $marchDist: Distance to march the grid from the airfoil surface.

"""

import numpy as np
from pyhyp import pyHyp
import prefoil

if __name__ == "__main__":

    coords = prefoil.utils.readCoordFile($design_fname) # type: ignore
    airfoil = prefoil.Airfoil(coords)
    print("Running pre-process.py")
    input_blunted = $input_blunted
    if not input_blunted:
        airfoil.normalizeAirfoil()
        airfoil.makeBluntTE(xCut=$xCut)

    N_sample = $N_sample
    nTEPts = $nTEPts

    coords = airfoil.getSampledPts(
        N_sample,
        spacingFunc=prefoil.sampling.conical,
        func_args={"coeff": 1.2},
        nTEPts=nTEPts,
    )

    # Write a fitted FFD with 10 chordwise points
    ffd_ymarginu = $ffd_ymarginu
    ffd_ymarginl = $ffd_ymarginl
    ffd_fname = $ffd_fname
    ffd_pts = $ffd_pts
    airfoil.generateFFD(ffd_pts, ffd_fname, ymarginu=ffd_ymarginu, ymarginl=ffd_ymarginl)

    # write out plot3d
    airfoil.writeCoords($tmp_xyz_fname, file_format="plot3d")

    # GenOptions
    options = {
        # ---------------------------
        #        Input Parameters
        # ---------------------------
        "inputFile": $tmp_xyz_fname + ".xyz",
        "unattachedEdgesAreSymmetry": False,
        "outerFaceBC": "farfield",
        "autoConnect": True,
        "BC": {1: {"jLow": "zSymm", "jHigh": "zSymm"}},
        "families": "wall",
        # ---------------------------
        #        Grid Parameters
        # ---------------------------
        "N": $N_grid,
        "nConstantStart": 8,
        "s0": $s0,
        "marchDist": $marchDist,
        # Smoothing parameters
        "volSmoothIter": 150,
        "volCoef": 0.25,
        "volBlend": 0.001
        # "volSmoothSchedule": [[0, 0], [0.2, 2], [0.5, 200], [1.0, 1000]],
    }

    hyp = pyHyp(options=options)
    hyp.run()
    hyp.writeCGNS($mesh_fname)

    print(f"Generated files FFD and mesh in ${ffd_fname}, ${mesh_fname}")
