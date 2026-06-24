# mypy: ignore-errors
"""2D airfoil surface mesh + fitted FFD (v1).

Based on the MACH-Aero tutorials and the reference 2D ``mesh_template.py`` used to
collect the v1 dataset. It builds a thin single-section O-mesh with symmetry planes on
both spanwise (z) faces so ADflow returns a true 2D solution, and a fitted FFD cage.

Unlike the v0 ``pre_process.py`` this does NOT call ``makeBluntTE``: the v1 baselines
are already clean blunt-TE sections (ordered TE->upper->LE->lower->TE, x in [0, 0.98]),
so re-blunting would shrink the section and break the near-TE constraint rays.
"""

import json
import sys

from cli_interface import PreprocessParameters
import prefoil
from pyhyp import pyHyp

if __name__ == "__main__":
    args = PreprocessParameters.from_dict(json.loads(sys.argv[1]))

    coords = prefoil.utils.readCoordFile(args.design_fname)
    airfoil = prefoil.Airfoil(coords)
    print("Running pre_process.py (v1)")

    if not args.input_blunted:
        # Safety path only: the v1 pipeline feeds pre-blunted sections.
        airfoil.normalizeAirfoil()
        airfoil.makeBluntTE(xCut=args.x_cut)

    # Resample for a smooth CFD surface distribution (preserves x_max=0.98).
    coords = airfoil.getSampledPts(
        args.N_sample,
        spacingFunc=prefoil.sampling.conical,
        func_args={"coeff": 1.2},
        nTEPts=args.n_tept_s,
    )

    # Write a fitted FFD with ffd_pts chordwise control points.
    airfoil.generateFFD(args.ffd_pts, args.ffd_fname, ymarginu=args.ffd_ymarginu, ymarginl=args.ffd_ymarginl)

    # write out plot3d surface for pyHyp (sampled coords; x_max stays 0.98)
    airfoil.writeCoords(args.tmp_xyz_fname, file_format="plot3d")

    s0 = max(args.s0, 5e-6)

    # pyHyp options matched to the reference 2D mesh. The only 2D-specific deviation
    # from the 3D wing mesh is the boundary conditions: a 2D solution needs explicit
    # symmetry planes on both spanwise (z) faces.
    options = {
        # General options
        "inputFile": args.tmp_xyz_fname + ".xyz",
        "fileType": "PLOT3D",
        "unattachedEdgesAreSymmetry": False,
        "outerFaceBC": "farfield",
        "autoConnect": True,
        "BC": {1: {"jLow": "zSymm", "jHigh": "zSymm"}},
        "families": "wall",
        # Grid parameters
        "N": args.N_grid,
        "s0": s0,
        "marchDist": args.march_dist,
        # Pseudo grid parameters
        "ps0": args.ps0,
        "pGridRatio": args.p_grid_ratio,
        "cMax": args.c_max,
        # Smoothing parameters
        "epsE": args.eps_e,
        "epsI": args.eps_i,
        "theta": args.theta,
        "volCoef": args.vol_coef,
        "volBlend": args.vol_blend,
        "volSmoothIter": args.vol_smooth_iter,
        # Solution parameters
        "kspRelTol": args.ksp_rel_tol,
        "kspMaxIts": args.ksp_max_its,
        "kspSubspaceSize": args.ksp_subspace_size,
    }

    hyp = pyHyp(options=options)
    hyp.run()
    hyp.writeCGNS(args.mesh_fname)

    print(f"Generated FFD and mesh in {args.ffd_fname}, {args.mesh_fname}")
