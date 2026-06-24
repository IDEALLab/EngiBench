# mypy: ignore-errors
"""2D airfoil shape optimization (v1).

Based on the MACH-Aero tutorials and the reference 2D ``airfoil_opt_v0_2d.py`` used to
collect the v1 dataset. Relative to the v0 ``airfoil_opt.py`` this:

- uses IPOPT (gradient-based, adjoint gradients) by default,
- treats lift as an INEQUALITY constraint (``0 <= cl - cl_target <= 0.2*cl_target``),
- widens the FFD shape design variables to +/-0.10 (chord-normalized),
- samples temperature as part of the flow condition,
- applies an internal per-Mach ANK/NK solver schedule so the user does not have to
  tune the CFD solver (override via the advanced ``solver_options``),
- writes richer debugging artifacts: the full optimization history (``opt.hst`` with
  stored sensitivities), per-iteration surface/section files (intermediates tarred,
  first+last kept loose), the final absolute area (``final_abs_volume.npy``), and the
  IPOPT log (``IPOPT.out``).

The volume/area and thickness constraints are intentionally kept identical to v0 (the
constraint is scaled against a fixed baseline ``area_input_design`` so optimized areas
stay comparable across designs).
"""

import glob
import json
import os
import sys
import tarfile

from adflow import ADFLOW
from baseclasses import AeroProblem
from cli_interface import Algorithm
from cli_interface import DEFAULT_SURFACE_VARIABLES
from cli_interface import OptimizeParameters
from idwarp import USMesh
from mpi4py import MPI
from multipoint import multiPointSparse
import numpy as np
from pygeo import DVConstraints
from pygeo import DVGeometry
from pyoptsparse import OPT
from pyoptsparse import Optimization

# Mach-regime thresholds for the ANK/NK solver schedule.
_MACH_TRANSONIC_LO = 0.65
_MACH_TRANSONIC_HI = 0.8
_MACH_SONIC = 1.0


def ank_schedule(mach: float) -> dict:
    """Per-Mach ANK/NK solver schedule (abstracts CFD-solver tuning from the user).

    A pure pseudo-transient (ANK) scheme stalls in the last few orders on the tiny 2D
    O-mesh, which corrupts the adjoint and fails the optimizer; the schedule below was
    tuned (alongside the NK finisher) across the Mach range so transonic/supersonic
    cases still converge. Returns the Mach-dependent ADflow options.
    """
    ank_physical_ls_tol = 0.2
    cfl = 1.0
    ank_cfl_cutback = 0.05
    n_cycles = 10000
    ank_second_ord_switch_tol = 5e-8

    if _MACH_TRANSONIC_LO < mach < _MACH_TRANSONIC_HI:
        ank_second_ord_switch_tol = 1e-6
        ank_physical_ls_tol = 0.3
    elif _MACH_TRANSONIC_HI <= mach < _MACH_SONIC:
        ank_second_ord_switch_tol = 1e-5
        ank_physical_ls_tol = 0.4
    elif mach >= _MACH_SONIC:
        ank_second_ord_switch_tol = 1e-5
        ank_physical_ls_tol = 0.4
        cfl = 0.75

    return {
        "nCycles": n_cycles,
        "CFL": cfl,
        "ANKPhysicalLSTol": ank_physical_ls_tol,
        "ANKCFLCutback": ank_cfl_cutback,
        "ANKSecondOrdSwitchTol": ank_second_ord_switch_tol,
        "useDissContinuation": False,
        "dissContMagnitude": 0.2 * mach,
    }


def ipopt_options(output_dir: str) -> dict:
    """Default IPOPT options used to collect the v1 dataset."""
    return {
        "max_iter": 300,
        "constr_viol_tol": 5e-06,
        "nlp_scaling_method": "gradient-based",
        "bound_relax_factor": 1e-8,
        "acceptable_tol": 5e-03,
        "acceptable_iter": 5,
        "tol": 1e-3,
        "dual_inf_tol": 5e-4,
        "print_level": 5,
        "output_file": os.path.join(output_dir, "IPOPT.out"),
        "file_print_level": 5,
        "mu_strategy": "adaptive",
        "limited_memory_max_history": 50,
        "print_user_options": "yes",
        "mumps_mem_percent": 2000,
        "check_derivatives_for_naninf": "yes",
        "watchdog_shortened_iter_trigger": 3,
        "watchdog_trial_iter_max": 5,
        # Cap wasted line-search effort on extreme/failed trial geometries: max_soc=0
        # disables second-order corrections (each adds CFD evals) and alpha_min_frac
        # abandons a search direction once the step gets small instead of grinding it down.
        "max_soc": 0,
        "alpha_min_frac": 0.05,
    }


def _archive_intermediates(output_dir: str) -> None:
    """Tar the per-iteration surface/section files, keeping the first+last loose.

    The first (baseline) and last (optimized) designs stay loose so downstream parsing
    (``simulator_output_to_design`` reads ``output/*_slices.dat``) keeps working.
    """
    for pattern in ("fc_*_surf.cgns", "fc_*_slices.dat"):
        file_list = sorted(glob.glob(os.path.join(output_dir, pattern)))
        if len(file_list) > 2:  # noqa: PLR2004  -- need at least a baseline + optimized + 1 intermediate
            tar_fname = os.path.join(output_dir, pattern.replace("*", "intermediate") + ".tar.gz")
            with tarfile.open(tar_fname, "w:gz") as tar:
                for fname in file_list[1:-1]:
                    tar.add(fname, arcname=os.path.basename(fname))
            for fname in file_list[1:-1]:
                os.remove(fname)
    for pattern in ("fc_*_surf.plt", "fc_*_lift.dat"):
        for fname in glob.glob(os.path.join(output_dir, pattern)):
            os.remove(fname)


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Entry point of the script."""
    args = OptimizeParameters.from_dict(json.loads(sys.argv[1]))
    # cL target (inequality lower bound)
    mycl = args.cl_target
    alpha = args.alpha
    mach = args.mach
    reynolds = args.reynolds
    altitude = args.altitude
    temperature = args.temperature
    use_altitude = args.use_altitude
    reynolds_length = 1.0
    # Area constraint (kept identical to v0: scaled against a fixed baseline)
    area_ratio_min = args.area_ratio_min
    area_initial = args.area_initial
    area_input_design = args.area_input_design

    opt = args.opt
    surface_variables = args.surface_variables if args.surface_variables is not None else DEFAULT_SURFACE_VARIABLES

    # ======================================================================
    #         Create multipoint communication object
    # ======================================================================
    mp = multiPointSparse(MPI.COMM_WORLD)
    mp.addProcessorSet("cruise", nMembers=1, memberSizes=MPI.COMM_WORLD.size)
    comm, *_ = mp.createCommunicators()
    if not os.path.exists(args.output_dir) and comm.rank == 0:
        os.mkdir(args.output_dir)

    # ======================================================================
    #         ADflow Set-up
    # ======================================================================
    aero_options = {
        # Common Parameters
        "gridFile": args.mesh_fname,
        "outputDirectory": args.output_dir,
        "writeVolumeSolution": False,
        "surfaceVariables": surface_variables,
        "writeTecplotSurfaceSolution": True,
        "monitorvariables": ["cpu", "cl", "cd", "yplus"],
        # Physics Parameters
        "equationType": "RANS",
        "smoother": "DADI",
        "rkreset": True,
        "nrkreset": 20,
        # NK Options (ANK does the heavy lifting; NK finishes near convergence)
        "useNKSolver": True,
        "nkswitchtol": 1e-5,
        # ANK Options
        "useanksolver": True,
        "ankswitchtol": 1.0,
        "liftIndex": 2,
        "infchangecorrection": True,
        "nsubiterturb": 10,
        # Convergence Parameters
        "L2Convergence": 1e-8,
        "L2ConvergenceCoarse": 1e-4,
        # Adjoint Parameters
        "adjointSolver": "GMRES",
        "adjointL2Convergence": 1e-8,
        "ADPC": False,
        "adjointMaxIter": 1500,
        "adjointSubspaceSize": 150,
        "ILUFill": 3,
        "useQCR": True,
    }
    # Internal per-Mach schedule (CFL, nCycles, ANK tolerances), then user overrides.
    aero_options.update(ank_schedule(mach))
    if args.solver_options:
        aero_options.update(args.solver_options)

    # Create solver
    cfd_solver = ADFLOW(options=aero_options, comm=comm)

    # ======================================================================
    #         Set up flow conditions with AeroProblem
    # ======================================================================
    # 2D sectional reference (chord * unit span = 1.0) -> conventional 2D cl/cd.
    if use_altitude:
        ap = AeroProblem(
            name="fc",
            alpha=alpha,
            mach=mach,
            altitude=altitude,
            areaRef=1.0,
            chordRef=1.0,
            evalFuncs=["cl", "cd"],
        )
    else:
        ap = AeroProblem(
            name="fc",
            alpha=alpha,
            mach=mach,
            T=temperature,
            reynolds=reynolds,
            reynoldsLength=reynolds_length,
            areaRef=1.0,
            chordRef=1.0,
            evalFuncs=["cl", "cd"],
        )

    # Angle of attack as a trim design variable
    ap.addDV("alpha", value=alpha, lower=-1.0, upper=10.0, scale=0.5)

    # ======================================================================
    #         Geometric Design Variable Set-up
    # ======================================================================
    dv_geo = DVGeometry(args.ffd_fname)
    # Widened FFD shape bounds (+/-0.10 chord) -- the v1 headline change.
    dv_geo.addLocalDV("shape", lower=-0.10, upper=0.10, axis="y", scale=1.0)

    # Slice the section so fc_*_slices.dat is written (the canonical per-design label).
    span = 1.0
    cfd_solver.addSlices("z", np.array([0.5]) * span, sliceType="absolute")

    cfd_solver.setDVGeo(dv_geo)

    # ======================================================================
    #         DVConstraint Setup
    # ======================================================================
    dv_con = DVConstraints()
    dv_con.setDVGeo(dv_geo)
    dv_con.setSurface(cfd_solver.getTriangulatedMeshSurface())

    # Le/Te constraints: keep leading/trailing-edge FFD points moving in
    # equal-and-opposite pairs so the optimizer cannot shear the TE/LE open.
    l_index = dv_geo.getLocalIndex(0)
    ind_set_a = []
    ind_set_b = []
    for k in range(l_index.shape[2]):
        ind_set_a.append(l_index[0, 0, k])
        ind_set_b.append(l_index[0, 1, k])
    for k in range(l_index.shape[2]):
        ind_set_a.append(l_index[-1, 0, k])
        ind_set_b.append(l_index[-1, 1, k])
    dv_con.addLeTeConstraints(0, indSetA=ind_set_a, indSetB=ind_set_b)

    # Span-linking: force the two spanwise FFD planes to move identically so the
    # thin extruded section stays a true 2D airfoil.
    l_index = dv_geo.getLocalIndex(0)
    ind_set_a = []
    ind_set_b = []
    for i in range(l_index.shape[0]):
        ind_set_a.append(l_index[i, 0, 0])
        ind_set_b.append(l_index[i, 0, 1])
    for i in range(l_index.shape[0]):
        ind_set_a.append(l_index[i, 1, 0])
        ind_set_b.append(l_index[i, 1, 1])
    dv_con.addLinearConstraintsShape(ind_set_a, ind_set_b, factorA=1.0, factorB=-1.0, lower=0, upper=0)

    # The blunt section reaches x=0.98, so the TE probe sits just inboard at 0.978 so
    # the rays stay embedded; LE probe at x=0.01. Offset 0.0025 off each symmetry plane.
    z_off = 0.0025
    te_stop = 0.978
    le_list = [[0.01, 0, z_off], [0.01, 0, span - z_off]]
    te_list = [[te_stop, 0, z_off], [te_stop, 0, span - z_off]]

    dv_con.addVolumeConstraint(
        le_list,
        te_list,
        2,
        100,
        lower=area_ratio_min * area_initial / area_input_design,
        upper=1.2 * area_initial / area_input_design,
        scaled=True,
    )
    dv_con.addThicknessConstraints2D(le_list, te_list, 2, 100, lower=0.15, upper=3.0)
    # Keep TE thickness at original or greater.
    dv_con.addThicknessConstraints1D(ptList=te_list, nCon=2, axis=[0, 1, 0], lower=1.0, scaled=True)

    if comm.rank == 0:
        dv_con.writeTecplot(os.path.join(args.output_dir, "constraints.dat"))

    # ======================================================================
    #         Mesh Warping Set-up
    # ======================================================================
    mesh = USMesh(options={"gridFile": args.mesh_fname, "LdefFact": 1000.0, "zeroCornerRotations": True}, comm=comm)

    # Track the absolute (unscaled) area for post-processing parity.
    dv_con_vol = DVConstraints()
    dv_con_vol.setDVGeo(dv_geo)
    dv_con_vol.setSurface(cfd_solver.getTriangulatedMeshSurface())
    dv_con_vol.addVolumeConstraint(le_list, te_list, 2, 100, lower=0.0, upper=1e6, scaled=False, name="vol_abs")

    cfd_solver.setMesh(mesh)

    # ======================================================================
    #         Optimization callbacks
    # ======================================================================
    def cruise_funcs(x):
        if MPI.COMM_WORLD.rank == 0:
            print(x)
        dv_geo.setDesignVars(x)
        ap.setDesignVars(x)
        cfd_solver(ap)
        funcs = {}
        dv_con.evalFunctions(funcs)
        cfd_solver.evalFunctions(ap, funcs)
        cfd_solver.checkSolutionFailure(ap, funcs)
        if MPI.COMM_WORLD.rank == 0:
            print(funcs)
        return funcs

    def cruise_funcs_sens(_x, _funcs):
        funcs_sens = {}
        dv_con.evalFunctionsSens(funcs_sens)
        cfd_solver.evalFunctionsSens(ap, funcs_sens)
        cfd_solver.checkAdjointFailure(ap, funcs_sens)
        if MPI.COMM_WORLD.rank == 0:
            print(funcs_sens)
        return funcs_sens

    def obj_con(funcs, print_ok):
        funcs["obj"] = funcs[ap["cd"]]
        funcs["cl_con_" + ap.name] = funcs[ap["cl"]] - mycl
        if print_ok:
            print("funcs in obj:", funcs)
        return funcs

    # ======================================================================
    #         Optimization Problem Set-up
    # ======================================================================
    opt_prob = Optimization("opt", mp.obj, comm=MPI.COMM_WORLD)
    opt_prob.addObj("obj", scale=1e1)
    ap.addVariablesPyOpt(opt_prob)
    dv_geo.addVariablesPyOpt(opt_prob)
    dv_con.addConstraintsPyOpt(opt_prob)
    # Lift as a one-sided inequality: cl >= cl_target with a generous upper guardrail.
    opt_prob.addCon("cl_con_" + ap.name, lower=0.0, upper=1.2 * mycl, scale=0.5)

    mp.setProcSetObjFunc("cruise", cruise_funcs)
    mp.setProcSetSensFunc("cruise", cruise_funcs_sens)
    mp.setObjCon(obj_con)
    mp.setOptProb(opt_prob)
    opt_prob.printSparsity()

    # Set up optimizer
    if opt == Algorithm.SLSQP:
        opt_options = {"IFILE": os.path.join(args.output_dir, "SLSQP.out")}
    elif opt == Algorithm.SNOPT:
        opt_options = {
            "Major feasibility tolerance": 1e-4,
            "Major optimality tolerance": 1e-4,
            "Hessian full memory": None,
            "Function precision": 1e-8,
            "Print file": os.path.join(args.output_dir, "SNOPT_print.out"),
            "Summary file": os.path.join(args.output_dir, "SNOPT_summary.out"),
        }
    else:  # IPOPT (default for v1)
        opt_options = ipopt_options(args.output_dir)
    # User-supplied overrides win.
    opt_options.update(args.opt_options or {})
    optimizer = OPT(opt.name, options=opt_options)

    # Run Optimization
    sol = optimizer(
        opt_prob,
        mp.sens,
        sensMode="pgc",
        sensStep=1e-6,
        storeHistory=os.path.join(args.output_dir, "opt.hst"),
        storeSens=True,
    )
    if MPI.COMM_WORLD.rank == 0:
        print(sol)
        # Final absolute area (debugging / parity artifact).
        funcs_final = {}
        dv_con_vol.evalFunctions(funcs_final)
        np.save(os.path.join(args.output_dir, "final_abs_volume.npy"), np.array(list(funcs_final.values())))
        # Archive per-iteration files (keep baseline + optimized loose).
        _archive_intermediates(args.output_dir)

    MPI.COMM_WORLD.Barrier()


if __name__ == "__main__":
    main()
