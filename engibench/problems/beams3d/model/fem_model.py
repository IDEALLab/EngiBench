"""Structural-only Python implementation of the Beams 3D problem."""

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import numpy.typing as npt

from engibench.core import OptiStep
from engibench.problems.beams3d.model.fem_matrix_builder import fe_mech_3d
from engibench.problems.beams3d.model.fem_structural_setup import build_structural_context_3d
from engibench.problems.beams3d.model.fem_structural_setup import fe_structural_bc_3d
from engibench.problems.beams3d.model.mma_subroutine import MMAInputs
from engibench.problems.beams3d.model.mma_subroutine import mmasub

SECOND_ITERATION_THRESHOLD = 2
FIRST_ITERATION_THRESHOLD = 1
MIN_ITERATIONS = 10
MAX_ITERATIONS = 200
UPDATE_THRESHOLD = 0.01


def _filter_offsets(rmin: float) -> list[tuple[int, int, int, float]]:
    rceil = int(np.ceil(rmin)) - 1
    deltas = np.arange(-rceil, rceil + 1)
    dx_grid, dy_grid, dz_grid = np.meshgrid(deltas, deltas, deltas, indexing="ij")
    distances = np.linalg.norm(np.stack((dx_grid, dy_grid, dz_grid), axis=-1), axis=-1)
    weights = rmin - distances
    active = weights > 0.0

    return [
        (int(dx), int(dy), int(dz), float(weight))
        for dx, dy, dz, weight in zip(
            dx_grid[active],
            dy_grid[active],
            dz_grid[active],
            weights[active],
            strict=True,
        )
    ]


def _valid_filter_offset(shape: tuple[int, int, int], offset: tuple[int, int, int, float]) -> bool:
    return all(abs(delta) < size for delta, size in zip(offset[:3], shape, strict=True))


def _count_filter_entries(shape: tuple[int, int, int], offsets: tuple[tuple[int, int, int, float], ...]) -> int:
    nnz = 0
    for dx, dy, dz, _ in offsets:
        nnz += (shape[0] - abs(dx)) * (shape[1] - abs(dy)) * (shape[2] - abs(dz))
    return nnz


@dataclass(frozen=True)
class SensitivityFilter3D:
    """Matrix-free cone-filter data."""

    offsets: tuple[tuple[int, int, int, float], ...]
    hs: npt.NDArray[np.float64]
    shape: tuple[int, int, int]


def _offset_slices(size: int, delta: int) -> tuple[slice, slice]:
    dst_start = max(0, -delta)
    dst_stop = min(size, size - delta)
    return slice(dst_start, dst_stop), slice(dst_start + delta, dst_stop + delta)


def _apply_filter(
    values: npt.NDArray[np.float64],
    sensitivity_filter: SensitivityFilter3D,
) -> npt.NDArray[np.float64]:
    axis0, axis1, axis2 = sensitivity_filter.shape
    values_3d = values.reshape(axis0, axis1, axis2)
    filtered = np.zeros_like(values_3d, dtype=np.float64)

    for daxis0, daxis1, daxis2, weight in sensitivity_filter.offsets:
        dst_axis0, src_axis0 = _offset_slices(axis0, daxis0)
        dst_axis1, src_axis1 = _offset_slices(axis1, daxis1)
        dst_axis2, src_axis2 = _offset_slices(axis2, daxis2)
        filtered[dst_axis0, dst_axis1, dst_axis2] += weight * values_3d[src_axis0, src_axis1, src_axis2]

    return filtered.reshape(-1, 1) / sensitivity_filter.hs[:, None]


class FeaModel3D:
    """Finite Element Analysis model for structural 3D topology optimization."""

    def __init__(self, *, eval_only: bool = False, max_iter: int = MAX_ITERATIONS) -> None:
        """Instantiates a new structural 3D model.

        Args:
            eval_only: If True, evaluate the given design once and return objective components only.
            max_iter: Maximal number of iterations for the `run` method.
        """
        self.eval_only = eval_only
        self.max_iter = max_iter

    def has_converged(self, change: float, iterr: int) -> bool:
        """Return True when the optimizer should stop before starting another iteration."""
        if iterr >= self.max_iter:
            return True
        return change < UPDATE_THRESHOLD and iterr >= MIN_ITERATIONS

    def get_initial_design(self, volume_fraction: float, nelx: int, nely: int, nelz: int) -> np.ndarray:
        """Generates the initial design variable field for the optimization process.

        Args:
            volume_fraction (float): The initial volume fraction for the material distribution.
            nelx (int): Number of elements in the x-direction.
            nely (int): Number of elements in the y-direction.
            nelz (int): Number of elements in the z-direction.

        Returns:
            np.ndarray: A 3D NumPy array of shape (nely, nelx, nelz) initialized with the given volume fraction.
        """
        return volume_fraction * np.ones((nely, nelx, nelz), dtype=float)

    def get_matrices(self, nu: float, e: float) -> np.ndarray:
        """Computes and returns the structural element stiffness matrix.

        Args:
            nu (float): Poisson's ratio.
            e (float): Young's modulus (modulus of elasticity).

        Returns:
            The stiffness matrix for mechanical analysis.
        """
        return fe_mech_3d(nu, e)

    def get_filter(self, nelx: int, nely: int, nelz: int, rmin: float) -> SensitivityFilter3D:
        """Constructs matrix-free sensitivity filter data.

        The filter helps mitigate checkerboarding issues in topology optimization by averaging
        sensitivities over neighboring elements.

        Args:
            nelx (int): Number of elements in the x-direction.
            nely (int): Number of elements in the y-direction.
            nelz (int): Number of elements in the z-direction.
            rmin (float): Minimum filter radius.

        Returns:
            Matrix-free cone-filter offsets and normalization factors.
        """
        filter_shape = (nely, nelx, nelz)
        n = nelx * nely * nelz
        offsets = tuple(offset for offset in _filter_offsets(rmin) if _valid_filter_offset(filter_shape, offset))
        nnz = _count_filter_entries(filter_shape, offsets)
        if nnz <= 0:
            raise RuntimeError("3D filter has no active entries.")

        hs_3d = np.zeros(filter_shape, dtype=np.float64)
        for daxis0, daxis1, daxis2, weight in offsets:
            dst_axis0, _src_axis0 = _offset_slices(filter_shape[0], daxis0)
            dst_axis1, _src_axis1 = _offset_slices(filter_shape[1], daxis1)
            dst_axis2, _src_axis2 = _offset_slices(filter_shape[2], daxis2)
            hs_3d[dst_axis0, dst_axis1, dst_axis2] += weight

        hs = hs_3d.reshape(n)
        return SensitivityFilter3D(offsets=offsets, hs=hs, shape=filter_shape)

    def run(  # noqa: PLR0915
        self,
        bcs: dict[str, Any],
        x_init: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run structural topology optimization with no thermal assembly or solves."""
        fixed_nodes_mask = np.asarray(bcs["fixed_elements"], dtype=bool)

        nxp, nyp, nzp = fixed_nodes_mask.shape
        nelx, nely, nelz = nxp - 1, nyp - 1, nzp - 1
        n = nelx * nely * nelz

        volfrac = bcs["volfrac"]
        opti_steps: list[OptiStep] = []

        x = self.get_initial_design(volfrac, nelx, nely, nelz) if x_init is None else x_init.copy()

        penal = bcs.get("penal", 3.0)
        rmin = bcs.get("rmin", 1.1)
        e = 1.0
        nu = 0.3
        change = 1.0
        iterr = 0
        xmin, xmax = 1e-3, 1.0
        xold1 = x.reshape(n, 1)
        xold2 = x.reshape(n, 1)
        m = 1
        a0 = 1.0
        a = np.zeros((m, 1))
        c = 10000.0 * np.ones((m, 1))
        d = np.zeros((m, 1))
        low = xmin
        upp = xmax

        low_vec = None
        upp_vec = None

        ke = self.get_matrices(nu, e)
        context = build_structural_context_3d(nely, nelx, nelz, bcs)

        sensitivity_filter: SensitivityFilter3D | None = None
        if not self.eval_only:
            t_filter_build_start = time.time()
            sensitivity_filter = self.get_filter(nelx, nely, nelz, rmin)
            print(f"3D sensitivity filter built in {time.time() - t_filter_build_start:.3f}s.")

        change_evol = []
        obj_evol = []
        f0valm = 0.0

        while not self.has_converged(change, iterr):
            iterr += 1
            t0 = time.time()
            tcur = t0

            res = fe_structural_bc_3d(nely, nelx, nelz, penal, x, ke, bcs, context=context)
            um = res.um

            t_forward = time.time() - tcur
            tcur = time.time()

            um_e = um[res.edof24]
            element_energy = np.einsum("ij,jk,ik->i", um_e, ke, um_e, optimize=True)
            densities = x[res.ey, res.ex, res.ez]
            density_power = densities**penal
            density_derivative = penal * (densities ** (penal - 1))

            f0valm = float(np.sum(density_power * element_energy))
            df0dx_vec_raw = -density_derivative * element_energy
            df0dx_m = np.zeros_like(x)
            df0dx_m[res.ey, res.ex, res.ez] = df0dx_vec_raw
            f0val = f0valm

            if self.eval_only:
                vf_error = abs(np.mean(x) - volfrac)
                return {
                    "structural_compliance": float(f0valm),
                    "volume_fraction": vf_error,
                }

            obj_values = np.array([f0valm])
            x_curr = x.copy()

            xval = x.reshape(n, 1)
            volconst = np.sum(x) / (volfrac * n) - 1.0
            fval = volconst
            dfdx = np.ones((1, n), dtype=float) / (volfrac * n)

            df0dx_vec = df0dx_m.reshape(n, 1)
            if sensitivity_filter is None:
                raise RuntimeError("Sensitivity filter is required for optimization.")
            df0dx_filt = _apply_filter(xval * df0dx_vec, sensitivity_filter) / np.maximum(1e-3, xval)

            t_sens = time.time() - tcur
            tcur = time.time()

            if low_vec is None or upp_vec is None:
                upp_vec = np.ones((n,), dtype=float) * upp
                low_vec = np.ones((n,), dtype=float) * low

            mmainputs = MMAInputs(
                m=1,
                n=n,
                iterr=iterr,
                xval=xval[:, 0],
                xmin=xmin,
                xmax=xmax,
                xold1=xold1,
                xold2=xold2,
                df0dx=df0dx_filt[:, 0],
                fval=fval,
                dfdx=dfdx,
                low=low_vec,
                upp=upp_vec,
                a0=a0,
                a=a[0],
                c=c[0],
                d=d[0],
                f0val=f0val,
            )
            xmma, low_vec, upp_vec = mmasub(mmainputs)

            low_vec = np.squeeze(low_vec)
            upp_vec = np.squeeze(upp_vec)

            if iterr > SECOND_ITERATION_THRESHOLD:
                xold2 = xold1
                xold1 = xval
            elif iterr > FIRST_ITERATION_THRESHOLD:
                xold1 = xval

            x = xmma.reshape(nely, nelx, nelz)
            x_update = x.copy() - x_curr
            df0dx_all = np.stack([df0dx_m, dfdx.reshape(nely, nelx, nelz)], axis=0)
            if opti_steps:
                opti_steps[-1].obj_values_update = obj_values.copy() - opti_steps[-1].obj_values
            opti_steps.append(
                OptiStep(
                    obj_values=obj_values,
                    step=iterr,
                    x=x_curr,
                    x_sensitivities=df0dx_all,
                    x_update=x_update,
                )
            )

            change = np.max(np.abs(xmma - xold1))
            change_evol.append(change)
            obj_evol.append(f0val)

            t_mma = time.time() - tcur
            t_total = time.time() - t0
            print(
                f" It.: {iterr:4d} Obj.: {f0val:10.4f} "
                f"Vol.: {np.sum(x) / (nelx * nely * nelz):6.3f} ch.: {change:6.3f} "
                f"|| t_forward:{t_forward:6.3f} + t_sens:{t_sens:6.3f} + t_mma:{t_mma:6.3f} = {t_total:6.3f}"
            )

        if opti_steps and opti_steps[-1].obj_values_update is None:
            opti_steps[-1].obj_values_update = np.zeros_like(opti_steps[-1].obj_values)

        print("3D structural optimization finished.")
        vf_error = abs(np.mean(x) - volfrac)

        return {
            "design": x,
            "bcs": bcs,
            "structural_compliance": float(f0valm),
            "volume_fraction": vf_error,
            "opti_steps": opti_steps,
        }
