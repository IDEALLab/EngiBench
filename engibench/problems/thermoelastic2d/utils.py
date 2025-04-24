"""Utility functions for the thermoelastic2d problem."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def get_res_bounds(x_res: int, y_res: int) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Generates the indices corresponding to the left, top, right, and bottom elements in the domain.

    Args:
        x_res: The number of elements in the x-direction
        y_res: The number of elements in the y-direction

    Returns:
        The indices corresponding to the left, top, right, and bottom elements in the domain

    """
    row_elements = x_res
    col_elements = y_res

    bottom_row_indices = np.arange(col_elements - 1, row_elements * col_elements, col_elements)
    right_col_indices = np.arange((row_elements - 1) * col_elements, row_elements * col_elements)
    top_row_indices = np.arange(0, row_elements * col_elements, col_elements)
    left_col_indices = np.arange(0, col_elements, 1)

    return left_col_indices, top_row_indices, right_col_indices, bottom_row_indices


def binary_matrix_to_indices(matrix: npt.NDArray) -> npt.NDArray:
    """Flattens a binary matrix and returns the indices where the matrix is 1.

    Args:
        matrix (npt.NDArray): The binary matrix.

    Returns:
        npt.NDArray: The list of indices where the matrix is 1.
    """
    matrix_flat = matrix.flatten()
    return np.where(matrix_flat == 1)[0]


def indices_to_binary_matrix(indices: list[int], nelx: int, nely: int) -> npt.NDArray[np.int64]:
    """Converts a list of indices to a binary matrix of a specific size.

    Args:
        indices (list[int]): The list of indices to set to 1 in the binary matrix.
        nelx (int): Number of elements in the x-direction.
        nely (int): Number of elements in the y-direction.

    Returns:
        npt.NDArray: The binary matrix.
    """
    flat_matrix = np.zeros((nelx * nely,), dtype=int)
    flat_matrix[indices] = 1
    return flat_matrix.reshape((nelx, nely))


def plot_multi_physics(  # noqa: PLR0913, PLR0915
    design: npt.NDArray,
    structural_bcs: npt.NDArray,
    thermal_bcs: npt.NDArray,
    nelx: int,
    nely: int,
    _fp: npt.NDArray | None = None,
    _um: npt.NDArray | None = None,
    _t: npt.NDArray | None = None,
    *,
    open_plot: bool = False,
) -> Figure:
    """Plot the multi-physics design along with the boundary conditions.

    Args:
        design (npt.NDArray): The design array.
        structural_bcs (npt.NDArray): Structural boundary conditions.
        thermal_bcs (npt.NDArray): Thermal boundary conditions.
        nelx (int): Number of elements in the x-direction.
        nely (int): Number of elements in the y-direction.
        _fp (Optional[npt.NDArray]): Force points (default: None).
        _um (None): Unused parameter (default: None).
        _t (None): Unused parameter (default: None).
        open_plot (bool): Whether to open the plot (default: False).

    Returns:
        fig (Figure): The figure generated.
    """
    x_elements = nelx + 1
    y_elements = nely + 1

    # Get even and odd Fp elements
    if _fp is None:
        _fp = np.zeros((x_elements * y_elements * 2,))
    fp_x = _fp[::2]  # 8450 / 2 = 4225 = 65 * 65
    fp_y = _fp[1::2]

    if _um is None:
        _um = np.zeros((x_elements * y_elements * 2,))

    left_col_indices, top_row_indices, right_col_indices, bottom_row_indices = get_res_bounds(x_elements, y_elements)

    structural_bcs_img: npt.NDArray[np.float64] = np.zeros((x_elements * y_elements,))
    structural_bcs_img[structural_bcs // 2] = 1
    structural_bcs_img = structural_bcs_img.reshape((x_elements, y_elements))
    structural_bcs_img_clip = np.clip(structural_bcs_img * 127.5 + 127.5, 0.0, 255.0).astype(np.uint8)

    structural_bcs_img_clip = structural_bcs_img_clip.T  # transpose to flip bottom left and top right

    thermal_bcs_img: npt.NDArray[np.float64] = np.zeros((x_elements * y_elements,))
    thermal_bcs_img[thermal_bcs] = 1
    thermal_bcs_img[right_col_indices] = 1
    thermal_bcs_img = thermal_bcs_img.reshape((x_elements, y_elements))
    thermal_bcs_img_clip = np.clip(thermal_bcs_img * 127.5 + 127.5, 0.0, 255.0).astype(np.uint8)

    thermal_bcs_img_clip = thermal_bcs_img_clip.T  # transpose to flip bottom left and top right

    fpx_img = fp_x.reshape((x_elements, y_elements))
    fpx_img_clip = np.clip(fpx_img * 127.5 + 127.5, 0.0, 255.0).astype(np.uint8)

    fpx_img_clip = fpx_img_clip.T  # transpose to flip bottom left and top right

    fpy_img = fp_y.reshape((x_elements, y_elements))
    fpy_img_clip = np.clip(fpy_img * 127.5 + 127.5, 0.0, 255.0).astype(np.uint8)
    fpy_img_clip = fpy_img_clip.T  # transpose to flip bottom left and top right

    # Create Plots
    fig, ax = plt.subplots(2, 4, figsize=(7, 5))
    ax[0][0].imshow(structural_bcs_img_clip)
    ax[0][0].axis("off")
    ax[0][0].set_title("Structural BCs")

    ax[0][1].imshow(fpx_img_clip)
    ax[0][1].axis("off")
    ax[0][1].set_title("Force X")

    ax[0][2].imshow(fpy_img_clip)
    ax[0][2].axis("off")
    ax[0][2].set_title("Force Y")

    ax[0][3].imshow(thermal_bcs_img_clip)
    ax[0][3].axis("off")
    ax[0][3].set_title("Thermal BCs")

    ax[1][0].imshow(-design, cmap="gray", interpolation="none", norm=colors.Normalize(vmin=-1, vmax=0))
    ax[1][0].axis("off")
    ax[1][0].set_title("Design")

    if _t is None:
        ax[1][1].axis("off")
    else:
        t_img = _t.reshape((x_elements, y_elements))
        t_img_clip = t_img.T  # transpose to flip bottom left and top right
        # normalize temperature
        t_img_clip = (t_img_clip - np.min(t_img_clip)) / (np.max(t_img_clip) - np.min(t_img_clip)) * 255
        t_xxx = np.arange(nelx + 1)
        t_yyy = -np.arange(nely + 1)
        ax[1][1].contourf(t_xxx, t_yyy, t_img_clip, 50)
        ax[1][1].axis("image")
        ax[1][1].set_title("Temperature")

    # Hide axis on the last subplots
    ax[1][2].axis("off")
    ax[1][3].axis("off")

    plt.tight_layout()
    if open_plot is True:
        plt.show()
    return fig
