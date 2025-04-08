# engibench/problems/photonics2d/backend.py

"""Backend helper functions for the Photonics2D problem.
Includes domain setup, parameterization, operators, and simulation utilities.

This is essentially re-factored from the code at
https://nbviewer.org/github/fancompute/workshop-invdesign/blob/master/04_Invdes_wdm_scheduling.ipynb

Author: Mark Fuge @markfuge
"""  # noqa: D205

import collections

import autograd.numpy as npa
from autograd.scipy.signal import convolve as conv_npa  # Use autograd's convolve
import ceviche.modes
import numpy as np
import numpy.typing as npt
from skimage.draw import disk as circle

# --- Data Structures ---
# Container for slice coordinates used for sources and probes
Slice = collections.namedtuple("Slice", "x y")


# --- Domain Initialization ---


def init_domain(  # noqa: PLR0913
    num_elems_x: int, num_elems_y: int, num_elems_pml: int, space: int, wg_width: int, space_slice: int, wg_shift: int
) -> tuple[npt.NDArray, npt.NDArray, Slice, Slice, Slice]:
    """Initializes the background geometry, design region mask, and source/probe slices.

    Args:
        num_elems_x: Number of cells in x-direction.
        num_elems_y: Number of cells in y-direction.
        num_elems_pml: Number of PML cells.
        space: Space between PML and design region (pixels).
        wg_width: Width of input/output waveguides (pixels).
        space_slice: Extra width for source/probe slices (pixels).
        wg_shift: Lateral shift for output waveguides (pixels).

    Returns:
        Tuple containing:
        - bg_rho: Background material distribution (waveguides).
        - design_region: Mask indicating the optimizable design area (1s inside, 0s outside).
        - input_slice: Slice object for the input waveguide source/probe.
        - output_slice1: Slice object for the first output waveguide probe.
        - output_slice2: Slice object for the second output waveguide probe.
    """
    # Initialize arrays
    bg_rho = np.zeros((num_elems_x, num_elems_y))
    design_region = np.zeros((num_elems_x, num_elems_y))

    # --- Define Input Waveguide and Slice ---
    in_wg_y_min = num_elems_y // 2 - wg_width // 2
    in_wg_y_max = num_elems_y // 2 + wg_width // 2
    bg_rho[0 : num_elems_pml + space, in_wg_y_min:in_wg_y_max] = 1.0  # Use 1.0 for float

    # Define input slice coordinates slightly inside PML
    input_slice_x = num_elems_pml + 1  # Consistent slice position
    input_slice_y_min = in_wg_y_min - space_slice
    input_slice_y_max = in_wg_y_max + space_slice
    input_slice = Slice(x=np.array(input_slice_x), y=np.arange(input_slice_y_min, input_slice_y_max))

    # --- Define Output Waveguide 1 and Slice ---
    out_wg1_y_min = num_elems_pml + space + wg_shift
    out_wg1_y_max = num_elems_pml + space + wg_width + wg_shift
    bg_rho[int(num_elems_x - num_elems_pml - space) :, out_wg1_y_min:out_wg1_y_max] = 1.0

    # Define output slice 1 coordinates slightly inside PML
    output_slice1_x = num_elems_x - num_elems_pml - 1
    output_slice1_y_min = out_wg1_y_min - space_slice
    output_slice1_y_max = out_wg1_y_max + space_slice
    output_slice1 = Slice(x=np.array(output_slice1_x), y=np.arange(output_slice1_y_min, output_slice1_y_max))

    # --- Define Output Waveguide 2 and Slice ---
    out_wg2_y_min = num_elems_y - num_elems_pml - space - wg_width - wg_shift
    out_wg2_y_max = num_elems_y - num_elems_pml - space - wg_shift
    bg_rho[int(num_elems_x - num_elems_pml - space) :, out_wg2_y_min:out_wg2_y_max] = 1.0

    # Define output slice 2 coordinates slightly inside PML
    output_slice2_x = num_elems_x - num_elems_pml - 1
    output_slice2_y_min = out_wg2_y_min - space_slice
    output_slice2_y_max = out_wg2_y_max + space_slice
    output_slice2 = Slice(x=np.array(output_slice2_x), y=np.arange(output_slice2_y_min, output_slice2_y_max))

    # --- Define Design Region ---
    design_region[
        num_elems_pml + space : num_elems_x - num_elems_pml - space,
        num_elems_pml + space : num_elems_y - num_elems_pml - space,
    ] = 1.0

    # Return only geometry, rho initialization is handled by random_design
    return bg_rho, design_region, input_slice, output_slice1, output_slice2


# --- Mathematical Operators (using autograd.numpy for differentiability) ---


def operator_proj(rho: npt.NDArray, eta: float = 0.5, beta: float = 100, num_projections: int = 1) -> npt.NDArray:
    """Density projection operator using tanh functions. Drives rho towards 0 or 1.

    Args:
        rho: Input density array.
        eta: Center of the projection (typically 0.5).
        beta: Strength of the projection. Higher values lead to sharper transitions.
        num_projections: Number of times to apply the projection.

    Returns:
        Projected density array.
    """
    # Use autograd numpy functions for differentiability
    for _ in range(num_projections):
        rho = npa.divide(
            npa.tanh(beta * eta) + npa.tanh(beta * (rho - eta)), npa.tanh(beta * eta) + npa.tanh(beta * (1 - eta))
        )
    return rho


def _create_blur_kernel(radius: int) -> npt.NDArray:
    """Creates a circular convolution kernel using skimage.draw.disk.

    Args:
        radius: The radius of the circular kernel (in pixels).

    Returns:
        A 2D numpy array representing the normalized convolution kernel.
    """
    radius = int(max(1, radius))  # Ensure radius is at least 1
    diameter = 2 * radius + 1
    # Create coordinates for a circle centered in the kernel array
    rr, cc = circle((radius, radius), radius=radius + 0.1)  # Use radius+0.1 to ensure center pixel is included

    # Ensure coordinates are within bounds
    valid = (rr >= 0) & (rr < diameter) & (cc >= 0) & (cc < diameter)
    rr, cc = rr[valid], cc[valid]
    kernel = np.zeros((diameter, diameter), dtype=np.float64)
    kernel[rr, cc] = 1.0

    # Normalize the kernel
    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        return kernel / kernel_sum
    else:
        # Handle case where radius is so small kernel is empty (shouldn't happen with max(1, radius))
        # Return an identity kernel (1 in the middle)
        kernel[radius, radius] = 1.0
        return kernel


def operator_blur(rho: npt.NDArray, radius: int = 2, num_blurs: int = 1) -> npt.NDArray:
    """Density blurring operator using 2D convolution with a circular kernel.

    Args:
        rho: Input density array.
        radius: Radius of the blurring kernel. If < 0.5, blurring is skipped.
        num_blurs: Number of times to apply the blur filter.

    Returns:
        Blurred density array.
    """
    min_blur_radius = 0.5
    if radius < min_blur_radius:  # Skip blurring if radius is negligible
        return rho

    kernel = _create_blur_kernel(radius)
    pad_width = int(radius)  # Padding width needed for 'valid' mode convolution

    # Use autograd's convolve for differentiability
    rho_blurred = rho
    for _ in range(num_blurs):
        # --- Corrected Padding Line ---
        # Change mode to 'constant' and specify constant_values (e.g., 0)
        rho_padded = npa.pad(rho_blurred, pad_width, mode="constant", constant_values=0)
        # -----------------------------

        # Perform convolution using 'valid' mode, which handles boundary automatically
        rho_blurred = conv_npa(rho_padded, kernel, mode="valid")

    return rho_blurred


# --- Parameterization ---


def wavelength_to_frequency(wavelength: float) -> float:
    """Converts wavelength (in micrometers) to frequency (in Hz)."""
    c0 = 299792458  # Speed of light in vacuum (m/s)
    return (c0 * 2 * np.pi * 1e6) / wavelength


def mask_combine_rho(rho: npt.NDArray, bg_rho: npt.NDArray, design_region: npt.NDArray) -> npt.NDArray:
    """Combines the design rho with the background rho using the design region mask.

    Args:
        rho: Density array for the design region.
        bg_rho: Density array for the background (e.g., waveguides).
        design_region: Mask (1s for design area, 0s for background).

    Returns:
        Combined density array.
    """
    # Ensure float math, especially important if design_region is int
    design_region_f = design_region.astype(float)
    return rho * design_region_f + bg_rho * (1.0 - design_region_f)


def epsr_parameterization(  # noqa: PLR0913
    rho: npt.NDArray,
    bg_rho: npt.NDArray,
    design_region: npt.NDArray,
    radius: int,
    num_blurs: int,
    beta: float,
    eta: float,
    num_projections: int,
    epsr_min: float,
    epsr_max: float,
) -> npt.NDArray:
    """Applies filtering and projection to rho and converts to permittivity.

    Implements the standard rho -> blur -> project -> mask -> eps_r
    parameterization chain using differentiable operators (autograd compatible).

    Args:
        rho: Input design density (typically 0-1).
        bg_rho: Background density (waveguides).
        design_region: Design region mask.
        radius: Blur kernel radius.
        num_blurs: Number of blur applications.
        beta: Projection strength.
        eta: Projection center.
        num_projections: Number of projection applications.
        epsr_min: Minimum relative permittivity.
        epsr_max: Maximum relative permittivity.

    Returns:
        The calculated relative permittivity distribution array (eps_r).
    """
    # Combine rho and bg_rho: Blurring sees the combined structure initially
    rho_combined = mask_combine_rho(rho, bg_rho, design_region)

    # Apply blur operator
    rho_blurred = operator_blur(rho_combined, radius=radius, num_blurs=num_blurs)

    # Apply projection operator
    # Apply to the blurred version of the combined structure
    rho_projected = operator_proj(rho_blurred, beta=beta, eta=eta, num_projections=num_projections)

    # Final masking: Restore the exact background structure and apply projection
    # result only within the design region. This prevents blurring/projection
    # artifacts from affecting the fixed background parts (like waveguides).
    rho_final = mask_combine_rho(rho_projected, bg_rho, design_region)

    # Convert final density to permittivity using linear interpolation
    return epsr_min + (epsr_max - epsr_min) * rho_final


# --- Simulation Utilities ---


def mode_overlap(elec_field1: npt.NDArray, elec_field2: npt.NDArray) -> npt.NDArray:
    """Calculates the mode overlap integral: |sum(conj(elec_field1) * elec_field2)|.

    Uses autograd.numpy for differentiability during optimization.

    Args:
        elec_field1: Complex electric field array 1.
        elec_field2: Complex electric field array 2.

    Returns:
        Scalar value of the overlap integral magnitude, scaled by 1e6 (from source).
    """
    # Use autograd numpy for sum and abs as this is used in the objective function
    overlap = npa.sum(npa.conj(elec_field1) * elec_field2)
    return npa.abs(overlap) * 1e6  # Scaling factor from source


def insert_mode(*args, **kwargs) -> npt.NDArray:  # noqa: ANN002
    """Wrapper for ceviche.modes.insert_mode.

    Passes arguments directly to the ceviche function. This allows using
    a consistent backend function name.

    Returns:
        The source or probe field array generated by ceviche.
    """
    return ceviche.modes.insert_mode(*args, **kwargs)
