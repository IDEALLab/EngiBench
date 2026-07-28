"""Conversions between MTO2D density arrays and OpenFOAM gamma fields.

The EngiBench-native representation is the non-redundant, visually oriented
``(400, 200)`` half-domain. OpenFOAM stores that domain in two blocks, followed
by 6,400 fixed/non-design cells:

* values ``0:64000`` form a ``(400, 160)`` block;
* values ``64000:80000`` form a ``(400, 40)`` block;
* values ``80000:86400`` are fixed cells copied from a case template.

The solver blocks are vertically flipped to obtain the visual orientation used
by EngiBench. Each legacy published ``(256, 256)`` array is the entire native
half-domain anisotropically resized to a square, not a full mirrored image.
The legacy ``gamma_npy.py`` helper separately mirrors that half-domain into a
``(400, 400)`` tensor for visualization and conversion; its returned tensor is
not the storage convention of the published ``256 x 256`` NumPy designs.
Conversions involving that format reproduce PyTorch's non-antialiased Keys
bicubic interpolation and are explicitly lossy; they must not be used on the
simulator path when exact reproduction is required.
"""

from pathlib import Path
import re

import numpy as np
import numpy.typing as npt

HALF_DESIGN_SHAPE = (400, 200)
"""Shape of the non-redundant, visually oriented EngiBench design."""

FULL_DESIGN_SHAPE = (400, 400)
"""Shape of the mirrored visualization."""

LEGACY_DESIGN_SHAPE = (256, 256)
"""Shape used by the legacy published NumPy dataset."""

FIRST_BLOCK_SHAPE = (400, 160)
SECOND_BLOCK_SHAPE = (400, 40)
DESIGN_CELL_COUNT = 80_000
FIXED_CELL_COUNT = 6_400
GAMMA_CELL_COUNT = DESIGN_CELL_COUNT + FIXED_CELL_COUNT

_FOAM_FILE_RE = re.compile(r"(?P<start>\bFoamFile\s*\{)(?P<body>.*?)(?P<end>\})", flags=re.DOTALL)
_INTERNAL_FIELD_RE = re.compile(
    r"\binternalField\s+nonuniform\s+List\s*<\s*scalar\s*>\s*"
    r"(?P<count>[0-9]+)\s*\(\s*(?P<values>.*?)\s*\)\s*;",
    flags=re.DOTALL,
)
_LOCATION_RE = re.compile(
    r'(?m)^(?P<indent>[ \t]*)location\s+(?:"[^"\r\n]*"|[^;\r\n]+)\s*;',
)


def _validate_density_array(
    design: npt.NDArray,
    expected_shape: tuple[int, int],
    *,
    name: str,
) -> npt.NDArray[np.float32]:
    if not isinstance(design, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if design.dtype != np.dtype(np.float32):
        raise TypeError(f"{name} must have dtype float32; got {design.dtype}")
    if design.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}; got {design.shape}")
    if not np.all(np.isfinite(design)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any((design < 0.0) | (design > 1.0)):
        raise ValueError(f"{name} values must lie in [0, 1]")
    return design


def _validate_gamma_values(values: npt.NDArray) -> npt.NDArray[np.float64]:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"gamma values must have a floating dtype; got {array.dtype}")
    if array.shape != (GAMMA_CELL_COUNT,):
        raise ValueError(f"gamma must contain exactly {GAMMA_CELL_COUNT} values; got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("gamma must contain only finite values")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("gamma values must lie in [0, 1]")
    return np.asarray(array, dtype=np.float64)


def _foam_header(content: str) -> re.Match[str] | None:
    return _FOAM_FILE_RE.search(content)


def _internal_field_match(content: str) -> re.Match[str]:
    header = _foam_header(content)
    if header is not None and re.search(r"\bformat\s+binary\s*;", header.group("body"), flags=re.IGNORECASE):
        raise ValueError("Only ASCII OpenFOAM fields are supported; the FoamFile header declares binary format")

    match = _INTERNAL_FIELD_RE.search(content)
    if match is None:
        raise ValueError("Could not find an ASCII 'internalField nonuniform List<scalar>' declaration")
    return match


def parse_internal_field(content: str, *, expected_count: int | None = None) -> npt.NDArray[np.float64]:
    """Parse an ASCII OpenFOAM nonuniform scalar internal field.

    Args:
        content: Complete OpenFOAM field-file text.
        expected_count: Optional required number of scalar values.

    Returns:
        A one-dimensional float64 array in OpenFOAM storage order.

    Raises:
        ValueError: If the declaration is missing, malformed, non-finite, or
            its declared/expected count does not match the value list.
    """
    if not isinstance(content, str):
        raise TypeError("OpenFOAM content must be text")

    match = _internal_field_match(content)
    declared_count = int(match.group("count"))
    if expected_count is not None and declared_count != expected_count:
        raise ValueError(f"OpenFOAM internalField declares {declared_count} values; expected {expected_count}")

    tokens = match.group("values").split()
    if len(tokens) != declared_count:
        raise ValueError(
            f"OpenFOAM internalField declares {declared_count} values but contains {len(tokens)} scalar tokens"
        )

    try:
        values = np.asarray([float(token) for token in tokens], dtype=np.float64)
    except ValueError as error:
        invalid_token = next((token for token in tokens if not _is_float(token)), "<unknown>")
        raise ValueError(f"OpenFOAM internalField contains an invalid scalar token: {invalid_token!r}") from error

    if not np.all(np.isfinite(values)):
        raise ValueError("OpenFOAM internalField contains non-finite scalar values")
    return values


def _is_float(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def gamma_to_half_design(gamma_values: npt.NDArray) -> npt.NDArray[np.float32]:
    """Convert all 86,400 OpenFOAM gamma values to the native half-domain."""
    gamma = _validate_gamma_values(gamma_values)
    first_block = gamma[: FIRST_BLOCK_SHAPE[0] * FIRST_BLOCK_SHAPE[1]].reshape(FIRST_BLOCK_SHAPE)
    second_block = gamma[FIRST_BLOCK_SHAPE[0] * FIRST_BLOCK_SHAPE[1] : DESIGN_CELL_COUNT].reshape(SECOND_BLOCK_SHAPE)
    solver_oriented = np.concatenate((first_block, second_block), axis=1)
    return np.ascontiguousarray(np.flipud(solver_oriented), dtype=np.float32)


def half_design_to_gamma(
    design: npt.NDArray,
    template_values: npt.NDArray,
) -> npt.NDArray[np.float64]:
    """Convert a native half-domain to gamma order, preserving fixed template cells."""
    half = _validate_density_array(design, HALF_DESIGN_SHAPE, name="design")
    template = _validate_gamma_values(template_values)

    solver_oriented = np.flipud(half)
    design_values = np.concatenate(
        (
            solver_oriented[:, : FIRST_BLOCK_SHAPE[1]].reshape(-1),
            solver_oriented[:, FIRST_BLOCK_SHAPE[1] :].reshape(-1),
        )
    )
    gamma = template.copy()
    gamma[:DESIGN_CELL_COUNT] = design_values
    return gamma


def half_to_full(design: npt.NDArray) -> npt.NDArray[np.float32]:
    """Mirror a native half-domain horizontally to a ``(400, 400)`` field."""
    half = _validate_density_array(design, HALF_DESIGN_SHAPE, name="design")
    return np.ascontiguousarray(np.concatenate((half, np.fliplr(half)), axis=1))


def _validate_symmetry(full: npt.NDArray[np.float32], half_width: int, tolerance: float) -> None:
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("symmetry_tolerance must be a finite, non-negative number")
    difference = np.abs(full[:, :half_width] - np.fliplr(full[:, half_width:]))
    max_difference = float(np.max(difference))
    if max_difference > tolerance:
        raise ValueError(
            f"full design is not horizontally symmetric within tolerance {tolerance}; "
            f"maximum difference is {max_difference}"
        )


def full_to_half(
    design: npt.NDArray,
    *,
    validate_symmetry: bool = True,
    symmetry_tolerance: float = 1e-6,
) -> npt.NDArray[np.float32]:
    """Extract the native left half from a full visualization.

    Args:
        design: Float32 ``(400, 400)`` density array.
        validate_symmetry: Whether to verify that the right half mirrors the
            left half before discarding it.
        symmetry_tolerance: Maximum absolute difference accepted by symmetry
            validation.
    """
    full = _validate_density_array(design, FULL_DESIGN_SHAPE, name="design")
    if validate_symmetry:
        _validate_symmetry(full, HALF_DESIGN_SHAPE[1], symmetry_tolerance)
    return np.ascontiguousarray(full[:, : HALF_DESIGN_SHAPE[1]])


def _cubic_convolution1(value: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    coefficient = np.float32(-0.75)
    return ((coefficient + np.float32(2.0)) * value - (coefficient + np.float32(3.0))) * value * value + np.float32(1.0)


def _cubic_convolution2(value: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    coefficient = np.float32(-0.75)
    return (
        (coefficient * value - np.float32(5.0) * coefficient) * value + np.float32(8.0) * coefficient
    ) * value - np.float32(4.0) * coefficient


def _axis_indices_and_weights(input_size: int, output_size: int) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
    """Build PyTorch-style bicubic indices and weights for one axis."""
    scale = np.float32(input_size / output_size)
    output_coordinates = np.arange(output_size, dtype=np.float64)
    # PyTorch stores the scale as float32 and commonly evaluates this expression
    # with fused multiply-add. Computing with the exact float32 scale in float64
    # and rounding once reproduces that coordinate convention in NumPy.
    source_coordinates = np.asarray(
        (output_coordinates + 0.5) * np.float64(scale) - 0.5,
        dtype=np.float32,
    )
    base_indices = np.floor(source_coordinates).astype(np.int64)
    fractions = np.asarray(source_coordinates - base_indices, dtype=np.float32)

    weights = np.stack(
        (
            _cubic_convolution2(fractions + np.float32(1.0)),
            _cubic_convolution1(fractions),
            _cubic_convolution1(np.float32(1.0) - fractions),
            _cubic_convolution2(np.float32(2.0) - fractions),
        ),
        axis=1,
    )
    offsets = np.arange(-1, 3, dtype=np.int64)
    indices = np.clip(base_indices[:, None] + offsets, 0, input_size - 1)
    return indices, weights


def _weighted_sum_four(samples: npt.NDArray[np.float32], weights: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    terms = samples * weights
    return ((terms[..., 0] + terms[..., 1]) + terms[..., 2]) + terms[..., 3]


def _resize_density(design: npt.NDArray[np.float32], target_shape: tuple[int, int]) -> npt.NDArray[np.float32]:
    """Apply non-antialiased Keys bicubic interpolation like PyTorch.

    PyTorch uses ``a=-0.75``, half-pixel source coordinates for
    ``align_corners=False``, and edge-index clamping. Interpolation is applied
    horizontally and then vertically to preserve its separable evaluation
    order.
    """
    row_indices, row_weights = _axis_indices_and_weights(design.shape[0], target_shape[0])
    column_indices, column_weights = _axis_indices_and_weights(design.shape[1], target_shape[1])

    horizontal_samples = design[:, column_indices]
    horizontal = _weighted_sum_four(horizontal_samples, column_weights[None, :, :])

    vertical_samples = np.moveaxis(horizontal[row_indices, :], 1, -1)
    resized = _weighted_sum_four(vertical_samples, row_weights[:, None, :])
    return np.ascontiguousarray(np.clip(resized, 0.0, 1.0), dtype=np.float32)


def half_to_legacy_256(design: npt.NDArray) -> npt.NDArray[np.float32]:
    """Lossily convert a native half-domain to a legacy ``(256, 256)`` array.

    The complete native ``(400, 200)`` half-domain is anisotropically resized
    directly to ``(256, 256)``. It is not mirrored first. Bicubic overshoot is
    clipped to ``[0, 1]``.
    """
    half = _validate_density_array(design, HALF_DESIGN_SHAPE, name="design")
    return _resize_density(half, LEGACY_DESIGN_SHAPE)


def legacy_256_to_half(design: npt.NDArray) -> npt.NDArray[np.float32]:
    """Lossily convert a legacy ``(256, 256)`` field to the native half-domain.

    The complete legacy array represents one half-domain. It is concatenated
    with its horizontal mirror to form ``(256, 512)``, bicubically resized to
    the native ``(400, 400)`` full field, and reduced to the first 200 columns.
    This matches the VQGAN/MTO conversion geometry. Bicubic overshoot is clipped
    to ``[0, 1]``.
    """
    legacy = _validate_density_array(design, LEGACY_DESIGN_SHAPE, name="legacy design")
    mirrored = np.concatenate((legacy, np.fliplr(legacy)), axis=1)
    native_full = _resize_density(mirrored, FULL_DESIGN_SHAPE)
    return np.ascontiguousarray(native_full[:, : HALF_DESIGN_SHAPE[1]])


def _replace_internal_field(content: str, values: npt.NDArray[np.float64]) -> str:
    match = _internal_field_match(content)
    scalar_lines = "\n".join(format(float(value), ".17g") for value in values)
    replacement = f"internalField   nonuniform List<scalar>\n{values.size}\n(\n{scalar_lines}\n);"
    return content[: match.start()] + replacement + content[match.end() :]


def _update_foam_location(content: str, location: str) -> str:
    if not location or not location.isascii() or any(character in location for character in '"\r\n'):
        raise ValueError("OpenFOAM location must be a non-empty ASCII string without quotes or newlines")

    header = _foam_header(content)
    if header is None:
        raise ValueError("Could not find a FoamFile header in the OpenFOAM template")

    body = header.group("body")
    location_match = _LOCATION_RE.search(body)
    replacement = f'location    "{location}";'
    if location_match is None:
        updated_body = body.rstrip() + f"\n    {replacement}\n"
    else:
        updated_body = (
            body[: location_match.start()] + location_match.group("indent") + replacement + body[location_match.end() :]
        )
    return content[: header.start("body")] + updated_body + content[header.end("body") :]


def _read_ascii(path: Path) -> str:
    try:
        return path.read_text(encoding="ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"OpenFOAM field is not ASCII text: {path}") from error


def read_half_design(path: str | Path) -> npt.NDArray[np.float32]:
    """Read an ASCII OpenFOAM gamma file as a native half-domain design."""
    gamma = parse_internal_field(_read_ascii(Path(path)), expected_count=GAMMA_CELL_COUNT)
    return gamma_to_half_design(gamma)


def write_half_design(
    design: npt.NDArray,
    template_path: str | Path,
    output_path: str | Path,
    *,
    location: str = "0",
) -> Path:
    """Write a native design into an ASCII OpenFOAM gamma template.

    Only the first 80,000 internal-field values are replaced. The final 6,400
    fixed/non-design values and all text outside ``internalField`` are retained
    from ``template_path``. The ``FoamFile`` location is changed to ``"0"`` by
    default.

    Returns:
        The output path.
    """
    template = _read_ascii(Path(template_path))
    template_gamma = parse_internal_field(template, expected_count=GAMMA_CELL_COUNT)
    gamma = half_design_to_gamma(design, template_gamma)
    rendered = _update_foam_location(_replace_internal_field(template, gamma), location)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="ascii")
    return destination
