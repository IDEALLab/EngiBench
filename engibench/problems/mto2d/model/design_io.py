"""Conversions between MTO2D density arrays and OpenFOAM gamma fields.

The EngiBench-native representation is the non-redundant, visually oriented
``(400, 200)`` half-domain. OpenFOAM stores that domain in two blocks, followed
by 6,400 fixed/non-design cells:

* values ``0:64000`` form a ``(400, 160)`` block;
* values ``64000:80000`` form a ``(400, 40)`` block;
* values ``80000:86400`` are fixed cells copied from a case template.

The solver blocks are vertically flipped to obtain the visual orientation used
by EngiBench.
"""

from pathlib import Path
import re

import numpy as np
import numpy.typing as npt

HALF_DESIGN_SHAPE = (400, 200)
"""Shape of the non-redundant, visually oriented EngiBench design."""

FULL_DESIGN_SHAPE = (400, 400)
"""Shape of the mirrored visualization."""

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
