"""Conversions between MTO3D designs and the OpenFOAM ``xh`` field.

The EngiBench-native representation is the non-redundant ``(20, 100, 100)`` float32
half-domain in ``(y, z, x)`` order. ``blockMesh`` numbers the 264,000 ``xh`` cells block by
block, and inside a block x varies fastest, then y, then z:

* ``[0, 12000)`` outlet block and ``[252000, 264000)`` inlet block: fixed fluid (1);
* ``[12000, 20000)``, ``[100000, 112000)``, ``[152000, 164000)`` and ``[244000, 252000)``:
  fixed solid shells (0);
* ``[20000, 100000)``, ``[112000, 152000)`` and ``[164000, 244000)``: the three design
  bodies. Each holds three separate x sub-blocks of widths 2, 78 and 20, so every sub-block is
  reshaped ``(z, y, x_b)`` and the sub-blocks are concatenated along x.

Axis conventions of the half-design:

* axis 0 (y, 20 cells): increasing, between the solid shells;
* axis 1 (z, 100 cells): decreasing ("visual" orientation, as in MTO2D and the legacy 3D
  tooling) -- index 0 is the inlet side;
* axis 2 (x, 100 cells): increasing; the last index touches the symmetry plane ``x = 0``.

The half-design is shape-symmetric in its last two axes, so a z/x transpose cannot be
detected from the array; the axis order is part of the contract. Mirroring to
``(20, 100, 200)`` is for rendering only.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import gzip
from pathlib import Path
import re
import zlib

import numpy as np
import numpy.typing as npt

XH_CELL_COUNT = 264_000
"""Number of cells (and ``xh`` values) in the MTO3D mesh."""
DESIGN_CELL_COUNT = 200_000
SOLID_CELL_COUNT = 40_000
FLUID_CELL_COUNT = 24_000
FIXED_CELL_COUNT = SOLID_CELL_COUNT + FLUID_CELL_COUNT

HALF_DESIGN_SHAPE = (20, 100, 100)
"""Shape of the non-redundant EngiBench design, axes ``(y, z, x)``."""
FULL_DESIGN_SHAPE = (20, 100, 200)
"""Shape of the mirrored visualization."""

SUB_BLOCK_X_WIDTHS = (2, 78, 20)
"""x widths of the three sub-blocks of every design body."""
SUB_BLOCK_X_OFFSETS = (0, 2, 80)
PATCH_NAMES = ("outlet", "inlet", "force", "wall", "sym", "bot")
GZIP_MAGIC = b"\x1f\x8b"


@dataclass(frozen=True)
class DesignBody:
    """One design body: three consecutive x sub-blocks in ``blockMesh`` order."""

    name: str
    native_start: int
    nz: int
    z_offset: int
    """First z index counted upward from ``z = 0`` within the 100-cell design height."""


DESIGN_BODIES = (
    DesignBody("body1", 20_000, 40, 0),
    DesignBody("body2", 112_000, 20, 40),
    DesignBody("body3", 164_000, 40, 60),
)
FIXED_SPANS = (
    (0, 12_000, "fluid"),
    (12_000, 20_000, "solid"),
    (100_000, 112_000, "solid"),
    (152_000, 164_000, "solid"),
    (244_000, 252_000, "solid"),
    (252_000, 264_000, "fluid"),
)


def _build_native_design_index() -> npt.NDArray[np.int64]:
    upward = np.full(HALF_DESIGN_SHAPE, -1, dtype=np.int64)
    ny = HALF_DESIGN_SHAPE[0]
    for body in DESIGN_BODIES:
        start = body.native_start
        for width, x0 in zip(SUB_BLOCK_X_WIDTHS, SUB_BLOCK_X_OFFSETS, strict=True):
            count = width * ny * body.nz
            block = np.arange(start, start + count, dtype=np.int64).reshape(body.nz, ny, width)
            upward[:, body.z_offset : body.z_offset + body.nz, x0 : x0 + width] = block.transpose(1, 0, 2)
            start += count
    if np.any(upward < 0):  # pragma: no cover - constant layout check
        raise AssertionError("MTO3D design index map is not fully populated")
    index = np.ascontiguousarray(upward[:, ::-1, :])
    index.setflags(write=False)
    return index


def _span_index(kind: str) -> npt.NDArray[np.int64]:
    index = np.concatenate([np.arange(a, b, dtype=np.int64) for a, b, k in FIXED_SPANS if k == kind])
    index.setflags(write=False)
    return index


NATIVE_DESIGN_INDEX = _build_native_design_index()
"""``NATIVE_DESIGN_INDEX[j, k, i]`` is the ``xh`` position of half-design entry ``[j, k, i]``."""
SOLID_INDEX = _span_index("solid")
FLUID_INDEX = _span_index("fluid")


def validate_half_design(design: npt.NDArray, *, name: str = "design") -> npt.NDArray[np.float32]:
    """Require a finite float32 ``(20, 100, 100)`` array in ``[0, 1]``; never reshape."""
    if not isinstance(design, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if design.dtype != np.dtype(np.float32):
        raise TypeError(f"{name} must have dtype float32; got {design.dtype}")
    if design.shape != HALF_DESIGN_SHAPE:
        hint = ""
        if design.size == DESIGN_CELL_COUNT:
            hint = " (same element count: transposed, flattened or native-order arrays are rejected, never reshaped)"
        raise ValueError(f"{name} must have shape {HALF_DESIGN_SHAPE} (y, z, x); got {design.shape}{hint}")
    if not np.all(np.isfinite(design)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any((design < 0.0) | (design > 1.0)):
        raise ValueError(f"{name} values must lie in [0, 1]")
    return design


def _validate_xh_values(values: npt.NDArray, *, name: str = "xh") -> npt.NDArray[np.float64]:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} values must have a floating dtype; got {array.dtype}")
    if array.shape != (XH_CELL_COUNT,):
        raise ValueError(f"{name} must contain exactly {XH_CELL_COUNT} values; got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} values must lie in [0, 1]")
    return np.asarray(array, dtype=np.float64)


def check_fixed_cells(values: npt.NDArray, *, name: str = "template") -> None:
    """Require exactly 0 in every ``solid_area`` cell and exactly 1 in every ``fluid_area`` cell.

    Final time directories satisfy this; ``<final-2>`` directories do not, because the PDE
    filter leaves tiny nonzero values in the solid shells.
    """
    xh = _validate_xh_values(values, name=name)
    bad_solid = int(np.count_nonzero(xh[SOLID_INDEX] != 0.0))
    bad_fluid = int(np.count_nonzero(xh[FLUID_INDEX] != 1.0))
    if bad_solid or bad_fluid:
        raise ValueError(f"{name} fixed cells are not exact: {bad_solid} solid cells != 0 and {bad_fluid} fluid cells != 1")


def xh_to_half_design(xh_values: npt.NDArray) -> npt.NDArray[np.float32]:
    """Extract the 200,000 design cells of a native ``xh`` list as a float32 half-design."""
    xh = _validate_xh_values(xh_values)
    return np.ascontiguousarray(xh[NATIVE_DESIGN_INDEX], dtype=np.float32)


def half_design_to_xh(design: npt.NDArray, template_values: npt.NDArray) -> npt.NDArray[np.float64]:
    """Place a half-design into native order; all 64,000 fixed cells come from the template."""
    half = validate_half_design(design)
    check_fixed_cells(template_values)
    xh = _validate_xh_values(template_values).copy()
    xh[NATIVE_DESIGN_INDEX.reshape(-1)] = half.reshape(-1).astype(np.float64)
    return xh


def half_to_full(design: npt.NDArray) -> npt.NDArray[np.float32]:
    """Mirror across the symmetry plane to ``(20, 100, 200)``. For rendering only."""
    half = validate_half_design(design)
    return np.ascontiguousarray(np.concatenate((half, half[:, :, ::-1]), axis=2))


_FOAM_FILE_RE = re.compile(r"\bFoamFile\s*\{(?P<body>[^{}]*)\}", flags=re.DOTALL)
_ENTRY_RE = re.compile(r"(?P<key>\w+)\s+(?P<value>\"[^\"]*\"|[^;\s][^;]*?)\s*;")
_DIMENSIONS_RE = re.compile(r"\bdimensions\s+\[[^\]]*\]\s*;")
_INTERNAL_RE = re.compile(r"\binternalField\s+(?P<kind>\w+)")
_LIST_HEAD_RE = re.compile(r"\s*List\s*<\s*scalar\s*>\s*(?P<count>\d+)\s*\(", flags=re.DOTALL)
_LOCATION_RE = re.compile(r'(?m)^(?P<indent>[ \t]*)location\s+(?:"[^"\r\n]*"|[^;\r\n]+)\s*;')
_EXACT_TOKENS = {"0.0": "0", "1.0": "1"}


@dataclass(frozen=True)
class ScalarField:
    """A parsed ASCII OpenFOAM ``volScalarField`` whose text is kept for faithful rewriting."""

    text: str
    values: npt.NDArray[np.float64]
    tokens: tuple[str, ...] = field(repr=False)
    values_span: tuple[int, int]
    foam_file: dict[str, str]
    compressed: bool


def _parse_header(text: str) -> tuple[dict[str, str], int]:
    header = _FOAM_FILE_RE.search(text)
    if header is None:
        raise ValueError("malformed field: no FoamFile header")
    foam = {m.group("key"): m.group("value").strip('"') for m in _ENTRY_RE.finditer(header.group("body"))}
    if foam.get("format") == "binary":
        raise ValueError("binary OpenFOAM fields are not supported; the FoamFile header declares format binary")
    if foam.get("format") != "ascii":
        raise ValueError(f"malformed field: FoamFile format must be ascii; got {foam.get('format')!r}")
    if foam.get("class") != "volScalarField":
        raise ValueError(f"expected class volScalarField; got {foam.get('class')!r}")
    if _DIMENSIONS_RE.search(text, header.end()) is None:
        raise ValueError("malformed field: no dimensions entry")
    return foam, header.end()


def _locate_values(text: str, start: int, expected_count: int | None) -> tuple[int, int, int]:
    internal = _INTERNAL_RE.search(text, start)
    if internal is None:
        raise ValueError("malformed or truncated field: no internalField entry")
    if internal.group("kind") != "nonuniform":
        raise ValueError(f"internalField must be a nonuniform list of every cell; got {internal.group('kind')!r}")
    head = _LIST_HEAD_RE.match(text, internal.end())
    if head is None:
        raise ValueError("malformed internalField: expected 'nonuniform List<scalar> <count> ('")
    declared = int(head.group("count"))
    if expected_count is not None and declared != expected_count:
        raise ValueError(f"internalField declares {declared} values; expected {expected_count}")
    close = text.find(")", head.end())
    if close < 0:
        raise ValueError("truncated field: internalField value list is not closed")
    if re.match(r"\s*;", text[close + 1 :]) is None:
        raise ValueError("malformed field: internalField list must be followed by ';'")
    return head.end(), close, declared


def _check_boundary(text: str, start: int) -> None:
    boundary = text.find("boundaryField", start)
    if boundary < 0:
        raise ValueError("truncated field: no boundaryField after internalField")
    brace = text.find("{", boundary)
    depth = 0
    for position in range(brace, len(text)) if brace >= 0 else ():
        depth += (text[position] == "{") - (text[position] == "}")
        if depth == 0:
            return
    raise ValueError("truncated field: boundaryField is not closed")


def _parse_values(body: str, declared: int, bounds: tuple[float, float] | None) -> tuple[list[str], npt.NDArray]:
    if "(" in body:
        raise ValueError("malformed internalField: nested parentheses in a scalar list")
    tokens = body.split()
    if len(tokens) != declared:
        raise ValueError(f"internalField declares {declared} values but contains {len(tokens)} tokens")
    try:
        values = np.array(tokens, dtype=np.float64)
    except ValueError as error:
        bad = next(((i, t) for i, t in enumerate(tokens) if not _is_float(t)), (-1, "<unknown>"))
        raise ValueError(f"malformed internalField value {bad[1]!r} at cell {bad[0]}") from error
    finite = np.isfinite(values)
    if not finite.all():
        i = int(np.flatnonzero(~finite)[0])
        raise ValueError(f"non-finite internalField value {tokens[i]!r} at cell {i}")
    if bounds is not None:
        outside = (values < bounds[0]) | (values > bounds[1])
        if outside.any():
            i = int(np.flatnonzero(outside)[0])
            raise ValueError(f"internalField value {tokens[i]} at cell {i} is outside [{bounds[0]}, {bounds[1]}]")
    return tokens, values


def parse_scalar_field(
    text: str,
    *,
    expected_count: int | None = XH_CELL_COUNT,
    bounds: tuple[float, float] | None = (0.0, 1.0),
    compressed: bool = False,
) -> ScalarField:
    """Parse and validate an ASCII ``volScalarField`` with a nonuniform internal field.

    Raises:
        TypeError: If ``text`` is not a string.
        ValueError: For binary format, a wrong count, malformed or truncated text,
            non-finite values, or values outside ``bounds``.
    """
    if not isinstance(text, str):
        raise TypeError("OpenFOAM content must be text")
    foam, header_end = _parse_header(text)
    list_start, close, declared = _locate_values(text, header_end, expected_count)
    body = text[list_start:close]
    tokens, values = _parse_values(body, declared, bounds)
    _check_boundary(text, close)
    span = (list_start + len(body) - len(body.lstrip()), close - (len(body) - len(body.rstrip())))
    return ScalarField(text, values, tuple(tokens), span, foam, compressed)


def _is_float(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def read_scalar_field(path: str | Path, **kwargs: object) -> ScalarField:
    """Read a gzip or plain ASCII field; the ``.gz`` suffix must agree with the content."""
    file_path = Path(path)
    raw = file_path.read_bytes()
    is_gzip = raw[:2] == GZIP_MAGIC
    if is_gzip != (file_path.suffix == ".gz"):
        raise ValueError(f"{file_path}: gzip content and '.gz' suffix disagree (gzip={is_gzip})")
    if is_gzip:
        try:
            raw = gzip.decompress(raw)
        except (EOFError, OSError, zlib.error) as error:
            raise ValueError(f"{file_path}: truncated or corrupt gzip stream") from error
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"{file_path}: not ASCII text (binary content?)") from error
    return parse_scalar_field(text, compressed=is_gzip, **kwargs)  # type: ignore[arg-type]


def read_xh(path: str | Path, *, bounds: tuple[float, float] = (0.0, 1.0)) -> ScalarField:
    """Read a native ``xh`` field (264,000 values, by default required to lie in ``[0, 1]``)."""
    parsed = read_scalar_field(path, bounds=bounds)
    if parsed.foam_file.get("object") != "xh":
        raise ValueError(f"{path}: FoamFile object is {parsed.foam_file.get('object')!r}, expected 'xh'")
    return parsed


def read_half_design(path: str | Path) -> npt.NDArray[np.float32]:
    """Read an ``xh`` field file as a ``(20, 100, 100)`` float32 half-design."""
    return xh_to_half_design(read_xh(path).values)


def format_float32(values: npt.NDArray[np.float32]) -> list[str]:
    """Shortest decimal strings that round-trip each float32 value (``1.0`` is written ``1``)."""
    clean = np.asarray(values, dtype=np.float32) + np.float32(0.0)
    return [_EXACT_TOKENS.get(s, s) for s in map(str, clean.reshape(-1))]


def update_location(text: str, location: str) -> str:
    """Set the ``FoamFile`` ``location`` entry, inserting it if absent."""
    if not location or not location.isascii() or any(c in location for c in '"\r\n'):
        raise ValueError("OpenFOAM location must be a non-empty ASCII string without quotes or newlines")
    header = _FOAM_FILE_RE.search(text)
    if header is None:
        raise ValueError("Could not find a FoamFile header")
    body = header.group("body")
    found = _LOCATION_RE.search(body)
    entry = f'location    "{location}";'
    if found is None:
        new_body = body.rstrip() + f"\n    {entry}\n"
    else:
        new_body = body[: found.start()] + found.group("indent") + entry + body[found.end() :]
    return text[: header.start("body")] + new_body + text[header.end("body") :]


def render_half_design(design: npt.NDArray, template: ScalarField, *, location: str = "0") -> str:
    """Render a half-design as ``xh`` text, keeping the template's layout and fixed tokens."""
    half = validate_half_design(design)
    check_fixed_cells(template.values)
    tokens = np.array(template.tokens, dtype=object)
    tokens[NATIVE_DESIGN_INDEX.reshape(-1)] = format_float32(half)
    start, stop = template.values_span
    return update_location(template.text[:start] + "\n".join(tokens.tolist()) + template.text[stop:], location)


def write_half_design(
    design: npt.NDArray,
    template_path: str | Path,
    output_path: str | Path,
    *,
    location: str = "0",
) -> Path:
    """Write a half-design into a copy of a final-time ``xh`` template.

    The output is gzip-compressed (deterministically) when it ends in ``.gz``. Writing ``xh``
    next to an existing ``xh.gz``, or the reverse, is refused because OpenFOAM would silently
    prefer the uncompressed file. The written file is re-read and checked.

    Returns:
        The output path.
    """
    destination = Path(output_path)
    if destination.name not in {"xh", "xh.gz"}:
        raise ValueError(f"output file must be named 'xh' or 'xh.gz'; got {destination.name!r}")
    sibling = destination.with_name("xh" if destination.name == "xh.gz" else "xh.gz")
    if sibling.exists():
        raise FileExistsError(f"{sibling} exists; OpenFOAM would be ambiguous between xh and xh.gz")
    data = render_half_design(design, read_xh(template_path), location=location).encode("ascii")
    if destination.suffix == ".gz":
        data = gzip.compress(data, compresslevel=6, mtime=0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    if not np.array_equal(read_half_design(destination), design):
        raise RuntimeError(f"{destination}: re-read design differs from the written design")
    return destination
