from pathlib import Path
import re

import numpy as np
import pytest

from engibench.problems.mto2d.model.design_io import DESIGN_CELL_COUNT
from engibench.problems.mto2d.model.design_io import FIXED_CELL_COUNT
from engibench.problems.mto2d.model.design_io import FULL_DESIGN_SHAPE
from engibench.problems.mto2d.model.design_io import full_to_half
from engibench.problems.mto2d.model.design_io import GAMMA_CELL_COUNT
from engibench.problems.mto2d.model.design_io import gamma_to_half_design
from engibench.problems.mto2d.model.design_io import HALF_DESIGN_SHAPE
from engibench.problems.mto2d.model.design_io import half_design_to_gamma
from engibench.problems.mto2d.model.design_io import half_to_full
from engibench.problems.mto2d.model.design_io import parse_internal_field
from engibench.problems.mto2d.model.design_io import read_half_design
from engibench.problems.mto2d.model.design_io import write_half_design


def _foam_field(values: np.ndarray, *, location: str = "200", values_per_line: int = 1) -> str:
    tokens = [format(float(value), ".17g") for value in values]
    value_lines = [" \t ".join(tokens[start : start + values_per_line]) for start in range(0, len(tokens), values_per_line)]
    value_text = "\n".join(value_lines)
    return f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    location    "{location}";
    object gamma;
}}

dimensions [0 0 0 0 0 0 0];
internalField
  nonuniform   List < scalar >
 {len(values)}
 (
{value_text}
 ) ;

boundaryField
{{
    walls
    {{
        type zeroGradient;
    }}
}}
"""


@pytest.fixture
def gamma_values() -> np.ndarray:
    design_values = np.linspace(0.0, 1.0, DESIGN_CELL_COUNT, dtype=np.float64)
    fixed_values = np.linspace(0.2, 0.8, FIXED_CELL_COUNT, dtype=np.float64)
    return np.concatenate((design_values, fixed_values))


@pytest.fixture
def template_path(tmp_path: Path, gamma_values: np.ndarray) -> Path:
    path = tmp_path / "gamma.template"
    path.write_text(_foam_field(gamma_values, values_per_line=7), encoding="ascii")
    return path


def test_gamma_ordering_matches_two_solver_blocks_and_visual_flip(gamma_values: np.ndarray) -> None:
    design = gamma_to_half_design(gamma_values)

    expected = np.flipud(
        np.concatenate(
            (
                gamma_values[:64_000].reshape(400, 160),
                gamma_values[64_000:80_000].reshape(400, 40),
            ),
            axis=1,
        )
    ).astype(np.float32)

    assert design.dtype == np.float32
    assert design.shape == HALF_DESIGN_SHAPE
    np.testing.assert_array_equal(design, expected)


def test_read_write_round_trip_preserves_fixed_tail_and_updates_location(
    tmp_path: Path,
    template_path: Path,
    gamma_values: np.ndarray,
) -> None:
    design = np.linspace(1.0, 0.0, DESIGN_CELL_COUNT, dtype=np.float32).reshape(HALF_DESIGN_SHAPE)
    output_path = tmp_path / "case" / "app" / "0" / "gamma"

    returned_path = write_half_design(design, template_path, output_path)
    written_text = output_path.read_text(encoding="ascii")
    written_gamma = parse_internal_field(written_text, expected_count=GAMMA_CELL_COUNT)

    assert returned_path == output_path
    assert re.search(r'(?m)^\s*location\s+"0";', written_text)
    assert 'location    "200";' not in written_text
    assert "boundaryField" in written_text
    np.testing.assert_array_equal(written_gamma[DESIGN_CELL_COUNT:], gamma_values[DESIGN_CELL_COUNT:])
    np.testing.assert_array_equal(read_half_design(output_path), design)

    solver_oriented = np.flipud(design)
    np.testing.assert_array_equal(written_gamma[:64_000], solver_oriented[:, :160].reshape(-1))
    np.testing.assert_array_equal(written_gamma[64_000:80_000], solver_oriented[:, 160:].reshape(-1))


def test_half_design_to_gamma_preserves_template_tail(gamma_values: np.ndarray) -> None:
    design = gamma_to_half_design(gamma_values)
    converted = half_design_to_gamma(design, gamma_values)

    np.testing.assert_array_equal(converted[DESIGN_CELL_COUNT:], gamma_values[DESIGN_CELL_COUNT:])
    np.testing.assert_array_equal(gamma_to_half_design(converted), design)


def test_parse_internal_field_is_whitespace_tolerant() -> None:
    values = np.array([0.0, 0.125, 1.0, 2.5e-4], dtype=np.float64)
    text = _foam_field(values, values_per_line=3)

    np.testing.assert_array_equal(parse_internal_field(text, expected_count=4), values)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("FoamFile { format ascii; }", "internalField"),
        (
            "FoamFile { format ascii; }\ninternalField nonuniform List<scalar> 3 (0 1);",
            "declares 3 values but contains 2",
        ),
        (
            "FoamFile { format ascii; }\ninternalField nonuniform List<scalar> 2 (0 nope);",
            "invalid scalar token",
        ),
        (
            "FoamFile { format binary; }\ninternalField nonuniform List<scalar> 1 (0);",
            "declares binary",
        ),
        (
            "FoamFile { format ascii; }\ninternalField nonuniform List<scalar> 1 (nan);",
            "non-finite",
        ),
    ],
)
def test_parse_internal_field_reports_clear_errors(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_internal_field(text)


def test_parse_internal_field_checks_expected_count() -> None:
    with pytest.raises(ValueError, match="expected 2"):
        parse_internal_field(_foam_field(np.array([0.5], dtype=np.float64)), expected_count=2)


def test_half_to_full_and_full_to_half_round_trip() -> None:
    half = np.linspace(0.0, 1.0, DESIGN_CELL_COUNT, dtype=np.float32).reshape(HALF_DESIGN_SHAPE)

    full = half_to_full(half)

    assert full.shape == FULL_DESIGN_SHAPE
    np.testing.assert_array_equal(full[:, :200], half)
    np.testing.assert_array_equal(full[:, 200:], np.fliplr(half))
    np.testing.assert_array_equal(full_to_half(full), half)


def test_full_to_half_optionally_validates_symmetry() -> None:
    half = np.full(HALF_DESIGN_SHAPE, 0.5, dtype=np.float32)
    asymmetric = half_to_full(half)
    asymmetric[0, -1] = 0.75

    with pytest.raises(ValueError, match="not horizontally symmetric"):
        full_to_half(asymmetric)

    np.testing.assert_array_equal(full_to_half(asymmetric, validate_symmetry=False), half)


@pytest.mark.parametrize(
    ("design", "error_type", "message"),
    [
        (np.zeros((200, 400), dtype=np.float32), ValueError, "shape"),
        (np.zeros(HALF_DESIGN_SHAPE, dtype=np.float64), TypeError, "dtype float32"),
        (np.full(HALF_DESIGN_SHAPE, np.nan, dtype=np.float32), ValueError, "finite"),
        (np.full(HALF_DESIGN_SHAPE, -0.1, dtype=np.float32), ValueError, r"\[0, 1\]"),
        (np.full(HALF_DESIGN_SHAPE, 1.1, dtype=np.float32), ValueError, r"\[0, 1\]"),
    ],
)
def test_half_design_validation(design: np.ndarray, error_type: type[Exception], message: str) -> None:
    with pytest.raises(error_type, match=message):
        half_to_full(design)


def test_gamma_validation_rejects_invalid_count_dtype_and_bounds() -> None:
    with pytest.raises(ValueError, match=str(GAMMA_CELL_COUNT)):
        gamma_to_half_design(np.zeros(GAMMA_CELL_COUNT - 1, dtype=np.float64))
    with pytest.raises(TypeError, match="floating dtype"):
        gamma_to_half_design(np.zeros(GAMMA_CELL_COUNT, dtype=np.int64))

    invalid = np.zeros(GAMMA_CELL_COUNT, dtype=np.float64)
    invalid[-1] = 1.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        gamma_to_half_design(invalid)


def test_write_requires_foam_header_for_location_update(
    tmp_path: Path,
    gamma_values: np.ndarray,
) -> None:
    template = tmp_path / "gamma.no-header"
    template.write_text(
        re.sub(r"FoamFile\s*\{.*?\}", "", _foam_field(gamma_values), count=1, flags=re.DOTALL),
        encoding="ascii",
    )
    design = np.zeros(HALF_DESIGN_SHAPE, dtype=np.float32)

    with pytest.raises(ValueError, match="FoamFile header"):
        write_half_design(design, template, tmp_path / "gamma")


def test_write_rejects_unsafe_location(template_path: Path, tmp_path: Path) -> None:
    design = np.zeros(HALF_DESIGN_SHAPE, dtype=np.float32)

    with pytest.raises(ValueError, match="location"):
        write_half_design(design, template_path, tmp_path / "gamma", location='0"; bad')
