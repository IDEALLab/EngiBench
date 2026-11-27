import gzip
from pathlib import Path

import numpy as np
import pytest

from engibench.problems.mto3d.model import design_io as dio

BOUNDARY = (
    "boundaryField\n{\n"
    + "".join(
        f"    {p}\n    {{\n        type            {'symmetry' if p == 'sym' else 'zeroGradient'};\n    }}\n"
        for p in dio.PATCH_NAMES
    )
    + "}\n"
)


def field_text(tokens: list[str], *, internal: str | None = None, boundary: str = BOUNDARY) -> str:
    header = (
        "FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       volScalarField;\n"
        '    location    "1922";\n    object      xh;\n}\n\ndimensions      [0 0 0 0 0 0 0];\n\n'
    )
    if internal is None:
        internal = f"internalField   nonuniform List<scalar> \n{len(tokens)}\n(\n" + "\n".join(tokens) + "\n)\n;\n\n"
    return header + internal + boundary + "\n// end\n"


def template_tokens(design_value: str = "0.5") -> list[str]:
    tokens = np.full(dio.XH_CELL_COUNT, design_value, dtype=object)
    tokens[dio.SOLID_INDEX] = "0"
    tokens[dio.FLUID_INDEX] = "1"
    return tokens.tolist()


def sentinel_design() -> np.ndarray:
    """Exact binary fractions, representable in float32 and printed exactly."""
    return (np.arange(dio.DESIGN_CELL_COUNT) / 262144.0).astype(np.float32).reshape(dio.HALF_DESIGN_SHAPE)


@pytest.fixture
def template_path(tmp_path: Path) -> Path:
    path = tmp_path / "tpl" / "xh.gz"
    path.parent.mkdir()
    path.write_bytes(gzip.compress(field_text(template_tokens()).encode(), mtime=0))
    return path


def test_partition_is_complete_and_disjoint() -> None:
    design = dio.NATIVE_DESIGN_INDEX.reshape(-1)
    assert np.unique(design).size == dio.DESIGN_CELL_COUNT
    assert dio.SOLID_INDEX.size == dio.SOLID_CELL_COUNT
    assert dio.FLUID_INDEX.size == dio.FLUID_CELL_COUNT
    every = np.sort(np.concatenate((design, dio.SOLID_INDEX, dio.FLUID_INDEX)))
    np.testing.assert_array_equal(every, np.arange(dio.XH_CELL_COUNT))


@pytest.mark.parametrize(
    ("native", "position"),
    [
        (20_000, (0, 99, 0)),
        (20_001, (0, 99, 1)),
        (20_002, (1, 99, 0)),
        (20_040, (0, 98, 0)),
        (21_599, (19, 60, 1)),
        (21_600, (0, 99, 2)),
        (84_000, (0, 99, 80)),
        (99_999, (19, 60, 99)),
        (112_000, (0, 59, 0)),
        (243_999, (19, 0, 99)),
    ],
)
def test_known_native_positions(native: int, position: tuple[int, int, int]) -> None:
    assert dio.NATIVE_DESIGN_INDEX[position] == native


def test_sentinel_round_trip_in_memory() -> None:
    template = np.array(template_tokens(), dtype=np.float64)
    design = sentinel_design()
    xh = dio.half_design_to_xh(design, template)
    np.testing.assert_array_equal(xh[dio.NATIVE_DESIGN_INDEX], design.astype(np.float64))
    np.testing.assert_array_equal(dio.xh_to_half_design(xh), design)


@pytest.mark.parametrize("shape", [(100, 20, 100), (100, 100, 20), (200_000,), (264_000,), (20, 100, 200)])
def test_rejects_other_shapes_without_reshaping(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="must have shape"):
        dio.half_design_to_xh(np.zeros(shape, dtype=np.float32), np.array(template_tokens(), dtype=np.float64))


def test_rejects_bad_dtype_values_and_inexact_templates() -> None:
    template = np.array(template_tokens(), dtype=np.float64)
    good = np.full(dio.HALF_DESIGN_SHAPE, 0.5, dtype=np.float32)
    with pytest.raises(TypeError, match="float32"):
        dio.half_design_to_xh(good.astype(np.float64), template)
    for bad in (np.nan, 1.0000001, -1e-6):
        array = good.copy()
        array[3, 4, 5] = bad
        with pytest.raises(ValueError, match=r"finite|\[0, 1\]"):
            dio.half_design_to_xh(array, template)
    template[dio.SOLID_INDEX[0]] = 1e-40
    with pytest.raises(ValueError, match="fixed cells are not exact"):
        dio.half_design_to_xh(good, template)


def test_mirror_is_for_rendering_only() -> None:
    design = sentinel_design()
    full = dio.half_to_full(design)
    assert full.shape == dio.FULL_DESIGN_SHAPE
    np.testing.assert_array_equal(full[:, :, 100:], design[:, :, ::-1])


def test_reads_gzip_and_plain_identically(tmp_path: Path) -> None:
    text = field_text(template_tokens("0.123"))
    gz = tmp_path / "a" / "xh.gz"
    plain = tmp_path / "b" / "xh"
    gz.parent.mkdir()
    plain.parent.mkdir()
    gz.write_bytes(gzip.compress(text.encode(), mtime=0))
    plain.write_text(text, encoding="ascii")
    a, b = dio.read_xh(gz), dio.read_xh(plain)
    assert a.compressed
    assert not b.compressed
    assert a.tokens == b.tokens


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("format      ascii", "format      binary", "binary"),
        ("class       volScalarField", "class       volVectorField", "volScalarField"),
        ("\n264000\n(", "\n263999\n(", "declares 263999"),
        ("\n0\n", "\n0x\n", "malformed internalField value"),
        ("\n0\n", "\nnan\n", "non-finite"),
        ("\n0\n", "\n1.5\n", "outside"),
        ("\n0\n", "\n0 0\n", "contains 264001 tokens"),
        (")\n;", ")\n", "followed by ';'"),
    ],
)
def test_rejects_bad_fields(tmp_path: Path, old: str, new: str, message: str) -> None:
    path = tmp_path / "xh"
    path.write_text(field_text(template_tokens()).replace(old, new, 1), encoding="ascii")
    with pytest.raises(ValueError, match=message):
        dio.read_xh(path)


@pytest.mark.parametrize(
    ("cut", "message"), [(")\n;", "not closed"), ("boundaryField", "no boundaryField"), ("    sym", "is not closed")]
)
def test_rejects_truncated_text(tmp_path: Path, cut: str, message: str) -> None:
    text = field_text(template_tokens())
    path = tmp_path / "xh"
    path.write_text(text[: text.index(cut)], encoding="ascii")
    with pytest.raises(ValueError, match=message):
        dio.read_xh(path)


def test_rejects_uniform_truncated_gzip_and_suffix_mismatch(tmp_path: Path) -> None:
    uniform = tmp_path / "u" / "xh"
    uniform.parent.mkdir()
    uniform.write_text(field_text([], internal="internalField   uniform 0.49;\n\n"), encoding="ascii")
    with pytest.raises(ValueError, match="nonuniform"):
        dio.read_xh(uniform)
    data = gzip.compress(field_text(template_tokens()).encode(), mtime=0)
    truncated = tmp_path / "t" / "xh.gz"
    truncated.parent.mkdir()
    truncated.write_bytes(data[: len(data) // 2])
    with pytest.raises(ValueError, match="truncated or corrupt gzip"):
        dio.read_xh(truncated)
    mislabeled = tmp_path / "m" / "xh.gz"
    mislabeled.parent.mkdir()
    mislabeled.write_text(field_text(template_tokens()), encoding="ascii")
    with pytest.raises(ValueError, match="disagree"):
        dio.read_xh(mislabeled)


@pytest.mark.parametrize("name", ["xh", "xh.gz"])
def test_writer_round_trip_and_fidelity(tmp_path: Path, template_path: Path, name: str) -> None:
    design = sentinel_design()
    written = dio.read_xh(dio.write_half_design(design, template_path, tmp_path / "0" / name))
    template = dio.read_xh(template_path)
    assert written.foam_file["location"] == "0"
    np.testing.assert_array_equal(dio.xh_to_half_design(written.values), design)
    fixed = np.concatenate((dio.SOLID_INDEX, dio.FLUID_INDEX))
    np.testing.assert_array_equal(np.array(written.tokens)[fixed], np.array(template.tokens)[fixed])
    assert written.text.endswith(template.text[template.values_span[1] :])


def test_writer_is_deterministic_and_refuses_ambiguity(tmp_path: Path, template_path: Path) -> None:
    design = sentinel_design()
    first = dio.write_half_design(design, template_path, tmp_path / "a" / "xh.gz").read_bytes()
    second = dio.write_half_design(design, template_path, tmp_path / "b" / "xh.gz").read_bytes()
    assert first == second
    with pytest.raises(FileExistsError):
        dio.write_half_design(design, template_path, tmp_path / "a" / "xh")
    with pytest.raises(ValueError, match="named"):
        dio.write_half_design(design, template_path, tmp_path / "c" / "T.gz")


def test_float32_formatting_reproduces_three_significant_figure_tokens() -> None:
    tokens = ["0.123", "0.5", "0.000123", "1.23e-05", "9.99e-31", "0.999"]
    values = np.array([float(t) for t in tokens]).astype(np.float32)
    assert dio.format_float32(values) == tokens
    assert dio.format_float32(np.array([-0.0, 0.0, 1.0], dtype=np.float32)) == ["0", "0", "1"]
