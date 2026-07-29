#!/usr/bin/env python3
"""Run the deterministic MTO2D q=0.01 release reference."""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import tempfile

from datasets import load_dataset  # type: ignore[import-untyped, unused-ignore]
from datasets import load_from_disk  # type: ignore[import-untyped, unused-ignore]
import numpy as np

from engibench.problems.mto2d import MTO2D
from engibench.problems.mto2d.model.design_io import read_half_design

EXPECTED_INDEX = 2010
EXPECTED_SOURCE_CASE = 6799
EXPECTED_DESIGN_SHA256 = "7d5b35a291c987db2b0b77e73554aa20cde007065773acb11b50df9c1bef19b9"
EXPECTED_OBJECTIVES = np.array([13.8912, 63.8033], dtype=np.float64)
EXPECTED_Q = 0.01
EXPECTED_CONDITIONS = {
    "inlet_velocity": -0.074,
    "max_power_dissipation": 63.1,
    "volfrac": 0.61,
}
DETERMINISTIC_OUTPUTS = (
    "meanT.txt",
    "Disspower.txt",
    "Voluse.txt",
    "qu.txt",
    "aMax.txt",
    "HEAV.txt",
)
FINAL_RESULT_FIELDS = ("gamma",)
DEFAULT_GOLDEN = Path(__file__).with_name("source-reference-golden.json")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_reference_dataset(source: str) -> tuple[object, str]:
    path = Path(source).expanduser()
    if path.is_dir():
        resolved = path.resolve()
        return load_from_disk(str(resolved)), str(resolved)
    if path.exists():
        raise ValueError(f"MTO2D reference dataset must be a saved DatasetDict directory: {path}")
    return load_dataset(source), source


def _float64_bits(values: np.ndarray) -> list[str]:
    words = np.asarray(values, dtype=np.float64).view(np.uint64)
    return [f"0x{int(word):016x}" for word in words]


def _latest_time_directory(app: Path) -> Path:
    candidates = [path for path in app.iterdir() if path.is_dir() and path.name.isdigit() and int(path.name) > 0]
    if not candidates:
        raise FileNotFoundError(f"No positive numeric time directory found under {app}")
    return max(candidates, key=lambda path: int(path.name))


def _reference_record(run: dict[str, object]) -> dict[str, object]:
    objectives = run["objectives"]
    assert isinstance(objectives, np.ndarray)
    return {
        "conditions": EXPECTED_CONDITIONS,
        "design_sha256": run["design_sha256"],
        "final_design_sha256": run["final_design_sha256"],
        "forward_field_sha256": run["forward_field_sha256"],
        "objective_bits": run["objective_bits"],
        "objectives": objectives.tolist(),
        "output_sha256": run["output_sha256"],
        "q": run["q"],
        "source_case_id": run["source_case_id"],
        "train_index": run["train_index"],
    }


def _assert_golden_match(candidate: dict[str, object], golden_path: Path) -> None:
    try:
        golden = json.loads(golden_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise FileNotFoundError(
            "No trusted MTO2D source-reference golden is available. "
            "Run once against the hash-pinned private SIF oracle before release: "
            f"{golden_path}"
        ) from error
    if golden.get("schema_version") != 1 or not isinstance(golden.get("reference"), dict):
        raise ValueError(f"Unsupported MTO2D source-reference golden: {golden_path}")
    actual = _reference_record(candidate)
    expected = golden["reference"]
    if actual != expected:
        mismatched = sorted(key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key))
        raise AssertionError(
            f"candidate differs from the trusted MTO2D source-reference golden for: {', '.join(mismatched)}"
        )


def _assert_oracle_match(
    candidate: dict[str, object],
    oracle: dict[str, object],
) -> None:
    mismatches: list[str] = []
    candidate_objectives = candidate["objectives"]
    oracle_objectives = oracle["objectives"]
    assert isinstance(candidate_objectives, np.ndarray)
    assert isinstance(oracle_objectives, np.ndarray)
    if not np.array_equal(candidate_objectives, oracle_objectives):
        mismatches.append(f"objectives: {_float64_bits(candidate_objectives)} != {_float64_bits(oracle_objectives)}")
    candidate_design = candidate["final_design"]
    oracle_design = oracle["final_design"]
    assert isinstance(candidate_design, np.ndarray)
    assert isinstance(oracle_design, np.ndarray)
    if not np.array_equal(candidate_design, oracle_design):
        mismatches.append("final design array")
    candidate_fields = candidate["forward_field_bytes"]
    oracle_fields = oracle["forward_field_bytes"]
    assert isinstance(candidate_fields, dict)
    assert isinstance(oracle_fields, dict)
    mismatches.extend(
        f"final {name}: {_sha256(candidate_fields[name])} != {_sha256(oracle_fields[name])}"
        for name in FINAL_RESULT_FIELDS
        if candidate_fields[name] != oracle_fields[name]
    )
    candidate_outputs = candidate["output_bytes"]
    oracle_outputs = oracle["output_bytes"]
    assert isinstance(candidate_outputs, dict)
    assert isinstance(oracle_outputs, dict)
    mismatches.extend(
        f"{name}: {_sha256(candidate_outputs[name])} != {_sha256(oracle_outputs[name])}"
        for name in DETERMINISTIC_OUTPUTS
        if candidate_outputs[name] != oracle_outputs[name]
    )
    if mismatches:
        raise AssertionError(
            "candidate deterministic results differ from the legacy oracle:\n"
            + "\n".join(f"- {mismatch}" for mismatch in mismatches)
        )


def _run_reference(
    *,
    dataset: object,
    image: str,
    work_dir: Path,
    case_template: Path | None = None,
) -> dict[str, object]:
    work_dir.mkdir(parents=True, exist_ok=False)
    config: dict[str, object] = {
        "backend": "container",
        "container_image": image,
        "mpi_cores": 1,
        "retain_artifacts": True,
        "retain_on_failure": True,
        "work_dir": str(work_dir),
        **EXPECTED_CONDITIONS,
    }
    if case_template is not None:
        config["case_template"] = str(case_template)
    problem = MTO2D(seed=1, config=config, dataset=dataset)
    runtime_config = problem.config
    assert runtime_config is not None
    if runtime_config.qu_final != EXPECTED_Q:
        raise AssertionError(f"release simulation must use q={EXPECTED_Q}, got {runtime_config.qu_final}")
    design, index = problem.random_design()
    if index != EXPECTED_INDEX:
        raise AssertionError(f"expected train index {EXPECTED_INDEX}, got {index}")

    result = problem.simulate_verbose(design)
    if result.artifacts_path is None:
        raise AssertionError("reference run did not retain its solver artifacts")
    app = Path(result.artifacts_path) / "case" / "app"
    output_bytes = {name: (app / name).read_bytes() for name in DETERMINISTIC_OUTPUTS}
    final_time = _latest_time_directory(app)
    forward_field_bytes = {name: (final_time / name).read_bytes() for name in FINAL_RESULT_FIELDS}
    gamma_bytes = forward_field_bytes["gamma"]
    final_design = read_half_design(final_time / "gamma")
    np.testing.assert_array_equal(final_design, design)
    design_bytes = np.asarray(design, dtype="<f4").tobytes(order="C")
    final_design_bytes = np.asarray(final_design, dtype="<f4").tobytes(order="C")
    design_sha256 = _sha256(design_bytes)
    if design_sha256 != EXPECTED_DESIGN_SHA256:
        raise AssertionError(f"reference design SHA-256 mismatch: {design_sha256}")

    return {
        "artifacts_path": result.artifacts_path,
        "design_sha256": design_sha256,
        "final_design": final_design,
        "final_design_sha256": _sha256(final_design_bytes),
        "forward_field_bytes": forward_field_bytes,
        "forward_field_sha256": {name: _sha256(value) for name, value in forward_field_bytes.items()},
        "gamma_bytes": gamma_bytes,
        "objective_bits": _float64_bits(result.objective_values),
        "objectives": result.objective_values,
        "output_bytes": output_bytes,
        "output_sha256": {name: _sha256(value) for name, value in output_bytes.items()},
        "q": runtime_config.qu_final,
        "source_case_id": EXPECTED_SOURCE_CASE,
        "train_index": index,
    }


def main() -> None:
    """Load the exact design and enforce exact deterministic-output parity."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--dataset", default=MTO2D.dataset_id, help="Hugging Face dataset ID or saved DatasetDict directory"
    )
    parser.add_argument(
        "--case-template",
        type=Path,
        help="Optional prepared host case for --image instead of its bundled template",
    )
    parser.add_argument(
        "--oracle-image",
        help="Optional legacy image whose deterministic output bytes must match the candidate",
    )
    parser.add_argument(
        "--oracle-case-template",
        type=Path,
        help="Prepared host case used only by --oracle-image",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=DEFAULT_GOLDEN,
        help="Trusted exact-output manifest used when --oracle-image is omitted",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        help="Optional new directory in which to retain candidate/oracle run artifacts",
    )
    args = parser.parse_args()
    if args.oracle_case_template is not None and args.oracle_image is None:
        parser.error("--oracle-case-template requires --oracle-image")

    dataset, dataset_label = _load_reference_dataset(args.dataset)
    artifact_context: AbstractContextManager[str]
    if args.artifacts_dir is None:
        artifact_context = tempfile.TemporaryDirectory(prefix="engibench-mto2d-reference-")
    else:
        artifacts_dir = args.artifacts_dir.expanduser().resolve()
        artifacts_dir.mkdir(parents=True, exist_ok=False)
        artifact_context = nullcontext(str(artifacts_dir))
    with artifact_context as temporary:
        root = Path(temporary)
        candidate = _run_reference(
            dataset=dataset,
            image=args.image,
            work_dir=root / "candidate",
            case_template=args.case_template.expanduser().resolve() if args.case_template is not None else None,
        )
        np.testing.assert_array_equal(candidate["objectives"], EXPECTED_OBJECTIVES)

        oracle = None
        if args.oracle_image is not None:
            oracle_case_template = (
                args.oracle_case_template.expanduser().resolve() if args.oracle_case_template is not None else None
            )
            oracle = _run_reference(
                dataset=dataset,
                image=args.oracle_image,
                case_template=oracle_case_template,
                work_dir=root / "oracle",
            )
            _assert_oracle_match(candidate, oracle)
        else:
            _assert_golden_match(candidate, args.golden.expanduser().resolve())

        gamma_bytes = candidate["gamma_bytes"]
        assert isinstance(gamma_bytes, bytes)
        print(
            json.dumps(
                {
                    "bitwise_reference_match": True,
                    "dataset": dataset_label,
                    "gamma_sha256": _sha256(gamma_bytes),
                    "image": args.image,
                    "oracle_image": args.oracle_image,
                    "reference": _reference_record(candidate),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
