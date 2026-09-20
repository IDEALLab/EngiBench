import json
from pathlib import Path

from datasets import Dataset
import numpy as np
import pytest

from engibench.problems.power_electronics.dataset_generation import ContainerIdentity
from engibench.problems.power_electronics.dataset_generation import design_sha256
from engibench.problems.power_electronics.dataset_generation import generate_shard
from engibench.problems.power_electronics.dataset_generation import GitState
from engibench.problems.power_electronics.dataset_generation import resolve_container_identity
from engibench.problems.power_electronics.dataset_generation import select_indices
from engibench.problems.power_electronics.dataset_generation import validate_backend
from engibench.problems.power_electronics.utils.ngspice import NgSpiceIdentity
from engibench.problems.power_electronics.v1 import PowerElectronicsSimulationResult

SIMULATOR_IDENTITY = NgSpiceIdentity(
    version="44.2",
    major_version=44,
    executable_path="/opt/ngspice/bin/ngspice",
    executable_sha256="simulator-digest",
    platform_system="Linux",
    platform_machine="x86_64",
    version_output="ngspice-44.2 : Circuit level simulation program",
)
FAILED_DESIGN_MARKER = 2.0
OUTPUT_VOLTAGE_MEAN = -250.0
DC_GAIN = -0.25
FAILED_V0_DC_GAIN = -0.2
RECORD_COUNT = 2


class FakeProblem:
    simulator_identity = SIMULATOR_IDENTITY

    def __init__(self, *, target_dir: str, ngspice_path: str | None) -> None:
        self.target_dir = target_dir
        self.ngspice_path = ngspice_path
        netlist_path = Path(target_dir) / "source.net"
        netlist_path.parent.mkdir(parents=True, exist_ok=True)
        netlist_path.write_text("test netlist\n")
        self.config = type("Config", (), {"original_netlist_path": str(netlist_path)})()

    def simulate_verbose(self, design: np.ndarray) -> PowerElectronicsSimulationResult:
        if design[0] == FAILED_DESIGN_MARKER:
            raise RuntimeError("deliberate failure")
        return PowerElectronicsSimulationResult(
            objective_values=np.array([0.5, 0.1]),
            output_voltage_mean=-250.0,
            output_voltage_peak_to_peak=25.0,
            dc_gain=-0.25,
            dc_gain_error=0.5,
            relative_voltage_ripple=0.1,
            simulation_valid=True,
            objectives_valid=True,
            status="ok",
            simulator_identity=self.simulator_identity,
        )


def test_select_indices_is_deterministic_and_preserves_source_indices() -> None:
    assert select_indices(10, explicit_indices=None, limit=None, shard_index=1, num_shards=3) == [1, 4, 7]
    assert select_indices(10, explicit_indices=None, limit=2, shard_index=1, num_shards=3) == [1, 4]
    assert select_indices(10, explicit_indices=[8, 2], limit=None, shard_index=0, num_shards=1) == [8, 2]


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"row_count": 10, "explicit_indices": [10], "limit": None, "shard_index": 0, "num_shards": 1}, IndexError),
        ({"row_count": 10, "explicit_indices": [2, 2], "limit": None, "shard_index": 0, "num_shards": 1}, ValueError),
        ({"row_count": 10, "explicit_indices": [2], "limit": None, "shard_index": 1, "num_shards": 2}, ValueError),
    ],
)
def test_select_indices_rejects_ambiguous_or_invalid_selection(kwargs: dict, error_type: type[Exception]) -> None:
    with pytest.raises(error_type):
        select_indices(**kwargs)


def test_design_hash_is_stable_and_order_sensitive() -> None:
    assert design_sha256([1.0, 2.0]) == design_sha256([1.0, 2.0])
    assert design_sha256([1.0, 2.0]) != design_sha256([2.0, 1.0])


def test_container_is_required_unless_explicitly_relaxed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APPTAINER_CONTAINER", raising=False)
    with pytest.raises(RuntimeError, match="pinned Apptainer image"):
        resolve_container_identity(None, allow_uncontainerized=False)
    assert resolve_container_identity(None, allow_uncontainerized=True) == ContainerIdentity(path=None, sha256=None)


def test_canonical_backend_requires_matching_architecture_and_frozen_hashes() -> None:
    container = ContainerIdentity(path="/image.sif", sha256="container-digest")
    assert validate_backend(
        simulator_version="44.2",
        simulator_sha256="simulator-digest",
        simulator_system="Linux",
        simulator_machine="x86_64",
        container_identity=container,
        expected_simulator_sha256="simulator-digest",
        expected_container_sha256="container-digest",
        allow_noncanonical_backend=False,
    )
    with pytest.raises(RuntimeError, match="platform machine 'arm64' is not x86_64"):
        validate_backend(
            simulator_version="44.2",
            simulator_sha256="simulator-digest",
            simulator_system="Darwin",
            simulator_machine="arm64",
            container_identity=container,
            expected_simulator_sha256="simulator-digest",
            expected_container_sha256="container-digest",
            allow_noncanonical_backend=False,
        )


def test_generate_shard_preserves_rows_failures_and_provenance(tmp_path: Path) -> None:
    dataset = Dataset.from_dict(
        {
            "initial_design": [[1.0] * 20, [2.0] * 20],
            "DcGain": [0.1, -0.2],
            "Voltage_Ripple": [0.3, 0.4],
        }
    )
    output_path = tmp_path / "train-00000.jsonl"
    git_state = GitState(commit="engibench-commit", dirty=False)

    manifest = generate_shard(
        dataset=dataset,
        split="train",
        indices=[0, 1],
        output_path=output_path,
        work_dir=tmp_path / "work",
        source_revision="source-revision",
        git_state=git_state,
        container_identity=ContainerIdentity(path="/image.sif", sha256="container-digest"),
        ngspice_path="/ngspice",
        expected_simulator_sha256="simulator-digest",
        expected_container_sha256="container-digest",
        allow_noncanonical_backend=False,
        problem_factory=FakeProblem,
    )

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [row["source_index"] for row in rows] == [0, 1]
    assert rows[0]["output_voltage_mean"] == OUTPUT_VOLTAGE_MEAN
    assert rows[0]["dc_gain"] == DC_GAIN
    assert rows[0]["simulation_status"] == "ok"
    assert rows[0]["source_dataset_revision"] == "source-revision"
    assert rows[0]["engibench_git_commit"] == "engibench-commit"
    assert rows[0]["simulator_version"] == "44.2"
    assert rows[0]["simulator_platform_machine"] == "x86_64"
    assert rows[0]["container_sha256"] == "container-digest"
    assert rows[0]["netlist_sha256"] == manifest["netlist"]["sha256"]
    assert rows[1]["simulation_status"] == "simulation_error"
    assert rows[1]["error_type"] == "RuntimeError"
    assert rows[1]["dc_gain_error"] is None
    assert rows[1]["v0_DcGain"] == FAILED_V0_DC_GAIN

    manifest_path = output_path.with_suffix(".jsonl.manifest.json")
    written_manifest = json.loads(manifest_path.read_text())
    assert written_manifest == manifest
    assert manifest["source"]["selected_indices"] == [0, 1]
    assert manifest["simulator"]["executable_sha256"] == "simulator-digest"
    assert manifest["container"]["sha256"] == "container-digest"
    assert manifest["output"]["record_count"] == RECORD_COUNT
    assert manifest["output"]["status_counts"] == {"ok": 1, "simulation_error": 1}
    original_output = output_path.read_text()
    original_manifest = manifest_path.read_text()

    with pytest.raises(FileExistsError):
        generate_shard(
            dataset=dataset,
            split="train",
            indices=[0],
            output_path=output_path,
            work_dir=tmp_path / "work",
            source_revision="source-revision",
            git_state=git_state,
            container_identity=ContainerIdentity(path="/image.sif", sha256="container-digest"),
            ngspice_path="/ngspice",
            expected_simulator_sha256="simulator-digest",
            expected_container_sha256="container-digest",
            allow_noncanonical_backend=False,
            problem_factory=FakeProblem,
        )
    assert output_path.read_text() == original_output
    assert manifest_path.read_text() == original_manifest
